"""
Implements: 

- Ebbinghaus-based retention decay
- Reinforcement via recall
- Replay of weak memories
- Pruning of forgotten memories
- Semantic search (SentenceTransformers)
- Multi-user isolation using context filtering
- SQLite persistence (WAL mode)

"""

import uuid
import json
import sqlite3
import math
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from sentence_transformers import SentenceTransformer

from memory_core.ebbinghaus import (
    BASE_STABLE_HOURS,
    compute_retention,
    reinforce_memory,
    is_memory_forgotten,
)



# GLOBALS
DB_PATH = "memory.db"
_model = SentenceTransformer("all-MiniLM-L6-v2")



# EXCEPTIONS
class EpisodeNotFoundError(Exception):
    """Raised when an episode is not found."""
    pass



# JSON UTILITIES
def safe_json_dumps(obj: Any) -> str:
    """Safely serialize object to JSON."""
    try:
        return json.dumps(obj)
    except Exception:
        return json.dumps(str(obj))


def safe_json_loads(s: str) -> Any:
    """Safely deserialize JSON."""
    try:
        return json.loads(s)
    except Exception:
        return s
    

# DATABASE
def get_connection() -> sqlite3.Connection:
    """Create SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn



# EMBEDDINGS
def simple_embedding(text: str) -> List[float]:
    """Generate semantic embedding using SentenceTransformer."""
    return _model.encode(text).tolist()



# SIMILARITY
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)



# EPISODE MODEL
@dataclass
class Episode:
    """Represents a single episodic memory."""

    content: Any
    context: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    stability_hours: float = BASE_STABLE_HOURS
    last_reviewed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    review_count: int = 0

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    importance: float = 0.6


    def retention(self, now=None) -> float:
        return compute_retention(self.last_reviewed_at, self.stability_hours, now)


    def is_forgotten(self, now=None) -> bool:
        return is_memory_forgotten(self.retention(now))


    def priority_score(self, now=None) -> float:
        r = self.retention(now)
        freq = 1 + math.log1p(self.review_count)
        return r * self.importance * freq


# MEMORY STORE
class EpisodicMemoryStore:
    """
    Persistent episodic memory store.

    Handles:
    - Storage
    - Recall (reinforcement)
    - Replay of weak memories
    - Pruning forgotten memories
    - Semantic retrieval
    - Multi-user filtering
    """



    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
        self._init_db()



    def _init_db(self):
        conn = get_connection()

        conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            content TEXT,
            context TEXT,
            tags TEXT,
            stability_hours REAL,
            last_reviewed_at TEXT,
            review_count INTEGER,
            created_at TEXT,
            importance REAL,
            embedding TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT,
            event_type TEXT,
            timestamp TEXT
        )
        """)

        conn.commit()
        conn.close()



    def _row_to_episode(self, row) -> Episode:
        return Episode(
            episode_id=row[0],
            content=safe_json_loads(row[1]),
            context=safe_json_loads(row[2]),
            tags=safe_json_loads(row[3]),
            stability_hours=row[4],
            last_reviewed_at=datetime.fromisoformat(row[5]),
            review_count=row[6],
            created_at=datetime.fromisoformat(row[7]),
            importance=row[8],
        )
    


    def _log(self, eid: str, event: str):
        conn = get_connection()
        conn.execute(
            "INSERT INTO memory_logs VALUES (NULL, ?, ?, ?)",
            (eid, event, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()




    def _save(self, ep: Episode, emb=None):
        with self._lock:
            conn = get_connection()
            conn.execute("""
            INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ep.episode_id,
                safe_json_dumps(ep.content),
                safe_json_dumps(ep.context),
                safe_json_dumps(ep.tags),
                ep.stability_hours,
                ep.last_reviewed_at.isoformat(),
                ep.review_count,
                ep.created_at.isoformat(),
                ep.importance,
                safe_json_dumps(emb) if emb else None
            ))
            conn.commit()
            conn.close()



    def add(self, content, context=None, tags=None,
            importance=0.6, stability_hours=BASE_STABLE_HOURS):

        importance = max(0.0, min(importance, 1.0))

        ep = Episode(
            content=content,
            context=context or {},
            tags=tags or [],
            importance=importance,
            stability_hours=stability_hours
        )

        emb = simple_embedding(str(content))
        self._save(ep, emb)
        self._log(ep.episode_id, "add")

        return ep.episode_id



    def get(self, eid):
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM episodes WHERE episode_id=?", (eid,)
        ).fetchone()
        conn.close()

        if not row:
            raise EpisodeNotFoundError(f"{eid} not found")

        return self._row_to_episode(row)



    def recall(self, eid, quality=1.0):
        ep = self.get(eid)

        if ep.is_forgotten():
            return None

        r = ep.retention()
        q = min(quality, r)

        ep.stability_hours = reinforce_memory(ep.stability_hours, q)
        ep.last_reviewed_at = datetime.now(timezone.utc)
        ep.review_count += 1

        self._save(ep)
        self._log(eid, "recall")

        return ep
    

    
    def retrieve(self, top_k=5, context_key=None, context_value=None):
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM episodes WHERE importance > 0.1 LIMIT 100"
        ).fetchall()
        conn.close()

        now = datetime.now(timezone.utc)
        episodes = []

        for r in rows:
            ep = self._row_to_episode(r)

            if ep.is_forgotten(now):
                continue

            if context_key and context_value:
                if ep.context.get(context_key) != context_value:
                    continue

            episodes.append(ep)

        return sorted(
            episodes,
            key=lambda x: x.priority_score(now),
            reverse=True
        )[:top_k]



    def semantic_search(self, query: str, top_k: int = 5,
                        user_id: Optional[str] = None):

        query_embedding = simple_embedding(query)

        conn = get_connection()
        rows = conn.execute("SELECT * FROM episodes LIMIT 100").fetchall()
        conn.close()

        scored = []

        for row in rows:
            emb = safe_json_loads(row[9])
            if not emb:
                continue

            ep = self._row_to_episode(row)

            if user_id and ep.context.get("user_id") != user_id:
                continue

            score = cosine_similarity(query_embedding, emb)
            scored.append((score, ep))

        return [ep for _, ep in sorted(scored, reverse=True)[:top_k]]
    


    def replay_weak(self, threshold=0.5, user_id=None):
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM episodes WHERE importance >= 0.5 LIMIT 100"
        ).fetchall()
        conn.close()

        now = datetime.now(timezone.utc)
        replayed = []

        for r in rows:
            ep = self._row_to_episode(r)

            if user_id and ep.context.get("user_id") != user_id:
                continue

            if ep.is_forgotten(now):
                continue

            ret = ep.retention(now)

            if ret < threshold:
                quality = 0.6 + (0.4 * (1 - ret))
                updated = self.recall(ep.episode_id, quality)

                if updated:
                    replayed.append(updated)

        return replayed
    


    def prune_forgotten(self):
        with self._lock:
            conn = get_connection()
            rows = conn.execute(
                "SELECT episode_id, last_reviewed_at, stability_hours FROM episodes"
            ).fetchall()

            to_delete = []

            for eid, last, stab in rows:
                last = datetime.fromisoformat(last)
                if is_memory_forgotten(compute_retention(last, stab)):
                    to_delete.append((eid,))

            if to_delete:
                conn.executemany(
                    "DELETE FROM episodes WHERE episode_id=?",
                    to_delete
                )

                for eid, in to_delete:
                    self._log(eid, "prune")

            conn.commit()
            conn.close()

            return len(to_delete)



    def start_maintenance(self, interval=60):
        with self._lock:
            if self._running:
                return
            self._running = True

        self._stop_event.clear()

        def run():
            while not self._stop_event.is_set():
                try:
                    self.replay_weak()
                    self.prune_forgotten()
                except Exception as e:
                    print("Maintenance error:", e)

                self._stop_event.wait(interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()


    def stop_maintenance(self):
        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=2)