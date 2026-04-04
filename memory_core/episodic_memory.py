import uuid
import json
import sqlite3
import math
import threading
import logging
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer

from memory_core.ebbinghaus import (
    BASE_STABLE_HOURS,
    compute_retention,
    reinforce_memory,
    is_memory_forgotten,
    time_until_forgotten
)

SIMILARITY_THRESHOLD = 0.75
MIN_RESULTS_REQUIRED = 1
DEFAULT_LIMIT = 100

# ✅ fixed: no magic number
REPLAY_HORIZON_HOURS = 6

DB_PATH = "memory.db"
_model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)


class RecallStatus(Enum):
    UPDATED = "updated"
    FORGOTTEN = "forgotten"
    NOT_FOUND = "not_found"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def safe_json_dumps(obj):
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def safe_json_loads(s):
    try:
        return json.loads(s)
    except (TypeError, ValueError, json.JSONDecodeError):
        return s


def simple_embedding(text: str):
    return _model.encode(text).tolist()


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Episode:
    content: Any
    context: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    stability_hours: float = BASE_STABLE_HOURS
    last_reviewed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    review_count: int = 0

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    importance: float = 0.6

    def retention(self, now=None):
        return compute_retention(self.last_reviewed_at, self.stability_hours, now)

    def is_forgotten(self, now=None):
        return is_memory_forgotten(self.retention(now))

    def priority_score(self, now=None):
        r = self.retention(now)
        freq = 1 + math.log1p(self.review_count)
        return r * self.importance * freq


class EpisodicMemoryStore:

    def __init__(self):
        self._lock = threading.RLock()

        self._running = False
        self._thread = None
        self._stop_event = threading.Event()

        self._init_db()

    def _query(self, query, params=()):
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()
        finally:
            conn.close()

    def _execute(self, query, params=(), many=False):
        conn = get_connection()
        try:
            cur = conn.cursor()
            if many:
                cur.executemany(query, params)
            else:
                cur.execute(query, params)
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        self._execute("""
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

        # ✅ indexing (kept minimal, no redesign)
        self._execute("CREATE INDEX IF NOT EXISTS idx_last_reviewed ON episodes(last_reviewed_at)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_importance ON episodes(importance)")

    def _row_to_episode(self, row):
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

    def _paginated_scan(self, limit, process_page):
        offset = 0
        results = []

        while True:
            rows = self._query(
                "SELECT * FROM episodes LIMIT ? OFFSET ?",
                (limit, offset)
            )

            if not rows:
                break

            results.extend(process_page(rows))
            offset += limit

        return results

    def add(self, content, context=None, tags=None, importance=0.6):
        ep = Episode(
            content=content,
            context=context or {},
            tags=tags or [],
            importance=max(0.0, min(importance, 1.0))
        )

        emb = simple_embedding(str(content))

        with self._lock:
            self._execute("""
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
                safe_json_dumps(emb)
            ))

        return ep.episode_id

    def recall(self, eid, quality=1.0) -> Tuple[Optional[Episode], RecallStatus]:
        with self._lock:
            rows = self._query(
                "SELECT * FROM episodes WHERE episode_id=?",
                (eid,)
            )
            if not rows:
                return None, RecallStatus.NOT_FOUND

            row = rows[0]
            ep = self._row_to_episode(row)
            embedding = row[9]

            if ep.is_forgotten():
                return None, RecallStatus.FORGOTTEN

            r = ep.retention()
            q = min(quality, r)

            # ✅ diminishing returns ACTIVE
            ep.stability_hours = reinforce_memory(
                ep.stability_hours,
                q,
                ep.review_count
            )

            ep.last_reviewed_at = datetime.now(timezone.utc)
            ep.review_count += 1

            self._execute("""
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
                embedding
            ))

            return ep, RecallStatus.UPDATED

    def _score_rows(self, query_emb, rows, user_id):
        scored = []
        for row in rows:
            emb = safe_json_loads(row[9])
            if not emb:
                continue

            ep = self._row_to_episode(row)

            if user_id and ep.context.get("user_id") != user_id:
                continue

            score = cosine_similarity(query_emb, emb)

            if score >= SIMILARITY_THRESHOLD:
                scored.append((score, ep, emb))

        return scored

    def semantic_search(self, query, top_k=5, user_id=None, limit=DEFAULT_LIMIT):
        query_emb = simple_embedding(query)

        all_scored = self._paginated_scan(
            limit,
            lambda rows: self._score_rows(query_emb, rows, user_id)
        )

        return [ep for _, ep, _ in sorted(all_scored, reverse=True)[:top_k]]

    def grounded_retrieve(self, query, top_k=5, user_id=None):
        query_emb = simple_embedding(query)

        all_scored = self._paginated_scan(
            DEFAULT_LIMIT,
            lambda rows: self._score_rows(query_emb, rows, user_id)
        )

        if len(all_scored) < MIN_RESULTS_REQUIRED:
            return [], False

        all_scored = sorted(all_scored, reverse=True)[:top_k]
        avg_score = sum(s for s, _, _ in all_scored) / len(all_scored)

        return [(ep, emb) for _, ep, emb in all_scored], avg_score >= SIMILARITY_THRESHOLD

    def detect_inconsistency(self, memory_pairs, threshold=0.8):
        """
        NOTE:
        Detects semantic dissimilarity, NOT factual contradiction.
        Embeddings cannot detect true conflicts.
        """
        for i in range(len(memory_pairs)):
            for j in range(i + 1, len(memory_pairs)):
                _, a = memory_pairs[i]
                _, b = memory_pairs[j]

                if cosine_similarity(a, b) < threshold:
                    return True
        return False

    def replay_weak(self, threshold=0.5, user_id=None):
        """
        NOTE:
        Full table scan by design.

        Ebbinghaus retention cannot be pushed into SQL easily,
        so filtering must happen in Python.
        """

        now = datetime.now(timezone.utc)

        rows = self._query(
            "SELECT episode_id, last_reviewed_at, stability_hours, context FROM episodes"
        )

        for eid, last, stab, context in rows:
            context = safe_json_loads(context)

            if user_id and context.get("user_id") != user_id:
                continue

            last_dt = datetime.fromisoformat(last)

            retention = compute_retention(last_dt, stab)

            if is_memory_forgotten(retention):
                continue

            remaining = time_until_forgotten(last_dt, stab, now)

            if remaining < REPLAY_HORIZON_HOURS:
                self.recall(eid)

    def prune_forgotten(self):
        rows = self._query(
            "SELECT episode_id, last_reviewed_at, stability_hours FROM episodes"
        )

        to_delete = []

        for eid, last, stab in rows:
            last_dt = datetime.fromisoformat(last)
            retention = compute_retention(last_dt, stab)

            if is_memory_forgotten(retention):
                to_delete.append((eid,))

        if to_delete:
            self._execute(
                "DELETE FROM episodes WHERE episode_id=?",
                to_delete,
                many=True
            )

        return len(to_delete)

    def start_maintenance(self, interval=60):
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
                    logger.error("Maintenance error: %s", e)

                self._stop_event.wait(interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop_maintenance(self):
        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=2)