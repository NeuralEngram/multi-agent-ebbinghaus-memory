"""
Episodic Memory Store

Persistent, semantically-searchable long-term memory backed by SQLite.
Retention is governed by the Ebbinghaus forgetting curve (see ebbinghaus.py).

4-Signal Dynamic Memory Stability Formula (v2)
───────────────────────────────────────────────
priority_score = retention(t) × (
    0.15 × access_frequency     +
    0.20 × task_importance      +
    0.35 × agent_feedback       +
    0.30 × task_success_rate
)

Signals
───────
- access_frequency   : normalised log(review_count), proxy for how often
                       this memory has been recalled.
- task_importance    : caller-supplied importance weight [0,1].
- agent_feedback     : explicit feedback signal [0,1]; defaults to 0.5
                       (neutral) until updated via update_feedback().
- task_success_rate  : outcome signal [0,1]; defaults to 0.5 (unknown)
                       until updated via update_feedback().

Multiplying by retention() ensures forgotten memories fall to zero
regardless of other signals.
"""

import json
import logging
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


from sentence_transformers import SentenceTransformer

from memory_core.ebbinghaus import (
    BASE_STABLE_HOURS,
    compute_retention,
    is_memory_forgotten,
    reinforce_memory,
    time_until_forgotten,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.30
MIN_RESULTS_REQUIRED  = 1
DEFAULT_LIMIT         = 100
REPLAY_HORIZON_HOURS  = 6
DB_PATH               = "memory.db"

# 4-signal weights (must sum to 1.0)
W_ACCESS_FREQ      = 0.15
W_TASK_IMPORTANCE  = 0.20
W_AGENT_FEEDBACK   = 0.35
W_TASK_SUCCESS     = 0.30

_model = SentenceTransformer("all-MiniLM-L6-v2")


# ── Enums ─────────────────────────────────────────────────────────────────────
class RecallStatus(Enum):
    UPDATED   = "updated"
    FORGOTTEN = "forgotten"
    NOT_FOUND = "not_found"


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def safe_json_loads(s: Any) -> Any:
    try:
        return json.loads(s)
    except (TypeError, ValueError, json.JSONDecodeError):
        return s


def simple_embedding(text: str) -> List[float]:
    return _model.encode(text).tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── Episode dataclass ─────────────────────────────────────────────────────────
@dataclass
class Episode:
    content:  Any
    context:  Dict[str, Any]
    tags:     List[str] = field(default_factory=list)

    stability_hours:  float    = BASE_STABLE_HOURS
    last_reviewed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    review_count:     int      = 0

    episode_id: str      = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    importance:        float = 0.6  # task importance signal
    agent_feedback:    float = 0.5  # 0=negative, 0.5=neutral, 1=positive
    task_success_rate: float = 0.5  # 0=failed, 0.5=unknown, 1=succeeded

    # ── Ebbinghaus retention ──────────────────────────────────────────────────
    def retention(self, now: Optional[datetime] = None) -> float:
        return compute_retention(self.last_reviewed_at, self.stability_hours, now)

    def is_forgotten(self, now: Optional[datetime] = None) -> bool:
        return is_memory_forgotten(self.retention(now))

    # ── Signal helpers ────────────────────────────────────────────────────────
    def access_frequency_score(self) -> float:
        """Normalised access frequency in [0,1] via log smoothing."""
        return min(math.log1p(self.review_count) / math.log1p(100), 1.0)

    # ── 4-Signal Dynamic Memory Stability formula ─────────────────────────────
    def priority_score(self, now: Optional[datetime] = None) -> float:
        """
        S = retention(t) × (
                0.15 × access_frequency   +
                0.20 × task_importance    +
                0.35 × agent_feedback     +
                0.30 × task_success_rate
            )

        Multiplying by retention() ensures forgotten memories score zero.
        """
        weighted = (
            W_ACCESS_FREQ     * self.access_frequency_score()                  +
            W_TASK_IMPORTANCE * max(0.0, min(self.importance,        1.0))     +
            W_AGENT_FEEDBACK  * max(0.0, min(self.agent_feedback,    1.0))     +
            W_TASK_SUCCESS    * max(0.0, min(self.task_success_rate, 1.0))
        )
        return weighted * self.retention(now)


# ── Store ─────────────────────────────────────────────────────────────────────
class EpisodicMemoryStore:
    """
    Persistent episodic memory store with semantic search and Ebbinghaus-based
    retention management, using the 4-signal weighted priority formula.
    """

    def __init__(self):
        self._lock        = threading.RLock()
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        self._init_db()

    # ── DB helpers ────────────────────────────────────────────────────────────
    def _query(self, query: str, params: tuple = ()) -> List[tuple]:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()
        finally:
            conn.close()

    def _execute(self, query: str, params: Any = (), many: bool = False) -> None:
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

    def _init_db(self) -> None:
        # Create table with full schema
        self._execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id         TEXT PRIMARY KEY,
            content            TEXT,
            context            TEXT,
            tags               TEXT,
            stability_hours    REAL,
            last_reviewed_at   TEXT,
            review_count       INTEGER,
            created_at         TEXT,
            importance         REAL,
            embedding          TEXT,
            agent_feedback     REAL DEFAULT 0.5,
            task_success_rate  REAL DEFAULT 0.5
        )
        """)
        # Migrate existing DBs that don't have the new columns yet
        for col, default in [("agent_feedback", 0.5), ("task_success_rate", 0.5)]:
            try:
                self._execute(
                    f"ALTER TABLE episodes ADD COLUMN {col} REAL DEFAULT {default}"
                )
                logger.info("DB migration: added column '%s'.", col)
            except Exception:
                pass  # column already exists — safe to ignore

        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_last_reviewed ON episodes(last_reviewed_at)"
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_importance ON episodes(importance)"
        )

    def _row_to_episode(self, row: tuple) -> Episode:
        return Episode(
            episode_id        = row[0],
            content           = safe_json_loads(row[1]),
            context           = safe_json_loads(row[2]),
            tags              = safe_json_loads(row[3]),
            stability_hours   = row[4],
            last_reviewed_at  = datetime.fromisoformat(row[5]),
            review_count      = row[6],
            created_at        = datetime.fromisoformat(row[7]),
            importance        = row[8],
            # row[9] = embedding — loaded separately
            agent_feedback    = row[10] if len(row) > 10 and row[10] is not None else 0.5,
            task_success_rate = row[11] if len(row) > 11 and row[11] is not None else 0.5,
        )

    def _paginated_scan(self, limit: int, process_page) -> List[Any]:
        offset  = 0
        results = []
        while True:
            rows = self._query(
                "SELECT * FROM episodes LIMIT ? OFFSET ?", (limit, offset)
            )
            if not rows:
                break
            results.extend(process_page(rows))
            offset += limit
        return results

    # ── Public API ────────────────────────────────────────────────────────────
    def add(
        self,
        content:           Any,
        context:           Optional[Dict[str, Any]] = None,
        tags:              Optional[List[str]]       = None,
        importance:        float = 0.6,
        agent_feedback:    float = 0.5,
        task_success_rate: float = 0.5,
    ) -> str:
        """
        Store a new episode.

        Args:
            content:           Memory content (any JSON-serialisable value).
            context:           Metadata dict (e.g. {'user_id': 'alice'}).
            tags:              Optional string tags.
            importance:        Task importance signal [0,1].
            agent_feedback:    Initial feedback signal [0,1] (default 0.5 = neutral).
            task_success_rate: Initial success signal [0,1] (default 0.5 = unknown).

        Returns:
            The new episode_id.
        """
        ep = Episode(
            content           = content,
            context           = context or {},
            tags              = tags or [],
            importance        = max(0.0, min(importance, 1.0)),
            agent_feedback    = max(0.0, min(agent_feedback, 1.0)),
            task_success_rate = max(0.0, min(task_success_rate, 1.0)),
        )
        emb = simple_embedding(str(content))

        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ep.episode_id,
                    safe_json_dumps(ep.content),
                    safe_json_dumps(ep.context),
                    safe_json_dumps(ep.tags),
                    ep.stability_hours,
                    ep.last_reviewed_at.isoformat(),
                    ep.review_count,
                    ep.created_at.isoformat(),
                    ep.importance,
                    safe_json_dumps(emb),
                    ep.agent_feedback,
                    ep.task_success_rate,
                ),
            )
        return ep.episode_id

    def recall(
        self,
        eid:     str,
        quality: float = 1.0,
    ) -> Tuple[Optional["Episode"], RecallStatus]:
        """
        Reinforce a memory after it is recalled.

        Args:
            eid:     Episode ID.
            quality: Recall quality [0,1]. High quality → larger stability boost.

        Returns:
            (Episode, RecallStatus) tuple.
        """
        with self._lock:
            rows = self._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
            if not rows:
                return None, RecallStatus.NOT_FOUND

            row = rows[0]
            ep  = self._row_to_episode(row)
            embedding = row[9]

            if ep.is_forgotten():
                return None, RecallStatus.FORGOTTEN

            ep.stability_hours   = reinforce_memory(ep.stability_hours, quality, ep.review_count)
            ep.last_reviewed_at  = datetime.now(timezone.utc)
            ep.review_count     += 1

            self._execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ep.episode_id,
                    safe_json_dumps(ep.content),
                    safe_json_dumps(ep.context),
                    safe_json_dumps(ep.tags),
                    ep.stability_hours,
                    ep.last_reviewed_at.isoformat(),
                    ep.review_count,
                    ep.created_at.isoformat(),
                    ep.importance,
                    embedding,
                    ep.agent_feedback,
                    ep.task_success_rate,
                ),
            )
            return ep, RecallStatus.UPDATED

    def update_feedback(
        self,
        eid:               str,
        agent_feedback:    Optional[float] = None,
        task_success_rate: Optional[float] = None,
    ) -> bool:
        """
        Update the agent_feedback and/or task_success_rate signals for an episode.

        Call this after a task completes or the user provides explicit feedback:
          - agent_feedback=1.0    → positive/helpful response
          - agent_feedback=0.0    → negative/unhelpful response
          - task_success_rate=1.0 → task completed successfully
          - task_success_rate=0.0 → task failed

        Returns:
            True if episode was found and updated, False otherwise.
        """
        with self._lock:
            rows = self._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
            if not rows:
                return False
            row = rows[0]
            ep  = self._row_to_episode(row)
            embedding = row[9]

            if agent_feedback is not None:
                ep.agent_feedback    = max(0.0, min(agent_feedback, 1.0))
            if task_success_rate is not None:
                ep.task_success_rate = max(0.0, min(task_success_rate, 1.0))

            self._execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ep.episode_id,
                    safe_json_dumps(ep.content),
                    safe_json_dumps(ep.context),
                    safe_json_dumps(ep.tags),
                    ep.stability_hours,
                    ep.last_reviewed_at.isoformat(),
                    ep.review_count,
                    ep.created_at.isoformat(),
                    ep.importance,
                    embedding,
                    ep.agent_feedback,
                    ep.task_success_rate,
                ),
            )
            logger.debug(
                "Feedback updated for %s: feedback=%.2f success=%.2f",
                eid[:8], ep.agent_feedback, ep.task_success_rate,
            )
            return True

    # ── Internal search helpers ───────────────────────────────────────────────
    def _score_rows(
        self,
        query_emb: List[float],
        rows:      List[tuple],
        user_id:   Optional[str],
    ) -> List[Tuple[float, str, "Episode"]]:
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
                scored.append((score, ep.episode_id, ep))
        return scored

    def semantic_search(
        self,
        query:   str,
        top_k:   int = 5,
        user_id: Optional[str] = None,
        limit:   int = DEFAULT_LIMIT,
    ) -> List["Episode"]:
        query_emb  = simple_embedding(query)
        all_scored = self._paginated_scan(
            limit,
            lambda rows: self._score_rows(query_emb, rows, user_id),
        )
        return [ep for _, _, ep in sorted(all_scored, reverse=True)[:top_k]]

    def grounded_retrieve(
        self,
        query:   str,
        top_k:   int = 5,
        user_id: Optional[str] = None,
    ) -> Tuple[List["Episode"], bool]:
        """
        Retrieve episodes and signal whether the result set is trustworthy.

        Returns:
            (episodes, grounded) — grounded=True when avg similarity >= threshold.
        """
        query_emb  = simple_embedding(query)
        all_scored = self._paginated_scan(
            DEFAULT_LIMIT,
            lambda rows: self._score_rows(query_emb, rows, user_id),
        )
        if len(all_scored) < MIN_RESULTS_REQUIRED:
            return [], False

        top       = sorted(all_scored, reverse=True)[:top_k]
        avg_score = sum(s for s, _, _ in top) / len(top)
        episodes  = [ep for _, _, ep in top]
        return episodes, avg_score >= SIMILARITY_THRESHOLD

    def get_top_by_priority(
        self,
        top_k:   int = 5,
        user_id: Optional[str] = None,
    ) -> List["Episode"]:
        """
        Return top_k episodes ranked by the 4-signal priority_score.
        Used as fallback when semantic search yields no results.
        """
        def score_page(rows):
            result = []
            for row in rows:
                ep = self._row_to_episode(row)
                if user_id and ep.context.get("user_id") != user_id:
                    continue
                if not ep.is_forgotten():
                    result.append(ep)
            return result

        all_eps = self._paginated_scan(DEFAULT_LIMIT, score_page)
        all_eps.sort(key=lambda ep: ep.priority_score(), reverse=True)
        return all_eps[:top_k]
    
    def get_by_type(
    self,
    memory_type: str,
    user_id:     Optional[str] = None,
    top_k:       int = 30,
    ) -> List["Episode"]:
        """Return episodes tagged with a specific memory_type (e.g. 'preference').
        Ranked by priority_score, forgotten memories excluded.
        """

        def score_page(rows):
            result = []
            for row in rows:
                ep = self._row_to_episode(row)
                if user_id and ep.context.get("user_id") != user_id:
                    continue
                if ep.context.get("memory_type") != memory_type:
                    continue
                if not ep.is_forgotten():
                    result.append(ep)
            return result

        all_eps = self._paginated_scan(DEFAULT_LIMIT, score_page)
        all_eps.sort(key=lambda ep: ep.priority_score(), reverse=True)
        return all_eps[:top_k]

    def detect_inconsistency(
        self,
        memory_pairs: List[Tuple[Any, List[float]]],
        threshold:    float = 0.8,
    ) -> bool:
        for i in range(len(memory_pairs)):
            for j in range(i + 1, len(memory_pairs)):
                _, a = memory_pairs[i]
                _, b = memory_pairs[j]
                if cosine_similarity(a, b) < threshold:
                    return True
        return False

    def replay_weak(
        self,
        threshold: float = 0.5,
        user_id:   Optional[str] = None,
    ) -> None:
        now  = datetime.now(timezone.utc)
        rows = self._query(
            "SELECT episode_id, last_reviewed_at, stability_hours, context FROM episodes"
        )
        for eid, last, stab, context_raw in rows:
            context = safe_json_loads(context_raw)
            if user_id and context.get("user_id") != user_id:
                continue
            last_dt   = datetime.fromisoformat(last)
            retention = compute_retention(last_dt, stab)
            if is_memory_forgotten(retention):
                continue
            remaining = time_until_forgotten(last_dt, stab, now)
            if remaining < REPLAY_HORIZON_HOURS:
                self.recall(eid)

    def prune_forgotten(self) -> int:
        rows = self._query(
            "SELECT episode_id, last_reviewed_at, stability_hours FROM episodes"
        )
        to_delete = []
        for eid, last, stab in rows:
            last_dt = datetime.fromisoformat(last)
            if is_memory_forgotten(compute_retention(last_dt, stab)):
                to_delete.append((eid,))
        if to_delete:
            self._execute(
                "DELETE FROM episodes WHERE episode_id=?", to_delete, many=True
            )
            logger.info("Pruned %d forgotten episodes.", len(to_delete))
        return len(to_delete)

    # ── Maintenance thread ────────────────────────────────────────────────────
    def start_maintenance(self, interval: int = 60) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()

        def run() -> None:
            while not self._stop_event.is_set():
                try:
                    self.replay_weak()
                    self.prune_forgotten()
                except Exception as exc:
                    logger.error("Maintenance error: %s", exc)
                self._stop_event.wait(interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        logger.info("Maintenance thread started (interval=%ds).", interval)

    def stop_maintenance(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
            logger.info("Maintenance thread stopped.")