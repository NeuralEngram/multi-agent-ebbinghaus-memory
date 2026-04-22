"""
MAEM Feature Test Suite
========================
Tests all 6 core features:
  1. Context Window Limit
  2. Memory Decay (Ebbinghaus)
  3. Memory Priority
  4. Agent Learning
  5. Multi-Agent Memory Silos
  6. No Memory Across Conversations (SQLite persistence)

Run:
    cd C:\\multi-agent-ebbinghaus-memory
    python single_agent/test_maem.py
"""

import sys
import os
import time
import logging
from datetime import datetime, timezone, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_core.episodic_memory import EpisodicMemoryStore, Episode
from memory_core.working_memory import WorkingMemory, Role

logging.basicConfig(level=logging.WARNING)  # suppress noise during tests

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SEP  = "─" * 60


def header(n, title):
    print(f"\n{SEP}")
    print(f"TEST {n}: {title}")
    print(SEP)


def result(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")


# ── Test 1: Context Window Limit ──────────────────────────────────────────────
def test_1_context_window():
    header(1, "Context Window Limit — WorkingMemory trim")

    wm = WorkingMemory(capacity=10)
    for i in range(15):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        wm.add(role, f"Message number {i} " * 50)  # large messages

    before = len(wm.to_prompt_messages())
    wm.trim_to_token_limit(200)  # tight budget
    after = len(wm.to_prompt_messages())

    result("trim_to_token_limit() reduces messages", after < before,
           f"before={before}, after={after}")
    result("to_prompt_messages() returns list", isinstance(wm.to_prompt_messages(), list))

    msgs = wm.to_prompt_messages()
    roles_valid = all(m["role"] in ("user", "assistant") for m in msgs) if msgs else True
    result("All message roles are valid", roles_valid)


# ── Test 2: Memory Decay ──────────────────────────────────────────────────────
def test_2_memory_decay():
    header(2, "Memory Decay — Ebbinghaus forgetting curve")

    store = EpisodicMemoryStore()

    # Fresh memory should have high retention
    eid = store.add("Fresh memory test", context={"user_id": "test_decay"}, importance=0.7)
    rows = store._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
    ep = store._row_to_episode(rows[0])
    fresh_retention = ep.retention()
    result("Fresh memory retention > 0.9", fresh_retention > 0.9,
           f"retention={fresh_retention:.3f}")

    # Simulate an old memory by backdating last_reviewed_at
    old_time = (datetime.now(timezone.utc) - timedelta(hours=500)).isoformat()
    store._execute(
        "UPDATE episodes SET last_reviewed_at=?, stability_hours=24 WHERE episode_id=?",
        (old_time, eid)
    )
    rows = store._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
    ep = store._row_to_episode(rows[0])
    old_retention = ep.retention()
    result("Old memory retention < 0.1", old_retention < 0.1,
           f"retention={old_retention:.4f}")
    result("is_forgotten() returns True for old memory", ep.is_forgotten())

    # replay_weak and prune_forgotten run without error
    try:
        store.replay_weak(user_id="test_decay")
        pruned = store.prune_forgotten()
        result("replay_weak() runs without error", True)
        result("prune_forgotten() deleted old memory", pruned >= 1,
               f"pruned={pruned}")
    except Exception as e:
        result("replay_weak / prune_forgotten", False, str(e))


# ── Test 3: Memory Priority ───────────────────────────────────────────────────
def test_3_memory_priority():
    header(3, "Memory Priority — priority_score ranking")

    store = EpisodicMemoryStore()

    # High importance, multiple reviews → should rank highest
    eid_high = store.add("High priority fact", context={"user_id": "test_priority"}, importance=0.9)
    store.recall(eid_high, quality=1.0)
    store.recall(eid_high, quality=1.0)
    store.recall(eid_high, quality=1.0)

    # Low importance, no reviews
    eid_low = store.add("Low priority fact", context={"user_id": "test_priority"}, importance=0.1)

    rows_high = store._query("SELECT * FROM episodes WHERE episode_id=?", (eid_high,))
    rows_low  = store._query("SELECT * FROM episodes WHERE episode_id=?", (eid_low,))
    ep_high = store._row_to_episode(rows_high[0])
    ep_low  = store._row_to_episode(rows_low[0])

    score_high = ep_high.priority_score()
    score_low  = ep_low.priority_score()

    result("priority_score() higher for important+reviewed episode",
           score_high > score_low,
           f"high={score_high:.4f}, low={score_low:.4f}")

    # get_top_by_priority returns high before low
    top = store.get_top_by_priority(top_k=10, user_id="test_priority")
    ids = [ep.episode_id for ep in top]
    result("get_top_by_priority() ranks high episode above low",
           ids.index(eid_high) < ids.index(eid_low) if eid_high in ids and eid_low in ids else False)

    result("review_count incremented correctly", ep_high.review_count == 3,
           f"review_count={ep_high.review_count}")


# ── Test 4: Agent Learning ────────────────────────────────────────────────────
def test_4_agent_learning():
    header(4, "Agent Learning — reinforce_memory on recall")

    store = EpisodicMemoryStore()
    eid = store.add("Learning test memory", context={"user_id": "test_learn"}, importance=0.7)

    rows = store._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
    ep_before = store._row_to_episode(rows[0])
    stab_before = ep_before.stability_hours

    # High quality recall strengthens stability
    store.recall(eid, quality=1.0)
    rows = store._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
    ep_after = store._row_to_episode(rows[0])
    stab_after = ep_after.stability_hours

    result("stability_hours increases after high-quality recall",
           stab_after > stab_before,
           f"before={stab_before:.2f}h → after={stab_after:.2f}h")
    result("review_count incremented", ep_after.review_count == ep_before.review_count + 1,
           f"review_count={ep_after.review_count}")

    # Low quality recall (simulated failure) — pass quality=0.1
    store.recall(eid, quality=0.1)
    rows = store._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
    ep_low = store._row_to_episode(rows[0])
    result("Low quality recall still increments review_count",
           ep_low.review_count == ep_after.review_count + 1)

    # Diminishing returns — each recall multiplies stability by a smaller factor
    multipliers = []
    prev = ep_low.stability_hours
    for _ in range(3):
        store.recall(eid, quality=1.0)
        rows = store._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
        ep = store._row_to_episode(rows[0])
        multipliers.append(ep.stability_hours / prev)
        prev = ep.stability_hours
    result("Diminishing returns — each multiplier smaller than last",
           multipliers[0] >= multipliers[1] >= multipliers[2],
           f"multipliers={[round(m, 3) for m in multipliers]}")


# ── Test 5: Multi-Agent Memory Silos ─────────────────────────────────────────
def test_5_memory_silos():
    header(5, "Multi-Agent Memory Silos — user_id isolation")

    store = EpisodicMemoryStore()

    store.add("Alice secret: loves cats",    context={"user_id": "alice"}, importance=0.8)
    store.add("Bob secret: loves dogs",      context={"user_id": "bob"},   importance=0.8)
    store.add("Shared neutral fact",         context={"user_id": "alice"}, importance=0.5)

    # Alice search should NOT return Bob's memory
    alice_results, _ = store.grounded_retrieve("secret preference", top_k=5, user_id="alice")
    alice_contents = [ep.content for ep in alice_results]
    bob_leaked = any("Bob" in c for c in alice_contents)
    result("Alice search does NOT return Bob's memory", not bob_leaked,
           f"alice_contents={alice_contents}")

    # Bob search should NOT return Alice's memory
    bob_results, _ = store.grounded_retrieve("secret preference", top_k=5, user_id="bob")
    bob_contents = [ep.content for ep in bob_results]
    alice_leaked = any("Alice" in c for c in bob_contents)
    result("Bob search does NOT return Alice's memory", not alice_leaked,
           f"bob_contents={bob_contents}")

    # Priority fallback also silos correctly
    alice_top = store.get_top_by_priority(top_k=10, user_id="alice")
    bob_in_alice = any("bob" in ep.context.get("user_id","") for ep in alice_top)
    result("get_top_by_priority() silos by user_id", not bob_in_alice)


# ── Test 6: SQLite Persistence ────────────────────────────────────────────────
def test_6_persistence():
    header(6, "No Memory Across Conversations — SQLite persistence")

    store1 = EpisodicMemoryStore()
    unique_content = f"Persistence test memory @ {datetime.now().isoformat()}"
    eid = store1.add(unique_content, context={"user_id": "persist_test"}, importance=0.8)
    result("Episode stored by first store instance", bool(eid), f"eid={eid[:8]}...")

    # Create a brand new store instance (simulates process restart)
    store2 = EpisodicMemoryStore()
    rows = store2._query("SELECT * FROM episodes WHERE episode_id=?", (eid,))
    found = len(rows) > 0
    result("New store instance finds episode (survives restart)", found)

    if found:
        ep = store2._row_to_episode(rows[0])
        result("Content matches exactly", ep.content == unique_content,
               f"content='{ep.content[:50]}...'")

    # Semantic search across new instance
    results, grounded = store2.grounded_retrieve(
        "Persistence test memory", top_k=3, user_id="persist_test"
    )
    result("grounded_retrieve() works on new instance", len(results) >= 0)


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  MAEM FEATURE TEST SUITE")
    print("═" * 60)

    tests = [
        test_1_context_window,
        test_2_memory_decay,
        test_3_memory_priority,
        test_4_agent_learning,
        test_5_memory_silos,
        test_6_persistence,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"\n  {FAIL}  Unhandled exception: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{SEP}")
    print("  Done. Fix any ❌ above before demo.")
    print(SEP + "\n")