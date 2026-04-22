"""
MAEM Demo Seed Script
Run this before your Unisys demo to pre-load perfect data.
Usage: python seed_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_core.episodic_memory import EpisodicMemoryStore
import time

store = EpisodicMemoryStore()

print("=" * 50)
print("MAEM Demo Seed Script")
print("=" * 50)

# ── Alice data (rich, impressive) ─────────────────
print("\n[1/3] Seeding Alice...")

alice_facts = [
    ("My name is Aryan and I am a backend developer from Bangalore", 0.95),
    ("I am building MAEM for my final year project at university", 0.95),
    ("My tech stack is Python FastAPI and SQLite", 0.90),
    ("I prefer Rust over Python for systems programming", 0.85),
    ("My monthly budget is 50000 rupees", 0.80),
    ("I have a dog named Bruno", 0.80),
    ("I am learning Kubernetes this month", 0.75),
    ("I work at a startup focused on AI products", 0.90),
    ("I prefer detailed technical explanations over simple ones", 0.85),
    ("My favorite database is PostgreSQL", 0.80),
]

for content, importance in alice_facts:
    eid = store.add(
        content,
        context={"user_id": "alice", "role": "user"},
        importance=importance,
        agent_feedback=0.8,
        task_success_rate=0.8,
    )
    if eid:
        # Simulate multiple reviews for some memories
        if importance >= 0.90:
            for _ in range(6):
                store.recall(eid, quality=0.9)
        elif importance >= 0.85:
            for _ in range(3):
                store.recall(eid, quality=0.8)
        print(f"  ✓ Stored: {content[:50]}")
    else:
        print(f"  ✗ Skipped (filtered): {content[:50]}")
    time.sleep(0.1)

# ── Bob data (minimal — for silo demo) ────────────
print("\n[2/3] Seeding Bob...")

bob_facts = [
    ("My name is Bob and I am a data scientist", 0.90),
    ("I work with machine learning models daily", 0.85),
    ("My favorite language is R", 0.80),
]

for content, importance in bob_facts:
    eid = store.add(
        content,
        context={"user_id": "bob", "role": "user"},
        importance=importance,
        agent_feedback=0.5,
        task_success_rate=0.5,
    )
    if eid:
        print(f"  ✓ Stored: {content[:50]}")
    time.sleep(0.1)

# ── Charlie data (empty — for clean silo demo) ────
print("\n[3/3] Charlie kept empty for silo isolation demo.")

# ── Verify ────────────────────────────────────────
print("\n" + "=" * 50)
print("Verification")
print("=" * 50)

import sqlite3, json
conn = sqlite3.connect("memory.db")

for user in ["alice", "bob", "charlie"]:
    rows = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE context LIKE ?",
        (f'%"user_id": "{user}"%',)
    ).fetchone()
    print(f"  {user}: {rows[0]} episodes")

# Show Alice top 3 by priority
print("\nAlice top 3 episodes:")
rows = conn.execute("""
    SELECT content, importance, review_count
    FROM episodes
    WHERE context LIKE '%"user_id": "alice"%'
    ORDER BY importance DESC
    LIMIT 3
""").fetchall()

for r in rows:
    print(f"  content: {r[0][:55]}")
    print(f"  importance: {r[1]:.2f} | reviews: {r[2]}")
    print()

conn.close()

print("=" * 50)
print("Demo data ready! Start your server and demo.")
print("=" * 50)