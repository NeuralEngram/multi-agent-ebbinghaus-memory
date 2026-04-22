---
title: MAEM Multi Agent Ebbinghaus Memory
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# multi-agent-ebbinghaus-memory
Multi-Agent AI Memory System using Ebbinghaus Forgetting Curve


# MemoryOS — Ebbinghaus Memory Agent

A multi-agent memory system that models human forgetting using the Ebbinghaus
retention curve, with semantic search and SQLite persistence.

## Architecture

```
memory_agent/
├── agent.py                    # Chat loop — entry point
├── prototype_ui.html           # Standalone browser prototype
├── requirements.txt
└── memory_core/
    ├── ebbinghaus.py           # Retention formula: R = e^(-t/S)
    ├── working_memory.py       # Short-term context buffer (deque)
    └── episodic_memory.py      # Long-term SQLite store + semantic search
```

## Key Features

| Feature | Implementation |
|---|---|
| Forgetting curve | `ebbinghaus.compute_retention()` — R = e^(−t/S) |
| Memory decay | `replay_weak()` + `prune_forgotten()` background thread |
| Memory priority | `priority_score()` = retention × importance × log(1 + reviews) |
| Agent learning | `reinforce_memory()` with diminishing returns |
| Multi-agent silos | `user_id` filtering in all retrieval paths |
| Persistence | SQLite WAL mode — survives process restarts |
| Semantic search | `sentence-transformers` all-MiniLM-L6-v2 cosine similarity |
| Context budget | `trim_to_token_limit()` keeps working memory within token limits |

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
python agent.py
```

Open `prototype_ui.html` in any browser for the visual demo (no server needed).

## Bug Fixes Applied

1. **`trim_to_token_limit` missing** — added to `WorkingMemory`; was called in
   `agent.py` but never defined, causing an `AttributeError` at runtime.

2. **`grounded_retrieve` return type mismatch** — previously returned
   `(ep, emb)` tuples; callers treated them as plain `Episode` objects,
   causing `AttributeError: 'tuple' has no attribute 'priority_score'`.
   Now returns `List[Episode]`.

3. **Spurious `increment_review` kwarg** — `agent.py` called
   `store.recall(..., increment_review=True)` but the parameter does not
   exist. Removed.

4. **`recall()` capped quality at retention** — `q = min(quality, r)`
   penalised deliberate recall of fading memories. Removed; quality is now
   passed through directly.

5. **`decay_curve_points` sampled 3× past the threshold** — useful range is
   up to the forgetting point, not 3× beyond it. Changed multiplier to 1.5×.

6. **Missing docstring params in `reinforce_memory`** — `review_count`,
   `boost`, and `diminishing_factor` were not documented.