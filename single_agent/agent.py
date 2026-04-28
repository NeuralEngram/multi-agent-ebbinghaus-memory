"""
Agent — memory-augmented chat loop (FINAL)

Gaps Filled
───────────
- #3 Memory Priority: grounded_retrieve results now re-ranked by full 4-signal
  priority_score() before truncating to top_k.
- #4 Agent Learning: dissatisfaction detection drives update_feedback() with
  negative signals; continued engagement gives implicit positive signal.

Updated: Switched from Anthropic Claude to Google Gemini API
"""

import atexit
import logging
import os
import re
import sys

from dotenv import load_dotenv
import google.generativeai as genai

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_core.episodic_memory import EpisodicMemoryStore
from memory_core.working_memory import WorkingMemory, Role

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment!")

genai.configure(api_key=api_key)

# ── Memory initialisation ─────────────────────────────────────────────────────
store = EpisodicMemoryStore()
wm    = WorkingMemory(capacity=10)

store.start_maintenance(interval=60)
atexit.register(store.stop_maintenance)

WM_TOKEN_BUDGET = 2_000

logger.info("Agent initialised.")


# ── Memory Type Classification (at WRITE time) ────────────────────────────────

_PREFERENCE_SIGNALS = re.compile(
    r"\b(like|love|hate|enjoy|prefer|adore|obsessed with|can't stand|"
    r"favorite|favourite|fan of|into|not a fan|dislike|loathe|"
    r"passionate about|keen on|fond of|addicted to|my go-to|"
    r"i('m| am) (really |super |totally )?(into|big on)|"
    r"nothing beats|can't get enough of)\b",
    re.IGNORECASE,
)

_FACT_SIGNALS = re.compile(
    r"\b(i am|i'm|my name is|i work|i study|i live|i'm from|"
    r"i have|i own|i use|i'm a|i am a|i'm \d+|i'm based)\b",
    re.IGNORECASE,
)

_EVENT_SIGNALS = re.compile(
    r"\b(yesterday|today|last (week|month|year)|just|recently|"
    r"i went|i did|i saw|i met|i finished|i started|i tried)\b",
    re.IGNORECASE,
)

_GOAL_SIGNALS = re.compile(
    r"\b(i want|i need|i plan|i'm trying|i hope|i wish|i'd like|"
    r"my goal|working on|i'm building|i'm learning)\b",
    re.IGNORECASE,
)

_HEURISTIC_RULES = [
    ("preference", _PREFERENCE_SIGNALS),
    ("fact",       _FACT_SIGNALS),
    ("event",      _EVENT_SIGNALS),
    ("goal",       _GOAL_SIGNALS),
]

_classification_cache: dict[str, str] = {}


def classify_memory_type(content: str, use_llm_fallback: bool = True) -> str:
    """
    Classify a memory string into one of:
      preference | fact | event | goal | general

    Strategy:
      1. Heuristic regex (fast, free)
      2. LLM fallback for ambiguous cases (accurate, costs tokens)
    """
    content_stripped = content.strip()

    if content_stripped in _classification_cache:
        return _classification_cache[content_stripped]

    # 1. Heuristic pass
    for mem_type, pattern in _HEURISTIC_RULES:
        if pattern.search(content_stripped):
            _classification_cache[content_stripped] = mem_type
            logger.debug("Heuristic classified '%s' → %s", content_stripped[:60], mem_type)
            return mem_type

    # 2. LLM fallback for short/ambiguous statements
    if use_llm_fallback and len(content_stripped) < 300:
        try:
            classifier_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=(
                    "Classify the user statement into exactly one of these memory types: "
                    "preference, fact, event, goal, general.\n"
                    "Reply with only the single word type, nothing else."
                ),
            )
            resp = classifier_model.generate_content(content_stripped)
            mem_type = resp.text.strip().lower()
            if mem_type not in {"preference", "fact", "event", "goal", "general"}:
                mem_type = "general"
            _classification_cache[content_stripped] = mem_type
            logger.debug("LLM classified '%s' → %s", content_stripped[:60], mem_type)
            return mem_type
        except Exception as e:
            logger.warning("LLM classification failed: %s", e)

    _classification_cache[content_stripped] = "general"
    return "general"


# ── Query Intent Detection ────────────────────────────────────────────────────

_PERSONAL_PATTERNS = re.compile(
    r"\b(my|i am|i'm|i have|i like|i prefer|i love|i hate|i work|i study|"
    r"me|mine|myself|my name|my age|my job|my project|what do i|who am i|"
    r"what am i|do i|did i|have i|tell me about me|what did i|remember)\b",
    re.IGNORECASE,
)

_GENERAL_PATTERNS = re.compile(
    r"^(what is|what are|what was|what were|who is|who was|who are|"
    r"how does|how do|how is|how are|why is|why are|why does|why do|"
    r"when did|when was|when is|explain|define|describe|tell me about|"
    r"what's the difference|compare|vs\b|versus)",
    re.IGNORECASE,
)

def is_personal_query(text: str) -> bool:
    if _PERSONAL_PATTERNS.search(text):
        return True
    if _GENERAL_PATTERNS.match(text.strip()):
        return False
    return True


_PREFERENCE_QUERY_PATTERNS = re.compile(
    r"\b("
    r"what (do i|do i usually|i do|i typically|am i into|are my)"
    r"|(my |tell me my |list my )?(likes?|interests?|preferences?|favorites?|favourites?|hobbies?)"
    r"|(things? (i (like|love|enjoy|hate|prefer|can'?t stand))|i('m| am) (into|obsessed|passionate))"
    r"|what (i'?m |am i |i )(into|obsessed|passionate about|keen on|fond of)"
    r")\b",
    re.IGNORECASE,
)

def is_preference_query(text: str) -> bool:
    return bool(_PREFERENCE_QUERY_PATTERNS.search(text))


# ── Dissatisfaction Detection (Gap #4) ───────────────────────────────────────
_DISSATISFACTION_SIGNALS = re.compile(
    r"\b(no|wrong|incorrect|not right|that'?s not|you'?re wrong|"
    r"that'?s wrong|not what i|didn'?t ask|stop|forget that|"
    r"never mind|ignore that|bad answer|not helpful|try again|"
    r"that's incorrect|you misunderstood|not accurate|completely wrong)\b",
    re.IGNORECASE,
)

def is_dissatisfied(text: str) -> bool:
    return bool(_DISSATISFACTION_SIGNALS.search(text))


# ── Chat function ─────────────────────────────────────────────────────────────
def chat(user_message: str, user_id: str = "default") -> str:

    wm.add(Role.USER, user_message)

    # ── Classify and store user message ──────────────────────────────────────
    is_question = user_message.strip().endswith("?")

    if is_question:
        mem_type   = "question"
        importance = 0.2
    else:
        mem_type   = classify_memory_type(user_message)
        importance = {
            "preference": 0.85,
            "fact":       0.80,
            "goal":       0.75,
            "event":      0.65,
            "general":    0.50,
        }.get(mem_type, 0.50)

    store.add(
        user_message,
        context={
            "user_id":     user_id,
            "role":        "user",
            "memory_type": mem_type,
        },
        importance=importance,
    )

    wm.trim_to_token_limit(WM_TOKEN_BUDGET)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    memories = []
    grounded = False
    personal = is_personal_query(user_message)

    if personal:
        if is_preference_query(user_message):
            memories = store.get_by_type(
                memory_type="preference",
                user_id=user_id,
                top_k=30,
            )
            grounded = len(memories) > 0

        else:
            memories, grounded = store.grounded_retrieve(
                user_message,
                top_k=10,
                user_id=user_id,
            )
            memories = sorted(
                memories,
                key=lambda ep: ep.priority_score(),
                reverse=True,
            )[:5]

        memories = [ep for ep in memories if ep.context.get("memory_type") != "question"]

        if not memories:
            memories = store.get_top_by_priority(top_k=5, user_id=user_id)
            grounded = len(memories) > 0

    logger.info(
        "Retrieved %d memories (personal=%s, pref_query=%s)",
        len(memories), personal, is_preference_query(user_message),
    )

    # ── Build system prompt ───────────────────────────────────────────────────
    if grounded and memories:
        memories.sort(key=lambda ep: ep.priority_score(), reverse=True)

        by_type: dict[str, list] = {}
        for ep in memories:
            t = ep.context.get("memory_type", "general")
            by_type.setdefault(t, []).append(ep.content)

        sections = []
        label_map = {
            "preference": "Preferences / Likes / Dislikes",
            "fact":       "Facts about the user",
            "goal":       "Goals / Plans",
            "event":      "Recent events",
            "general":    "Other context",
        }
        for t, label in label_map.items():
            if t in by_type:
                items = "\n".join(f"  - {c}" for c in by_type[t])
                sections.append(f"{label}:\n{items}")

        context_str = "\n\n".join(sections)

        for ep in memories:
            store.recall(ep.episode_id, quality=ep.retention())

        system_prompt = (
            "You are MAEM, a memory-augmented assistant.\n\n"
            "Confirmed information about this user:\n\n"
            f"{context_str}\n\n"
            "Use this to give personalised, accurate responses. "
            "When listing preferences, include ALL relevant items."
        )
    else:
        system_prompt = "You are MAEM, a helpful assistant."

    # ── Call Gemini ───────────────────────────────────────────────────────────
    messages = wm.to_prompt_messages()

    # Build conversation history for Gemini
    gemini_history = []
    for m in messages[:-1]:  # all except last message
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({
            "role": role,
            "parts": [m["content"]]
        })

    chat_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_prompt,
    )

    convo = chat_model.start_chat(history=gemini_history)
    response = convo.send_message(messages[-1]["content"])
    reply = response.text

    wm.add(Role.ASSISTANT, reply)

    # Store assistant reply
    store.add(
        reply,
        context={
            "user_id":     user_id,
            "role":        "assistant",
            "memory_type": "general",
        },
        importance=0.5,
    )

    # ── Gap #4: Agent learning via feedback signals ───────────────────────────
    if memories:
        if is_dissatisfied(user_message):
            for ep in memories:
                store.update_feedback(
                    ep.episode_id,
                    agent_feedback=0.1,
                    task_success_rate=0.2,
                )
            logger.info(
                "Dissatisfaction detected — decayed feedback for %d memories",
                len(memories),
            )
        else:
            for ep in memories:
                store.update_feedback(
                    ep.episode_id,
                    agent_feedback=0.8,
                    task_success_rate=0.8,
                )

    return reply


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user = "alice"
    print("MAEM Ready\n")

    while True:
        msg = input("You: ")
        if msg.lower() == "exit":
            break
        print("Agent:", chat(msg, user))