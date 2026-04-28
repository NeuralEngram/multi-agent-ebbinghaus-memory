import os, sys, logging, traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from memory_core.episodic_memory import EpisodicMemoryStore
from memory_core.working_memory import WorkingMemory, Role

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__)
CORS(app)

store = EpisodicMemoryStore()
store.start_maintenance(interval=60)
wms = {}

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

def get_wm(uid):
    if uid not in wms:
        wms[uid] = WorkingMemory(capacity=10)
    return wms[uid]

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "maem_demo.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/memories")
def get_memories():
    uid = request.args.get("user_id", "default")
    rows = store._query("SELECT * FROM episodes WHERE context LIKE ?", (f'%"user_id": "{uid}"%',))
    eps = []
    for row in rows:
        try:
            ep = store._row_to_episode(row)
            if not ep.is_forgotten():
                eps.append({"id": ep.episode_id, "content": ep.content, "role": ep.context.get("role", "user"),
                            "stability_hours": round(ep.stability_hours, 2), "review_count": ep.review_count,
                            "importance": round(ep.importance, 2), "retention": round(ep.retention(), 4),
                            "priority_score": round(ep.priority_score(), 4)})
        except Exception as e:
            logger.error("Row error: %s", e)
    eps.sort(key=lambda x: x["priority_score"], reverse=True)
    return jsonify({"episodes": eps, "total": len(eps)})

@app.route("/stats")
def get_stats():
    uid = request.args.get("user_id", "default")
    rows = store._query("SELECT * FROM episodes WHERE context LIKE ?", (f'%"user_id": "{uid}"%',))
    eps = []
    for row in rows:
        try:
            eps.append(store._row_to_episode(row))
        except Exception:
            pass
    active = [e for e in eps if not e.is_forgotten()]
    avg = sum(e.retention() for e in active) / len(active) if active else 0
    return jsonify({"total": len(active), "avg_retention": round(avg, 4)})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}
        msg = data.get("message", "").strip()
        uid = data.get("user_id", "default")

        logger.info("Chat request: uid=%s msg=%s", uid, msg[:30])

        if not msg:
            return jsonify({"error": "missing params"}), 400

       

        wm = get_wm(uid)
        wm.add(Role.USER, msg)

        store.add(msg, context={"user_id": uid, "role": "user"},
                  importance=0.2 if msg.endswith("?") else 0.7,
                  agent_feedback=0.5, task_success_rate=0.5)
        wm.trim_to_token_limit(2000)

        mems, grounded = store.grounded_retrieve(msg, top_k=5, user_id=uid)
        mems = [e for e in mems if not e.content.strip().endswith("?")]
        mems = [e for e in mems if e.importance >= 0.15]
        if not mems:
            mems = store.get_top_by_priority(top_k=5, user_id=uid)
            mems = [e for e in mems if not e.content.strip().endswith("?")]
            grounded = len(mems) > 0

        for e in mems:
            store.recall(e.episode_id, quality=e.retention())

        if grounded and mems:
            facts = "\n".join(f"- {e.content}" for e in mems)
            sys_p = f"You are MAEM, a memory-augmented AI.\n\nFACTS from memory:\n{facts}\n\nRULES:\n- Trust current message over stored memory if user corrects something.\n- Say you do not know if memory is empty.\n- Do not expose metadata."
        else:
            sys_p = "You are MAEM. No stored memories yet. Do not invent facts."

        # Gemini API call
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",  
            system_instruction=sys_p
        )
        messages = wm.to_prompt_messages()
        gemini_history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [m["content"]]})

        convo = model.start_chat(history=gemini_history)
        response = convo.send_message(messages[-1]["content"])
        reply = response.text

        wm.add(Role.ASSISTANT, reply)
        store.add(reply, context={"user_id": uid, "role": "assistant"}, importance=0.5,
                  agent_feedback=0.8 if grounded and mems else 0.5,
                  task_success_rate=0.8 if grounded and mems else 0.5)
        logger.info("[%s] retrieved=%d | %s", uid, len(mems), msg[:50])
        return jsonify({"reply": reply, "memory_used": grounded and len(mems) > 0,
                        "retrieved_ids": [e.episode_id for e in mems]})

    except Exception as e:
        logger.error("CHAT ERROR:\n%s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json(force=True, silent=True) or {}
        episode_id = data.get("episode_id")
        score = float(data.get("score", 0.5))
        if not episode_id:
            return jsonify({"error": "missing episode_id"}), 400

        success = store.update_feedback(
            eid=episode_id,
            agent_feedback=score,
            task_success_rate=score
        )
        return jsonify({"ok": success})
    except Exception as e:
        logger.error("FEEDBACK ERROR: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("STATIC_DIR = %s", STATIC_DIR)
    logger.info("HTML exists = %s", os.path.exists(os.path.join(STATIC_DIR, "maem_demo.html")))
    logger.info("Starting on http://0.0.0.0:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
