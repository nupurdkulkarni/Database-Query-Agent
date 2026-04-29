import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure the parent directory is in sys.path to import from frontend
_current_file = Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from backend.chatbot_langgraph import _ask_agent_stream, _get_thread_history  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

app = FastAPI(title="Database Chatbot API")
SESSIONS_FILE = Path("data/sessions.json")
TITLE_MAX_LEN = 50


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_sessions():
    if SESSIONS_FILE.exists():
        try:
            with SESSIONS_FILE.open() as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load sessions: %s", e)
            return []
    return []


def save_session(thread_id: str, title: str, history: list = None):
    sessions = load_sessions()
    existing = next((s for s in sessions if s["thread_id"] == thread_id), None)
    if not existing:
        title_summary = title[:TITLE_MAX_LEN] + ("..." if len(title) > TITLE_MAX_LEN else "")
        sessions.insert(
            0,
            {
                "thread_id": thread_id,
                "title": title_summary,
                "created_at": datetime.now().isoformat(),
                "history": history or [],
            },
        )
    elif history:
        existing["history"] = history

    with SESSIONS_FILE.open("w") as f:
        json.dump(sessions, f, indent=2, default=str)


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"


@app.get("/api/sessions")
def get_sessions():
    return load_sessions()


@app.get("/api/sessions/{thread_id}/history")
def get_history(thread_id: str):
    sessions = load_sessions()
    session = next((s for s in sessions if s["thread_id"] == thread_id), None)
    if session and session.get("history"):
        return session["history"]

    # Fallback to in-memory graph checkpointer
    return _get_thread_history(thread_id)


@app.delete("/api/sessions/{thread_id}")
def delete_session(thread_id: str):
    sessions = load_sessions()
    sessions = [s for s in sessions if s["thread_id"] != thread_id]
    with SESSIONS_FILE.open("w") as f:
        json.dump(sessions, f, indent=2, default=str)
    return {"status": "success"}


@app.post("/api/chat")
def chat_stream(request: ChatRequest):
    """
    Stream the agent execution using Server-Sent Events (SSE).
    """
    # Track session on first message
    save_session(request.thread_id, request.query)

    def event_generator():
        final_event = None
        try:
            # Fetch existing history to hydrate LangGraph if needed
            sessions = load_sessions()
            session = next((s for s in sessions if s["thread_id"] == request.thread_id), None)
            recovered_history = session.get("history", []) if session else []

            agent_gen = _ask_agent_stream(
                request.query, thread_id=request.thread_id, recovered_history=recovered_history
            )

            for event in agent_gen:
                if event.get("type") == "final":
                    final_event = event
                yield f"data: {json.dumps(event, default=str)}\n\n"

            # Persist RICH history to file after stream completes
            if final_event:
                sessions = load_sessions()
                session = next((s for s in sessions if s["thread_id"] == request.thread_id), None)
                if session:
                    if "history" not in session:
                        session["history"] = []

                    # Store user message
                    session["history"].append({"role": "user", "content": request.query})

                    # Store rich assistant message
                    session["history"].append(
                        {
                            "role": "assistant",
                            "content": final_event.get("answer"),
                            "sql": final_event.get("sql"),
                            "records": final_event.get("records"),
                            "sources": final_event.get("sources"),
                            "timings": final_event.get("timings"),
                            "status": final_event.get("status"),
                            "schema_info": final_event.get("schema_info"),
                        }
                    )

                    with SESSIONS_FILE.open("w") as f:
                        json.dump(sessions, f, indent=2)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    # reload=False keeps a single process so load_dotenv() runs before LangChain
    # imports at startup, which is required for LangSmith tracing to initialize.
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=False)
