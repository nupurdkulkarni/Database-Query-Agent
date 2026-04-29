# 🤖 Agentic SQL Chatbot (Stateful & Modular)

A production-grade, modular SQL Agent built with **LangGraph**, **FastAPI**, and **React**. This system translates natural language into secure SQL, executes it against PostgreSQL, and provides professional insights using a hybrid LLM pipeline.

![System Architecture](./images/system_architecture_upd.png)

## 🏗️ Architecture Overview

The system is designed with a "Deterministic Reasoning" philosophy, separating core logic from conversation and summarization.

- **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) state machine with explicit routing.
- **Reasoning Engine:** **GPT-4o-mini** for intent analysis and SQL generation (via Pydantic structured output).
- **Validation Layer:** **sqlglot** for query parsing, safety guarding, and automated repair.
- **Execution Layer:** **SQLAlchemy** for secure database interaction.
- **Summarization Layer:** **Llama 3.1 8B / 3.3 70B** for natural language responses.
- **Persistence:** Local JSON session management for stateful chat history.

---

## 🚀 Getting Started

### 1. Environment Configuration
Create a `.env` file in the root directory (use `docker/.env.example` as a template):
```bash
OPENAI_API_KEY=your_key_here
DB_URL=postgresql://user:pass@host:port/dbname
SUMMARY_LLM_API_KEY=your_openrouter_key_here
```

### 📦 Option A: Running with Docker (Recommended)
The easiest way to get started is using Docker Compose, which handles the frontend, backend, and database orchestration.

```bash
# Build and start the services
docker compose -f docker/docker-compose.yml up --build
```
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000

---

### 💻 Option B: Local Development
If you want to run the components individually:

#### Prerequisites
- Python 3.10+
- Node.js 18+
- A running PostgreSQL instance

#### Backend Setup
```bash
# Install dependencies
pip install -r docker/requirements.prod.txt

# Start the FastAPI server
python3 backend/fastapi_app.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | React 19, Tailwind CSS, Vite |
| **Backend** | FastAPI, Python 3.11 |
| **AI Graph** | LangGraph, Pydantic |
| **Database** | PostgreSQL, **SQLAlchemy** |
| **Safety** | **sqlglot** (SQL Guardrails) |
| **Models** | GPT-4o-mini, Llama 3.1/3.3 |

## 🧪 Development & Quality
This project enforces high code quality via a unified linting gate:
```bash
# Run all linters (Ruff & ESLint)
./lint.sh
```

---

## 📂 Project Structure
- `backend/`: Core LangGraph agent and FastAPI application.
- `frontend/`: React chat interface with SSE streaming support.
- `data/`: Persistent storage for sessions and schema metadata.
- `docker/`: Deployment configurations and environment templates.
- `scripts/`: Utility scripts for graph and diagram generation.
