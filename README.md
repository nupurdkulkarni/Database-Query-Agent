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

## 🧠 Design Rationale & Justifications

### 1. Why LangGraph?
Traditional linear LLM chains fail when faced with complex database schemas. We use **LangGraph** to implement a stateful directed acyclic graph (DAG) that allows for:
- **Conditional Routing:** Only executing SQL when necessary (avoiding hallucinations).
- **Automated Retries:** The agent can catch SQL errors and "self-correct" by looping back to the repair node.
- **Deterministic Guardrails:** Explicit nodes for validation ensure that no "dangerous" SQL (DROP, DELETE) ever hits the execution engine.

### 2. Multi-Model Pipeline Strategy
To optimize for both **cost** and **intelligence**, the system uses a tiered model approach:
- **Routing & SQL Drafting (GPT-4o-mini):** Fast and highly accurate at following schemas.
- **Conversational Summarization (Llama 3.3 70B):** Higher parameter count for better natural language generation and professional tone.
- This hybrid approach provides GPT-4 level results at a fraction of the token cost.

### 3. Functional Node Architecture
Nodes are implemented as **standalone functions** rather than complex class methods. This ensures:
- **Statelessness:** No hidden internal state; every transformation is visible in the `GraphState`.
- **Concurrency Safety:** Easier to scale horizontally in a production environment.
- **Strict Typing:** Python type hints and Pydantic models ensure data integrity at every step.

### 4. Security First (SQL Guardrails)
Instead of relying on LLM "promises" of safety, we use **sqlglot** for deterministic AST (Abstract Syntax Tree) parsing. Any query that isn't a `SELECT` or `WITH` is rejected before it reaches the database.

### 5. Latency & Token Streaming (SSE)
To solve the "slow agent" problem, we implement **Server-Sent Events (SSE)**.
- **Node-Level Feedback:** The agent emits messages as it enters each node (e.g., "Drafting SQL...", "Executing...").
- **Token-by-Token Streaming:** The final answer from Llama is streamed directly to the React UI as it is generated. 
- This reduces "Perceived Latency" by providing immediate feedback, even though the full graph execution might take several seconds.

### 6. Hybrid Conversational Memory
The system uses a two-tier memory strategy for robust sessions:
- **Short-Term (LangGraph Checkpointers):** Uses `MemorySaver` to track the exact state (intent, variables, retry counts) within a single execution loop.
- **Long-Term (JSON Persistence):** All chat history is mirrored to `data/sessions.json`. This ensures that even if the server restarts or the Docker container is rebuilt, user conversations remain intact.
- **Context Injection:** When a session resumes, the last few turns of history are injected into the Prompt to maintain coherence across different user queries.

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
