# 🏗️ End-to-End System Architecture

This diagram illustrates the complete flow of data and control in the Stateful SQL Agent project, following the architectural refactor.

```mermaid
graph TB
    subgraph "Client Layer (React)"
        User((User Query))
        UI["React Chat UI"]
    end

    subgraph "API Layer (FastAPI)"
        App["FastAPI App"]
        Sess["Session Manager"]
    end

    subgraph "LangGraph Agent Engine"
        Intent["Intent (Pydantic)"]
        General["General Handler"]
        Meta["Metadata Handler"]
        Loader["Schema Loader"]
        Generator["SQL Draft (Pydantic)"]
        Guard["SQL Guard (Pydantic)"]
        Executor["SQL Executor (SQLAlchemy)"]
    end

    subgraph "Data & Knowledge Layer"
        DB[("(PostgreSQL)")]
        Metadata["schema_metadata.json"]
        HistoryFile["sessions.json"]
        Env[".env (Config)"]
    end

    subgraph "Summarization (Llama 3.1/3.3)"
        Llama["Llama 3.1 8B / 3.3 70B"]
        Final["Natural Language Response"]
    end

    %% Connections
    User --> UI
    UI -- "SSE Streaming" --> App
    App <--> Sess
    Sess <--> HistoryFile
    
    App --> Intent
    App <--> Env
    
    Intent -- "general" --> General
    Intent -- "metadata" --> Meta
    Intent -- "data" --> Loader
    
    Loader <--> Metadata
    Loader --> Generator
    Generator --> Guard
    Guard --> Executor
    
    Executor <--> DB
    Executor --> Llama
    
    General --> Final
    Meta --> Final
    Llama --> Final
    Final -- "Streaming Output" --> UI
```

### 🧱 Production Architecture Highlights

| Layer | Responsibility | Technology Stack |
| :--- | :--- | :--- |
| **Client** | Real-time UI & SSE | React, Tailwind, Vite |
| **API** | App Logic & Sessions | FastAPI, Pydantic |
| **Reasoning** | **Structured AI** | LangGraph, **Pydantic**, GPT-4o-mini |
| **Validation** | SQL Parsing & Repair | **sqlglot** |
| **Execution** | DB Connectivity | **SQLAlchemy** |
| **Summary** | Professional Responses | **Llama 3.1/3.3** |
| **Storage** | Persistence | Local JSON Files |
