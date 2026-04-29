import json
import logging
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent")

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    Document = None
    FAISS = None
    OpenAIEmbeddings = None

try:
    from sqlglot import exp, parse
except ImportError as e:
    raise ImportError("sqlglot is required. Install with: pip install sqlglot") from e

try:
    from langsmith import traceable
except ImportError:

    def traceable(*_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator


DB_URL = os.getenv("DB_URL", "")
if not DB_URL:
    raise ValueError("Missing DB_URL. Set it in .env before running Streamlit.")

SCHEMA_METADATA_PATH = os.getenv("SCHEMA_METADATA_PATH", "data/schema_metadata.json")
SCHEMA_RETRIEVAL_MODE = os.getenv("SCHEMA_RETRIEVAL_MODE", "full").strip().lower()
SCHEMA_INDEX_PATH = os.getenv("SCHEMA_INDEX_PATH", "data/faiss_schema_index")

_true_values = {"1", "true", "yes", "on"}
SHOW_TIMINGS = os.getenv("SHOW_TIMINGS", "true").strip().lower() in _true_values
USE_LLM_SUMMARY = os.getenv("USE_LLM_SUMMARY", "true").strip().lower() in _true_values


# ============================================================================
# SQL Extraction & Validation Helpers
# ============================================================================


def _extract_sql_candidate(text_value: str) -> str:
    raw = (text_value or "").strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    raw = re.sub(r"^sql\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return (parts[0] + ";") if parts else raw


def _clean_terminate(text_value: str) -> str:
    cleaned = (text_value or "").strip()
    if cleaned.endswith("TERMINATE."):
        cleaned = cleaned[: -len("TERMINATE.")].rstrip()
    elif cleaned.endswith("TERMINATE"):
        cleaned = cleaned[: -len("TERMINATE")].rstrip()
    return cleaned


def _is_safe_select_sql(query: str) -> bool:
    no_comments = re.sub(r"(--.*)|(/\*[\s\S]*?\*/)", "", query)
    normalized = no_comments.strip().lower()

    if not (normalized.startswith("select") or normalized.startswith("with")):
        return False

    forbidden = re.search(
        r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|comment|merge|call|execute)\b",
        normalized,
    )
    return forbidden is None


# ============================================================================
# Schema Extraction & Caching
# ============================================================================


@lru_cache(maxsize=1)
def _get_schema_rows() -> list[dict[str, Any]]:
    path = Path(SCHEMA_METADATA_PATH)
    if path.is_file():
        with path.open() as f:
            return json.load(f)

    engine = _get_engine()
    inspector = inspect(engine)
    rows: list[dict[str, Any]] = []

    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)

        rows.append(
            {
                "table": table_name,
                "columns": [f"{col['name']} ({col['type']})" for col in columns],
                "foreign_keys": {fk["referred_table"]: fk["constrained_columns"] for fk in foreign_keys},
            }
        )

    try:
        with path.open("w") as f:
            json.dump(rows, f, indent=2)
    except Exception as e:
        logger.error("Failed to save schema metadata: %s", e)

    return rows


# ============================================================================
# Retrieval Logic (Full vs FAISS)
# ============================================================================


def _build_schema_documents(schema_rows: list[dict[str, Any]]) -> list[Any]:
    if Document is None:
        return []
    docs: list[Any] = []
    for row in schema_rows:
        table = str(row.get("table", ""))
        columns = row.get("columns", [])
        foreign_keys = row.get("foreign_keys", {})
        cat_values = row.get("categorical_values", {})

        text_value = (
            f"Table: {table}\n"
            f"Columns: {', '.join(str(c) for c in columns)}\n"
            f"Relationships: {foreign_keys if foreign_keys else 'None'}\n"
            f"Samples: {cat_values if cat_values else 'None'}"
        )
        docs.append(Document(page_content=text_value, metadata={"table": table, "row": row}))
    return docs


@lru_cache(maxsize=1)
def _get_schema_vectorstore() -> Any:
    if Document is None or FAISS is None or OpenAIEmbeddings is None:
        return None
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if Path(SCHEMA_INDEX_PATH).is_dir():
        try:
            return FAISS.load_local(SCHEMA_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error("Failed to load FAISS index: %s", e)

    schema_rows = _get_schema_rows()
    docs = _build_schema_documents(schema_rows)
    if not docs:
        return None

    vectorstore = FAISS.from_documents(docs, embedding_model)
    try:
        vectorstore.save_local(SCHEMA_INDEX_PATH)
    except Exception as e:
        logger.error("Failed to save FAISS index: %s", e)
    return vectorstore


def _select_schema_rows_by_faiss(user_query: str, top_k: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        vs = _get_schema_vectorstore()
        if not vs:
            return ([], [])
        results = vs.similarity_search_with_score(user_query, k=top_k)
        rows = [res[0].metadata["row"] for res in results]
        sources = [
            {"method": "faiss", "table": r.get("table"), "score": float(res[1])}
            for res, r in zip(results, rows, strict=False)
        ]
        return (rows, sources)
    except Exception as e:
        logger.error("FAISS retrieval failed: %s", e)
        return ([], [])


@traceable(name="Schema Context Bundle")
def _get_schema_context_bundle(user_query: str | None = None) -> dict[str, Any]:
    """Switchable retrieval: 'full' for small DBs, 'faiss' for large ones."""
    if SCHEMA_RETRIEVAL_MODE == "faiss" and user_query:
        rows, sources = _select_schema_rows_by_faiss(user_query)
        return {"rows": rows, "text": json.dumps(rows, indent=2), "sources": sources}

    # Default: Full Schema Injection
    schema_data = _get_schema_rows()
    return {
        "rows": schema_data,
        "text": json.dumps(schema_data, indent=2),
        "sources": [{"method": "full_schema", "table": "all"}],
    }


@traceable(name="Schema Metadata Text")
def _get_schema_metadata_text(user_query: str | None = None) -> str:
    bundle = _get_schema_context_bundle(user_query)
    return bundle["text"]


# Cached engine to avoid recreating per-query and reduce latency
@lru_cache(maxsize=1)
def _get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


# ============================================================================
# LLM Clients
# ============================================================================


@lru_cache(maxsize=1)
def _get_intent_llm() -> ChatOpenAI:
    """LLM for intent analysis and routing."""
    return ChatOpenAI(
        model=os.getenv("INTENT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        temperature=0,
    )


@lru_cache(maxsize=1)
def _get_sql_llm() -> ChatOpenAI:
    """LLM for SQL drafting and repair."""
    return ChatOpenAI(
        model=os.getenv("SQL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        temperature=0,
    )


@lru_cache(maxsize=1)
def _get_summary_llm() -> ChatOpenAI:
    """LLM for natural language summarization (Streaming)."""
    api_key = os.getenv("SUMMARY_LLM_API_KEY", os.getenv("OPENAI_API_KEY"))
    base_url = "https://openrouter.ai/api/v1" if api_key and api_key.startswith("sk-or-v1") else None

    return ChatOpenAI(
        model=os.getenv("SUMMARY_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("SUMMARY_LLM_TEMPERATURE", "0.2")),
        streaming=True,
    )


# ============================================================================
# SQL Generation, Validation, Execution
# ============================================================================


class SQLDraft(BaseModel):
    reasoning: str = Field(
        description="Briefly reason step-by-step about table selection, column matches, "
        "and categorical values to ensure they exist in the schema."
    )
    sql_query: str = Field(description="The valid PostgreSQL SELECT query.")


class QueryIntent(BaseModel):
    intent: str = Field(
        description="The category of the user's query: 'data' (needs SQL), "
        "'metadata' (about tables/schema), or 'general' (chat/greetings)."
    )
    reasoning: str = Field(description="Why you chose this intent.")


@traceable(name="SQL Draft Generation", run_type="llm")
def _generate_sql_draft(user_query: str, schema_text: str | None = None, chat_history: list[dict] | None = None) -> str:
    if schema_text is None:
        schema_text = _get_schema_metadata_text(user_query=user_query)

    current_date = datetime.now().strftime("%Y-%m-%d")

    history_str = ""
    if chat_history:
        history_str = "\nPAST CONVERSATION CONTEXT:\n"
        for h in chat_history[-3:]:  # only pass last 3 for context length
            history_str += f"User: {h['query']}\nSQL Generated: {h['sql']}\n\n"
        history_str += (
            "IMPORTANT: The user's new question may refer to the past conversation above. "
            "If it is a follow-up (e.g. 'now filter by X'), modify the past SQL query appropriately.\n"
        )

    system_prompt = f"""
You are a PostgreSQL expert. Your goal is to find the most relevant data.

1. TABLE SELECTION: If the user asks about a specific name/entity (e.g. "Tell me about X"), look for tables that
   represent primary entities (like 'projects' or 'users') before granular ones (like 'requirements').
2. JSONB HANDLING: If a column is (JSONB), you MUST cast it to text to use ILIKE.
   Correct: WHERE title::text ILIKE '%EV Battery%'
   Incorrect: WHERE title ILIKE '%EV Battery%'
3. SCHEMA CONTEXT:

- Use only SELECT or WITH queries.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, MERGE, GRANT, REVOKE, CALL, or EXECUTE.
- Prefer explicit JOIN conditions when needed.

1. REASONING: Before providing the SQL, you MUST reason about the schema, identify relevant tables/columns,
   and cross-reference with categorical_values to ensure exact string matches.
2. SEMANTIC MAPPING: If a user asks for a concept and no table matches that name, identify which columns
   in existing tables likely store that information (e.g., 'document_name' for 'files').
3. BROAD SEARCH: If searching for a specific entity, use wildcards: name::text ILIKE '%EV%Battery%'.
4. JSONB: Always cast JSONB to text: column::text ILIKE '%...%'.
5. TYPO CORRECTION: Users frequently make spelling mistakes. You MUST spell-check the search terms.
6. ENTITY RECOGNITION: Do not rely on capitalization. A lowercased phrase like 'chassis redesign'
   is a project name just like 'Chassis Redesign'.
7. TIME-AWARENESS: Today's date is {current_date}. Use this for relative time calculations.
8. USER PRONOUNS: If the user asks for "my" data, DO NOT filter by "ownerid" or a "users" table.
   Assume they want to see ALL records.
9. SCHEMA QUESTIONS: If the user asks about the structure, do NOT write SQL. Use the schema text below.
""".strip()

    context_prompt = f"{schema_text}\n{history_str}"

    retry_msg = ""

    prompt = f"{system_prompt}\n\n{context_prompt}\n\nUser Query: {user_query}{retry_msg}"

    structured_llm = _get_sql_llm().with_structured_output(SQLDraft)
    response = structured_llm.invoke(prompt)
    sql = response.sql_query if response else "SELECT 1 WHERE 1=0;"

    if not _is_safe_select_sql(sql):
        return "SELECT 1 WHERE 1=0;"

    if not re.search(r"\blimit\s+\d+", sql, flags=re.IGNORECASE):
        sql = sql.rstrip().rstrip(";")

    return sql


@traceable(name="SQL Validation (sqlglot)")
def _validate_sql(sql: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not _is_safe_select_sql(sql):
        errors.append("SQL must be a SELECT or WITH query")
        return {"valid": False, "errors": errors, "warnings": warnings}

    try:
        parsed_list = parse(sql, read="postgres")
    except Exception as e:
        errors.append(f"SQL parse error: {e}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    for stmt in parsed_list:
        if stmt is None:
            continue
        for join_obj in stmt.find_all(exp.Join):
            if join_obj.on is None:
                warnings.append(f"JOIN without ON condition: {join_obj}")

        if any(True for _ in stmt.find_all(exp.Star)):
            star_expressions = list(stmt.find_all(exp.Star))
            if any(not star.expressions for star in star_expressions):
                warnings.append("SELECT * detected - consider specifying columns")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@traceable(name="SQL Repair")
def _repair_sql(bad_sql: str, error_msg: str, user_query: str) -> str:
    """Attempts to fix a faulty SQL query using LLM reasoning."""
    system_prompt: str = """
You are a PostgreSQL expert specializing in SQL repair.
Return ONLY one corrected SQL query.

Rules:
- Use only SELECT or WITH.
- Use only tables/columns from schema context.
- Avoid SELECT *.
- Output SQL only, no explanation.
""".strip()

    prompt = f"""
{system_prompt}

User question:
{user_query}

Previous SQL:
{bad_sql}

Validation issues to fix:
{error_msg}
""".strip()

    structured_llm = _get_sql_llm().with_structured_output(SQLDraft)
    response = structured_llm.invoke(prompt)
    return response.sql_query if response else bad_sql


@traceable(name="SQL Guard & Fix")
def _guard_and_fix_sql(
    user_query: str,
    candidate_sql: str,
    max_attempts: int = 3,
    schema_text: str | None = None,
) -> dict[str, Any]:
    if schema_text is None:
        schema_text = _get_schema_metadata_text(user_query=user_query)

    current_sql = candidate_sql
    last_report: dict[str, Any] = {}

    for _attempt in range(max_attempts):
        report = _validate_sql(current_sql)
        last_report = report

        if report["valid"]:
            return {"status": "valid", "sql": current_sql, "report": report}

        all_issues = report.get("errors", []) + report.get("warnings", [])
        current_sql = _repair_sql(current_sql, "\n".join(all_issues), user_query)

    return {"status": "failed", "sql": current_sql, "report": last_report}


@traceable(name="SQL Execution", run_type="tool")
def _execute_sql(sql_query: str, row_limit: int = 200) -> dict[str, Any]:
    if not _is_safe_select_sql(sql_query):
        return {"status": "error", "error": "Unsafe SQL query", "records": [], "row_count": 0}
    sql_to_run = sql_query.strip()
    has_limit = bool(re.search(r"\blimit\s+\d+", sql_to_run, flags=re.IGNORECASE))
    sql_to_run = sql_to_run.rstrip().rstrip(";")
    if not has_limit:
        sql_to_run = sql_to_run + f" LIMIT {row_limit}"

    def _sanitize_val(v):
        if v is None:
            return None
        if isinstance(v, (int, float, str, bool)):
            return v
        return str(v)

    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql_to_run))
            columns = result.keys()
            rows = result.fetchall()
            records = [{k: _sanitize_val(v) for k, v in zip(columns, row, strict=False)} for row in rows]
            return {"status": "ok", "records": records, "row_count": len(records), "sql": sql_to_run}
    except Exception as e:
        return {"status": "error", "error": str(e), "records": [], "row_count": 0}


# ============================================================================
# Summarization
# ============================================================================


def _extract_final_text(chat_result: Any) -> str:
    summary = getattr(chat_result, "summary", None)
    if isinstance(summary, str) and summary.strip():
        return summary

    history = getattr(chat_result, "chat_history", None)
    if isinstance(history, list) and history:
        last_msg = history[-1]
        if isinstance(last_msg, dict):
            return last_msg.get("content", str(chat_result))

    return str(chat_result)


def _extract_execution_artifacts(chat_result: Any) -> dict[str, Any]:
    history = getattr(chat_result, "chat_history", None)
    if not isinstance(history, list):
        return {"sql": "", "records": [], "row_count": 0, "status": "ok", "error": ""}

    for message in reversed(history):
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        if "records" not in content or "sql" not in content:
            continue

        try:
            parsed = json.loads(content)
        except Exception as e:
            logger.debug("Failed to parse JSON artifact: %s", e)
            continue

        if isinstance(parsed, dict):
            records = parsed.get("records", [])
            if not isinstance(records, list):
                records = []

            return {
                "sql": str(parsed.get("sql", "")),
                "records": records,
                "row_count": int(parsed.get("row_count", 0) or 0),
                "status": parsed.get("status", "ok"),
                "error": parsed.get("error", ""),
            }

    return {"sql": "", "records": [], "row_count": 0, "status": "ok", "error": ""}


def _format_records_as_text(records: list[dict[str, Any]], max_rows: int = 5) -> str:
    if not records:
        return "No records returned."

    preview = records[:max_rows]
    lines: list[str] = []
    for row in preview:
        items = [f"{k}={v}" for k, v in row.items()]
        lines.append(" | ".join(items))

    suffix = ""
    if len(records) > max_rows:
        suffix = f"\n... and {len(records) - max_rows} more rows"

    return "\n".join(lines) + suffix


def _summarize_execution_rule_based(_user_query: str, execution_result: dict[str, Any]) -> str:
    if execution_result.get("status") != "ok":
        return f"Error: {execution_result.get('error', 'Unknown error')}"

    records = execution_result.get("records", [])
    row_count = int(execution_result.get("row_count", len(records) if records else 0))
    if row_count == 0:
        return "No matching records found."

    first_row = records[0] if records else {}
    preferred_order = ["id", "project_id", "name", "project_name", "title", "status", "count", "total", "created_at"]
    selected_keys = [k for k in preferred_order if k in first_row]
    if not selected_keys:
        selected_keys = list(first_row.keys())[:5]

    preview_lines: list[str] = []
    for row in records[:5]:
        preview_lines.append(", ".join(f"{k}={row.get(k)}" for k in selected_keys))

    header = f"Found {row_count} matching rows."
    if row_count == 1:
        header = "Found 1 matching row."

    max_preview_rows = 5
    suffix = ""
    if row_count > max_preview_rows:
        suffix = f"\n... showing first {max_preview_rows} of {row_count} rows"

    return header + "\n" + "\n".join(preview_lines) + suffix


def _build_summary_prompt(user_query: str, execution_result: dict[str, Any]) -> str:
    records = execution_result.get("records", [])
    total_count = execution_result.get("row_count", 0)
    records_preview_text = _format_records_as_text(records, max_rows=10)
    return f"""
You receive JSON output from SQL_Executor.
Write a paragraph that answers the user's question as if explaining to a person.
Do not mention queries or result mechanics.
If names repeat, include IDs so they are clearly distinct.
If more than 10 rows exist, briefly state the total then give a short example.

User question: {user_query}
Total records found in database: {total_count}

Preview of top records:
{records_preview_text}
""".strip()  # noqa: S608


def _stream_summary_execution(user_query: str, execution_result: dict[str, Any]):
    """Stream summary tokens from LLM using ChatOpenAI.stream() for real token-level streaming."""
    status = execution_result.get("status")

    if status == "general":
        prompt = (
            "The user is engaging in general conversation (greetings, thank you, etc.). "
            "Respond naturally and politely. Mention that you are an AI assistant "
            "ready to help them query the database when they are ready.\n\n"
            f"User message: {user_query}"
        )
        try:
            for chunk in _get_summary_llm().stream(prompt):
                if chunk.content:
                    yield chunk.content
            return
        except Exception as e:
            logger.error("Streaming general chat failed: %s", e)
            yield "Hello! I'm your database assistant. How can I help you today?"
            return

    if status == "metadata":
        # Handle schema/metadata queries without SQL records
        schema_info = execution_result.get("schema_info", "")
        prompt = (
            "The user is asking about the database structure. "
            f"Based on this schema metadata, explain the tables and columns: {schema_info}\n\n"
            f"User question: {user_query}"
        )
        try:
            for chunk in _get_summary_llm().stream(prompt):
                if chunk.content:
                    yield chunk.content
            return
        except Exception as e:
            logger.error("Streaming metadata summary failed: %s", e)
            yield "I can help with that. The database contains tables for projects, files, and requirements."
            return

    if status != "ok":
        yield f"Error: {execution_result.get('error', 'Unknown error')}"
        return

    try:
        prompt = _build_summary_prompt(user_query, execution_result)
        for chunk in _get_summary_llm().stream(prompt):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error("Streaming SQL summary failed: %s", e)
        yield _summarize_execution_rule_based(user_query, execution_result)


# ============================================================================
# Agent Execution
# ============================================================================


class GraphState(TypedDict):
    user_query: str
    chat_history: list[dict[str, str]]
    schema_text: str
    schema_sources: list[dict[str, Any]]
    candidate_sql: str
    guard_status: str
    guard_report: dict[str, Any]
    execution_result: dict[str, Any]
    retry_count: int
    intent: str  # "data" | "metadata" | "general"
    t_start: float
    t_schema: float
    t_draft: float
    t_guard: float
    t_execute: float


# ============================================================================
# Graph Nodes
# ============================================================================


def _node_route_query(state: GraphState) -> Dict[str, Any]:
    """Classifies user query into data, metadata, or general conversation."""
    t_start: float = time.perf_counter()
    history_str: str = ""
    if state.get("chat_history"):
        history_str = "\nCONVERSATION HISTORY:\n"
        for h in state["chat_history"][-2:]:
            history_str += f"User: {h.get('query', '')}\nAssistant: {h.get('answer', '')}\n"

    prompt: str = f"""
Analyze the user's query and decide the intent. When in doubt, default to 'data'.
{history_str}
User Query: {state["user_query"]}

INTENT CATEGORIES:
- 'data': User wants to SEE, RETRIEVE, LIST, COUNT, or FILTER actual records.
  Examples: "Show me projects", "Who are the users", "Tell me about the projects".
  When in doubt, pick 'data'.
- 'metadata': ONLY when the user explicitly asks about the database STRUCTURE,
  schema, table definitions, or "what columns exist".
- 'general': Greetings, thank you, or off-topic chat.
"""
    structured_llm = _get_intent_llm().with_structured_output(QueryIntent)
    res = structured_llm.invoke(prompt)
    return {"intent": res.intent if res else "data", "t_start": t_start}


def _node_retrieve_schema(state: GraphState) -> Dict[str, Any]:
    """Retrieves and bundles schema information for context."""
    t_start: float = time.perf_counter()
    bundle: Dict[str, Any] = _get_schema_context_bundle(state["user_query"])
    return {
        "schema_text": bundle["text"],
        "schema_sources": bundle.get("sources", []),
        "t_start": t_start,
        "t_schema": time.perf_counter(),
    }


def _node_draft_sql(state: GraphState) -> Dict[str, Any]:
    """Generates an initial SQL draft based on schema and history."""
    t_start: float = time.perf_counter()
    retry_msg: str = ""
    if state.get("retry_count", 0) > 0 and state.get("candidate_sql"):
        exec_res = state.get("execution_result", {})
        if exec_res.get("status") == "error":
            err = exec_res.get("error", "Unknown error")
            retry_msg = f"\n\nPrevious query failed: {err}"
        else:
            retry_msg = "\n\nPrevious query returned 0 rows. Try an alternative approach."

    sql = _generate_sql_draft(state["user_query"] + retry_msg, state["schema_text"], state.get("chat_history", []))
    return {"candidate_sql": sql, "t_draft": time.perf_counter()}


def _node_guard_sql(state: GraphState) -> dict:
    result = _guard_and_fix_sql(
        state["user_query"], state.get("candidate_sql", ""), schema_text=state.get("schema_text")
    )
    return {
        "candidate_sql": result.get("sql", state.get("candidate_sql", "")),
        "guard_status": result.get("status", "failed"),
        "guard_report": result.get("report", {}),
        "t_guard": time.perf_counter(),
    }


def _node_execute_sql(state: GraphState) -> dict:
    if state.get("guard_status") in ["valid", "fixed"]:
        result = _execute_sql(state.get("candidate_sql", ""))
    else:
        err_msg = str(state.get("guard_report", {}).get("errors", ["Failed to guard SQL"]))
        result = {"status": "error", "error": err_msg, "records": [], "row_count": 0}

    history = list(state.get("chat_history", []))
    if state.get("retry_count", 0) == 0 or result.get("row_count", 0) > 0:
        history.append({"query": state["user_query"], "sql": state.get("candidate_sql", "")})

    return {"execution_result": result, "chat_history": history, "t_execute": time.perf_counter()}


def _node_handle_metadata(_state: GraphState) -> dict:
    bundle = _get_schema_context_bundle()
    return {
        "execution_result": {"status": "metadata", "schema_info": bundle["text"], "records": [], "row_count": 0},
        "candidate_sql": "-- Schema Description",
        "t_execute": time.perf_counter(),
    }


def _node_handle_general(_state: GraphState) -> dict:
    return {
        "execution_result": {"status": "general", "records": [], "row_count": 0},
        "candidate_sql": "",
        "t_execute": time.perf_counter(),
    }


def _node_increment_retry(state: GraphState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_sql_graph() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("intent_analyzer", _node_route_query)
    workflow.add_node("schema_retriever", _node_retrieve_schema)
    workflow.add_node("sql_query_generator", _node_draft_sql)
    workflow.add_node("sql_guardrails", _node_guard_sql)
    workflow.add_node("query_executor", _node_execute_sql)
    workflow.add_node("metadata_handler", _node_handle_metadata)
    workflow.add_node("general_handler", _node_handle_general)
    workflow.add_node("retry_manager", _node_increment_retry)

    def decide_route(state: GraphState):
        intent = state.get("intent")
        if intent == "metadata":
            return "metadata_handler"
        if intent == "general":
            return "general_handler"
        return "schema_retriever"

    def route_after_execute(state: GraphState):
        result = state.get("execution_result", {})
        if state.get("retry_count", 0) < 1 and (result.get("status") == "error" or result.get("row_count") == 0):
            return "increment_retry"
        return END

    workflow.add_edge(START, "intent_analyzer")
    workflow.add_conditional_edges(
        "intent_analyzer",
        decide_route,
        {
            "schema_retriever": "schema_retriever",
            "metadata_handler": "metadata_handler",
            "general_handler": "general_handler",
        },
    )
    workflow.add_edge("schema_retriever", "sql_query_generator")
    workflow.add_edge("sql_query_generator", "sql_guardrails")
    workflow.add_edge("sql_guardrails", "query_executor")
    workflow.add_conditional_edges(
        "query_executor", route_after_execute, {"increment_retry": "retry_manager", END: END}
    )
    workflow.add_edge("retry_manager", "sql_query_generator")

    return workflow


memory = MemorySaver()


class _GraphCache:
    _compiled_graph = None

    @classmethod
    def get(cls):
        if cls._compiled_graph is None:
            cls._compiled_graph = build_sql_graph().compile(checkpointer=memory)
        return cls._compiled_graph


def get_compiled_app():
    return _GraphCache.get()


def _ask_agent(
    user_query: str,
    _team: dict[str, Any] | None = None,
    thread_id: str = "default",
    recovered_history: list | None = None,
) -> dict[str, Any]:
    """Process user query with LangGraph state machine and Pydantic validation."""
    app = get_compiled_app()
    config = {"configurable": {"thread_id": thread_id}}

    # If server restarted, memory is wiped. Recover from json if available.
    state = app.get_state(config)
    if not state.values and recovered_history:
        clean_history = []
        for h in recovered_history:
            if h.get("role") == "user":
                clean_history.append({"query": h.get("content")})
            elif h.get("role") == "assistant" and clean_history:
                clean_history[-1]["answer"] = h.get("content")
                clean_history[-1]["sql"] = h.get("sql", "")
        app.update_state(config, {"chat_history": clean_history})

    initial_state = {"user_query": user_query, "retry_count": 0}
    final_state = app.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})

    execution_result = final_state.get("execution_result", {})
    t_start = final_state.get("t_start", 0)
    t_schema = final_state.get("t_schema", 0)
    t_draft = final_state.get("t_draft", 0)
    t_guard = final_state.get("t_guard", 0)
    t_execute = final_state.get("t_execute", 0)

    timings = {
        "schema_lookup_s": max(0, t_schema - t_start) if t_schema > 0 and t_start > 0 else 0,
        "sql_draft_s": max(0, t_draft - t_schema) if t_draft > 0 and t_schema > 0 else 0,
        "sql_guard_s": max(0, t_guard - t_draft) if t_guard > 0 and t_draft > 0 else 0,
        "sql_execute_s": max(0, t_execute - t_guard) if t_execute > 0 and t_guard > 0 else 0,
        "total_s": max(0, t_execute - t_start) if t_execute > 0 and t_start > 0 else 0,
    }

    return {
        "answer": "",
        "sql": final_state.get("candidate_sql", ""),
        "records": execution_result.get("records", []),
        "row_count": execution_result.get("row_count", 0),
        "sources": final_state.get("schema_sources", []),
        "timings": timings,
        "status": execution_result.get("status", "ok"),
        "error": execution_result.get("error", ""),
        "schema_info": execution_result.get("schema_info", ""),
    }


def _ask_agent_stream(
    user_query: str,
    _team: dict[str, Any] | None = None,
    thread_id: str = "default",
    recovered_history: list | None = None,
):
    """Stream answer with real LLM tokens using ChatOpenAI.stream()."""
    result = _ask_agent(user_query, _team, thread_id=thread_id, recovered_history=recovered_history)
    sql = result.get("sql", "")
    records = result.get("records", [])
    sources = result.get("sources", [])
    timings = result.get("timings", {})
    execution_result = {
        "status": result.get("status", "ok"),
        "error": result.get("error", ""),
        "records": records,
        "row_count": len(records) if records else 0,
        "schema_info": result.get("schema_info", ""),
    }

    # Stream tokens from LLM summary using ChatOpenAI.stream()
    answer = ""
    for token in _stream_summary_execution(user_query, execution_result):
        answer += token
        yield {
            "type": "token",
            "content": token,
        }

    # Yield final metadata after all tokens
    yield {
        "type": "final",
        "answer": answer,
        "sql": sql,
        "records": records,
        "sources": sources,
        "timings": timings,
        "status": result.get("status", "ok"),
        "schema_info": result.get("schema_info", ""),
    }


def _get_thread_history(thread_id: str):
    """Retrieve full chat history for a specific thread from the checkpoint."""
    config = {"configurable": {"thread_id": thread_id}}
    app = get_compiled_app()
    state = app.get_state(config)
    if state and state.values:
        return state.values.get("chat_history", [])
    return []
