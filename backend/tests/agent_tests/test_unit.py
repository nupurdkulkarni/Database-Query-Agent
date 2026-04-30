import pytest
import json
from unittest.mock import MagicMock, patch
from backend.chatbot_langgraph import (
    _execute_sql,
    _is_safe_select_sql,
    _clean_history,
    _node_route_query,
    _node_retrieve_schema,
    _node_draft_sql,
    _node_increment_retry,
    _stream_summary_execution,
)

class TestHistoryCleaner:
    def test_clean_history_empty(self):
        assert _clean_history(None) == []
        assert _clean_history([]) == []

    def test_clean_history_full(self):
        raw = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there", "sql": "SELECT 1"}
        ]
        clean = _clean_history(raw)
        assert len(clean) == 1
        assert clean[0]["query"] == "hello"
        assert clean[0]["answer"] == "hi there"
        assert clean[0]["sql"] == "SELECT 1"

class TestSQLGuard:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM project",
            "SELECT COUNT(*) FROM file",
        ],
    )
    def test_safe_queries_pass(self, sql: str) -> None:
        assert _is_safe_select_sql(sql) is True

    def test_dangerous_queries_blocked(self) -> None:
        assert _is_safe_select_sql("DROP TABLE project") is False

class TestNodes:
    @patch("backend.chatbot_langgraph.ChatOpenAI")
    def test_route_query_node(self, mock_chat):
        mock_llm = MagicMock()
        mock_res = MagicMock()
        mock_res.intent = "data"
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_res
        mock_chat.return_value = mock_llm

        state = {"user_query": "projects", "chat_history": []}
        # In the unit test, we mock the node itself to avoid complex GraphState dependency
        # but let's try to mock the internal LLM call instead.
        with patch("backend.chatbot_langgraph._get_intent_llm") as mock_intent_llm:
            mock_intent_llm.return_value.with_structured_output.return_value.invoke.return_value = mock_res
            result = _node_route_query(state)
            assert result["intent"] == "data"

    @patch("backend.chatbot_langgraph._get_schema_context_bundle")
    def test_retrieve_schema_node(self, mock_bundle):
        mock_bundle.return_value = {"text": "schema text", "sources": []}
        state = {"user_query": "projects", "intent": "data"}
        result = _node_retrieve_schema(state)
        assert result["schema_text"] == "schema text"

    @patch("backend.chatbot_langgraph._generate_sql_draft")
    def test_draft_sql_node(self, mock_gen):
        mock_gen.return_value = "SELECT 1"
        state = {"user_query": "query", "schema_text": "schema", "chat_history": []}
        result = _node_draft_sql(state)
        assert result["candidate_sql"] == "SELECT 1"

    def test_increment_retry_node(self):
        state = {"retry_count": 1, "error_history": []}
        result = _node_increment_retry(state)
        assert result["retry_count"] == 2

class TestGenerators:
    @patch("backend.chatbot_langgraph._get_summary_llm")
    def test_stream_summary_execution(self, mock_llm_getter):
        mock_llm = MagicMock()
        mock_llm.stream.return_value = [
            MagicMock(content="The"), MagicMock(content=" result")
        ]
        mock_llm_getter.return_value = mock_llm
        
        # Mock status to "general" to trigger the first branch in _stream_summary_execution
        exec_result = {"status": "general", "records": [], "sql": ""}
        gen = _stream_summary_execution("hello", exec_result)
        tokens = list(gen)
        assert len(tokens) == 2
        # The function yields STRINGS directly
        assert tokens[0] == "The"

class TestSQLExecutor:
    @patch("backend.chatbot_langgraph.create_engine")
    def test_execute_sql_success(self, mock_engine):
        mock_conn = MagicMock()
        mock_row = MagicMock()
        mock_row._asdict.return_value = {"id": 1}
        mock_row.items.return_value = [("id", 1)]
        
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id"]
        mock_result.__iter__.return_value = [mock_row]
        
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        
        result = _execute_sql("SELECT 1")
        assert result["status"] == "ok"
