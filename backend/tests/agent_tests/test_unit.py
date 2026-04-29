import pytest

from frontend.chatbot_langgraph import (
    _execute_sql,
    _is_safe_select_sql,
    _validate_sql,
)


class TestSQLGuard:
    """Layer 1: Unit Tests — Deterministic safety components."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM project",
            "SELECT COUNT(*) FROM file",
            "WITH cte AS (SELECT id FROM project) SELECT * FROM cte",
        ],
    )
    def test_safe_queries_pass(self, sql):
        assert _is_safe_select_sql(sql) is True

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE project",
            "DELETE FROM file WHERE id = 1",
            "UPDATE project SET name = 'hacked'",
        ],
    )
    def test_dangerous_queries_blocked(self, sql):
        assert _is_safe_select_sql(sql) is False


class TestSQLValidator:
    def test_valid_select(self):
        result = _validate_sql("SELECT id, name FROM project")
        assert result["valid"] is True

    def test_invalid_syntax(self):
        result = _validate_sql("SELEC name FORM project")
        assert result["valid"] is False


class TestSQLExecutor:
    def test_valid_query_returns_results(self):
        result = _execute_sql("SELECT id, name FROM project LIMIT 3")
        assert result["status"] == "ok"
        assert result["row_count"] > 0
