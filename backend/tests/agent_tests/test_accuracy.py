import pytest

from frontend.chatbot_langgraph import _ask_agent, _execute_sql


class TestExecutionAccuracy:
    """Layer 3 & 7: End-to-end SQL accuracy (Spider-style)."""

    GOLDEN_QUERIES = [
        {"question": "How many projects do we have?", "expected_table": "project"},
        {"question": "List all files from the file table", "expected_table": "file"},
    ]

    @pytest.mark.parametrize("case", GOLDEN_QUERIES)
    def test_golden_query(self, case):
        result = _ask_agent(case["question"], thread_id=f"test_{hash(case['question'])}")
        sql = result.get("sql", "").lower()
        assert case["expected_table"] in sql
        assert result.get("status") == "ok"

    def test_execution_accuracy_sample(self):
        # Ground truth check
        gt_sql = "SELECT COUNT(*) FROM project"
        gt_res = _execute_sql(gt_sql)

        # Agent check
        agent_res = _ask_agent("How many projects?", thread_id="ex_acc_test")
        pred_sql = agent_res.get("sql", "")
        pred_res = _execute_sql(pred_sql)

        # Extract values only to ignore column name differences (e.g., 'count' vs 'total_projects')
        gt_values = [[str(v) for v in r.values()] for r in gt_res["records"]]
        pred_values = [[str(v) for v in r.values()] for r in pred_res["records"]]

        assert gt_values == pred_values, f"Result mismatch: {gt_values} != {pred_values}"
