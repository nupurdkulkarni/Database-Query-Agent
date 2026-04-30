import pytest
import time
from backend.chatbot_langgraph import _ask_agent, _execute_sql

class TestExecutionAccuracy:
    """High-level metrics for Text-to-SQL performance."""

    GOLDEN_QUERIES = [
        {"question": "How many projects do we have?", "gt_sql": "SELECT COUNT(*) FROM project"},
        {"question": "List all files", "gt_sql": "SELECT * FROM file"},
    ]

    @pytest.mark.parametrize("case", GOLDEN_QUERIES)
    def test_sql_accuracy_metrics(self, case):
        start_time = time.time()
        
        # 1. Get Agent Response
        result = _ask_agent(case["question"], thread_id=f"acc_test_{time.time()}")
        latency = time.time() - start_time
        
        pred_sql = result.get("sql", "")
        
        # 2. Metric: SQL Validity
        assert pred_sql != "", "Agent failed to generate any SQL"
        assert result.get("status") == "ok", f"Agent returned error: {result.get('error')}"

        # 3. Metric: Execution Accuracy (EX)
        # Run Ground Truth
        gt_res = _execute_sql(case["gt_sql"])
        # Run Predicted
        pred_res = _execute_sql(pred_sql)

        # Compare result values (normalized to strings for robust comparison)
        gt_values = [[str(v) for v in r.values()] for r in gt_res.get("records", [])]
        pred_values = [[str(v) for v in r.values()] for r in pred_res.get("records", [])]

        assert gt_values == pred_values, f"Execution Mismatch! GT: {gt_values}, Pred: {pred_values}"
        
        # 4. Log Latency (will show up in HTML report metadata)
        print(f"\nLatency: {latency:.2f}s | SQL: {pred_sql}")
