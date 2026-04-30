import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.fastapi_app import app

client = TestClient(app)

class TestAPIEndpoints:
    """Layer 6: INTEGRATION — FastAPI endpoint coverage using TestClient."""

    def test_sessions_endpoint(self) -> None:
        response = client.get("/api/sessions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @patch("backend.fastapi_app._ask_agent_stream")
    def test_chat_endpoint_streams(self, mock_stream) -> None:
        """Test the chat endpoint with a mocked stream."""
        def mock_gen():
            yield {"type": "info", "content": "test status"}
            yield {"type": "token", "content": "hello"}
            yield {"type": "final", "answer": "hello", "sql": "SELECT 1"}
        
        mock_stream.return_value = mock_gen()

        response = client.post(
            "/api/chat",
            json={"query": "How many projects?", "thread_id": "api_test"},
        )
        assert response.status_code == 200
        
        # In TestClient, iter_lines() returns strings, not bytes
        found = False
        for line in response.iter_lines():
            if "test status" in line:
                found = True
                break
        assert found

    def test_history_endpoint(self) -> None:
        response = client.get("/api/sessions/non_existent/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_delete_session(self) -> None:
        response = client.delete("/api/sessions/api_test")
        assert response.status_code == 200
        assert response.json() == {"status": "success"}
