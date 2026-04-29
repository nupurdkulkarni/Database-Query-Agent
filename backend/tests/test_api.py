import pytest
import requests

HTTP_OK = 200
TIMEOUT_S = 10


class TestAPIEndpoints:
    """Layer 6: INTEGRATION — FastAPI endpoint health."""

    API_BASE = "http://127.0.0.1:8000"

    @pytest.fixture(autouse=True)
    def _check_server_running(self):
        try:
            resp = requests.get(f"{self.API_BASE}/api/sessions", timeout=2)
            if resp.status_code != HTTP_OK:
                pytest.skip("Backend server not responding")
        except Exception:
            pytest.skip("Backend server not running")

    def test_sessions_endpoint(self):
        resp = requests.get(f"{self.API_BASE}/api/sessions", timeout=TIMEOUT_S)
        assert resp.status_code == HTTP_OK
        assert isinstance(resp.json(), list)

    def test_chat_endpoint_streams(self):
        resp = requests.post(
            f"{self.API_BASE}/api/chat",
            json={"query": "How many projects?", "thread_id": "api_test"},
            stream=True,
            timeout=60,
        )
        assert resp.status_code == HTTP_OK
        # Check for SSE stream
        events = []
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                events.append(line)
                if events:
                    break
        assert len(events) > 0
