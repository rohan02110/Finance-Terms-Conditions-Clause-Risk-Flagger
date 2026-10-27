"""
Test suite for api/analyze.py serverless handler.
"""
import io
import json
from api.analyze import handler


class MockRequest:
    def __init__(self, body_bytes):
        self.rfile = io.BytesIO(body_bytes)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": str(len(body_bytes))}
        self.status = None
        self.response_headers = {}

    def send_response(self, code):
        self.status = code

    def send_header(self, key, value):
        self.response_headers[key] = value

    def end_headers(self):
        pass


def test_api_analyze_handler():
    # Simulate a POST request to /api/analyze
    payload = json.dumps({
        "text": "We reserve the right to change these terms at any time without prior notice. You agree to arbitrate all disputes.",
        "threshold": 0.05
    }).encode("utf-8")

    req = MockRequest(payload)
    
    # Instantiate handler with mocked environment
    h = handler.__new__(handler)
    h.rfile = req.rfile
    h.wfile = req.wfile
    h.headers = req.headers
    h.send_response = req.send_response
    h.send_header = req.send_header
    h.end_headers = req.end_headers

    h.do_POST()

    assert req.status == 200
    res_bytes = req.wfile.getvalue()
    res_data = json.loads(res_bytes.decode("utf-8"))

    assert "clauses" in res_data
    assert "summary" in res_data
    assert res_data["summary"]["total"] >= 2
    assert res_data["summary"]["flagged"] >= 1
    assert any("unilateral_change" in c["categories"] or "arbitration" in c["categories"] for c in res_data["clauses"])
