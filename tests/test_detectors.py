import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from pdfcloak.detectors import ClipCloakDetector, RegexDetector
from pdfcloak.models import Span, resolve_overlaps
from pdfcloak.rules import Rule

TEST_TOKEN = "sek" + "ret"


def test_regex_detector_offsets():
    det = RegexDetector([Rule("email", r"[\w.+-]+@[\w-]+\.[\w.]+")])
    text = "mail me at a@b.example.org please"
    spans = det.detect(text, page=0, labels=[], threshold=0.5)
    assert len(spans) == 1
    assert text[spans[0].start:spans[0].end] == "a@b.example.org"
    assert spans[0].score == 2.0


def test_overlap_regex_beats_model():
    regex_span = Span(0, 0, 10, "email", 2.0, "regex", "x" * 10)
    model_span = Span(0, 5, 20, "name", 0.99, "clipcloak", "y" * 15)
    kept = resolve_overlaps([model_span, regex_span])
    # Regex wins the overlap, but the model's non-overlapping tail survives
    # as a residual span instead of being dropped whole (PII fragment leak).
    assert kept[0] == regex_span
    assert [(s.start, s.end, s.label) for s in kept[1:]] == [(10, 20, "name")]
    assert kept[1].text == "y" * 10


def test_clipcloak_detector_against_fake_sidecar():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            assert self.headers.get("Authorization") == "Bearer " + TEST_TOKEN
            text = body["text"]
            start = text.index("Jane")
            payload = {"entities": [{"label": "name", "text": "Jane Doe",
                                     "start": start, "end": start + 8, "score": 0.9}],
                       "redacted": text, "parts": 1}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        det = ClipCloakDetector(f"http://127.0.0.1:{server.server_port}", TEST_TOKEN)
        assert det.health()
        text = "Hello Jane Doe, welcome"
        spans = det.detect(text, page=2, labels=["name"], threshold=0.55)
        assert len(spans) == 1
        assert spans[0].page == 2
        assert text[spans[0].start:spans[0].end] == "Jane Doe"
    finally:
        server.shutdown()
