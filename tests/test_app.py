"""Unit tests for Quotex Signal Generator Web Application."""

import json
import unittest
import urllib.request
import threading
import time

from app import run_web_server, add_signal_to_dashboard, LATEST_SIGNALS
from generator import Signal


class TestWebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 8888
        cls.server = run_web_server(cls.port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_dashboard_html_endpoint(self):
        url = f"http://localhost:{self.port}/"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        html = req.read().decode("utf-8")
        self.assertIn("Quotex Signal Dashboard", html)

    def test_api_signals_endpoint(self):
        sig = Signal(
            pair="GBPUSD_otc",
            direction="UP",
            score=90,
            strength="STRONG",
            reasons=["BOS", "Order Block Retest"],
            price=1.2650,
            timestamp=1700000000,
            model_confidence=0.80,
        )
        add_signal_to_dashboard(sig)

        url = f"http://localhost:{self.port}/api/signals"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("signals", data)
        self.assertGreaterEqual(data["total_signals"], 1)
        self.assertEqual(data["signals"][0]["pair"], "GBPUSD_otc")

    def test_api_health_endpoint(self):
        url = f"http://localhost:{self.port}/api/health"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertEqual(data["status"], "ok")


if __name__ == "__main__":
    unittest.main()
