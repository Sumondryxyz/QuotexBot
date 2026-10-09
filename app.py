"""Quotex Signal Generator Web Application & Real-time Dashboard.

Serves an interactive Web UI for viewing live Quotex signals, market analysis, and SMC confluence stats.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Any, Optional

from generator import QuotexSignalGenerator, Signal

PORT = int(os.getenv("PORT", "8080"))
HOST_BIND = "0.0.0.0"

# In-memory storage for active/historical web dashboard signals
LATEST_SIGNALS: List[Dict[str, Any]] = []
STATS = {"total_scanned": 0, "signals_generated": 0, "start_time": time.time()}

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quotex Live Signal Generator</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --border-color: #334155;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 25px;
        }

        h1 {
            font-size: 1.5rem;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .live-badge {
            background-color: rgba(34, 197, 94, 0.2);
            color: var(--accent-green);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .pulse {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse-anim 1.5s infinite;
        }

        @keyframes pulse-anim {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }

        .stat-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 15px;
            border-radius: 10px;
        }

        .stat-card .label {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 5px;
        }

        .stat-card .value {
            font-size: 1.4rem;
            font-weight: bold;
        }

        .signals-header {
            font-size: 1.2rem;
            margin-bottom: 15px;
            font-weight: 600;
        }

        .signal-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .signal-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .signal-card.BUY {
            border-left: 6px solid var(--accent-green);
        }

        .signal-card.SELL {
            border-left: 6px solid var(--accent-red);
        }

        .signal-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .pair-name {
            font-size: 1.3rem;
            font-weight: bold;
        }

        .action-tag {
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 0.95rem;
        }

        .action-tag.BUY {
            background-color: rgba(34, 197, 94, 0.2);
            color: var(--accent-green);
        }

        .action-tag.SELL {
            background-color: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
        }

        .signal-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            background: rgba(15, 23, 42, 0.4);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
        }

        .detail-item .lbl {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 2px;
        }

        .detail-item .val {
            font-size: 0.95rem;
            font-weight: 600;
        }

        .reasons-list {
            margin: 0;
            padding-left: 20px;
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        .reasons-list li {
            margin-bottom: 4px;
        }

        .no-signals {
            text-align: center;
            padding: 40px;
            background-color: var(--card-bg);
            border-radius: 12px;
            color: var(--text-muted);
            border: 1px dashed var(--border-color);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Quotex Signal Dashboard</h1>
            <div class="live-badge">
                <div class="pulse"></div> LIVE SCANNING
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Status</div>
                <div class="value" style="color: var(--accent-green);">Active</div>
            </div>
            <div class="stat-card">
                <div class="label">Signals Generated</div>
                <div class="value" id="stat-count">0</div>
            </div>
            <div class="stat-card">
                <div class="label">Strategy Engine</div>
                <div class="value" style="font-size: 1rem; margin-top: 5px;">SMC + ML Confluence</div>
            </div>
        </div>

        <div class="signals-header">⚡ Live Trading Signals</div>
        <div class="signal-list" id="signals-container">
            <div class="no-signals">Scanning Quotex markets for SMC confluence signals...</div>
        </div>
    </div>

    <script>
        async function fetchSignals() {
            try {
                const res = await fetch('/api/signals');
                const data = await res.json();

                document.getElementById('stat-count').innerText = data.total_signals;
                const container = document.getElementById('signals-container');

                if (data.signals.length === 0) {
                    container.innerHTML = '<div class="no-signals">Scanning Quotex markets for SMC confluence signals...</div>';
                    return;
                }

                container.innerHTML = data.signals.map(s => `
                    <div class="signal-card ${s.action}">
                        <div class="signal-top">
                            <span class="pair-name">${s.pair}</span>
                            <span class="action-tag ${s.action}">${s.action === 'BUY' ? '🟢 BUY (UP)' : '🔴 SELL (DOWN)'}</span>
                        </div>
                        <div class="signal-details">
                            <div class="detail-item">
                                <div class="lbl">Entry Time</div>
                                <div class="val">⏰ ${s.entry_time}</div>
                            </div>
                            <div class="detail-item">
                                <div class="lbl">Entry Price</div>
                                <div class="val">💵 ${s.price.toFixed(5)}</div>
                            </div>
                            <div class="detail-item">
                                <div class="lbl">Expiration</div>
                                <div class="val">⏳ ${s.expiration}s (Next Candle)</div>
                            </div>
                            <div class="detail-item">
                                <div class="lbl">SMC Score</div>
                                <div class="val">🎯 ${s.score}/110 (${s.strength})</div>
                            </div>
                            <div class="detail-item">
                                <div class="lbl">ML Confidence</div>
                                <div class="val">🤖 ${s.confidence ? (s.confidence * 100).toFixed(0) + '%' : 'N/A'}</div>
                            </div>
                        </div>
                        <ul class="reasons-list">
                            ${s.reasons.map(r => `<li>✔ ${r}</li>`).join('')}
                        </ul>
                    </div>
                `).join('');
            } catch (err) {
                console.error("Error fetching signals:", err);
            }
        }

        setInterval(fetchSignals, 3000);
        fetchSignals();
    </script>
</body>
</html>
"""


class WebDashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/dashboard"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
        elif self.path == "/api/signals":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {
                "total_signals": len(LATEST_SIGNALS),
                "signals": LATEST_SIGNALS,
                "uptime_seconds": int(time.time() - STATS["start_time"]),
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logs


def add_signal_to_dashboard(sig: Signal) -> None:
    data = {
        "pair": sig.pair,
        "action": sig.action,
        "direction": sig.direction,
        "entry_time": sig.entry_time_utc,
        "price": sig.price,
        "expiration": sig.expiration_seconds,
        "score": sig.score,
        "strength": sig.strength,
        "confidence": sig.model_confidence,
        "reasons": sig.reasons,
        "timestamp": sig.timestamp,
    }
    LATEST_SIGNALS.insert(0, data)
    if len(LATEST_SIGNALS) > 50:
        LATEST_SIGNALS.pop()
    STATS["signals_generated"] += 1


async def start_demo_signal_stream():
    """Generates periodic signals for web testing when live API credentials are not provided."""
    generator = QuotexSignalGenerator({"score_medium": 60})
    pairs = ["EURUSD_otc", "USDJPY_otc", "GBPUSD_otc", "AUDUSD_otc"]
    i = 0
    while True:
        pair = pairs[i % len(pairs)]
        candles = QuotexSignalGenerator.generate_demo_candles(count=150, trend="up", seed=int(time.time()))
        sig = generator.analyze(pair, candles)
        if sig:
            add_signal_to_dashboard(sig)
        i += 1
        await asyncio.sleep(10)


def run_web_server(port: int = PORT) -> HTTPServer:
    server = HTTPServer((HOST_BIND, port), WebDashboardHandler)
    print(f"🚀 Quotex Web Dashboard active at http://localhost:{port}")
    return server


if __name__ == "__main__":
    server = run_web_server(PORT)
    server.serve_forever()
