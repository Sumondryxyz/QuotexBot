"""Institutional-style SMC Quotex Signal Bot — multi-timeframe, scored confluence, Telegram + Render.

Refactored to utilize QuotexSignalGenerator core engine.
"""

import asyncio
import os
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import requests
from pyquotex.stable_api import Quotex

from generator import QuotexSignalGenerator, train_model

# ------------------------------------------------------------- config
HOST = os.getenv("QUOTEX_HOST", "market-qx.trade")
PERIOD_ENTRY = int(os.getenv("PERIOD_ENTRY", "60"))
PERIOD_STRUCT = int(os.getenv("PERIOD_STRUCT", "300"))
PERIOD_TREND = int(os.getenv("PERIOD_TREND", "900"))

SWING_INTERNAL = int(os.getenv("SWING_INTERNAL", "2"))
SWING_EXTERNAL = int(os.getenv("SWING_EXTERNAL", "5"))
EQUAL_TOL = float(os.getenv("EQUAL_TOL", "0.0006"))
DISPLACEMENT_MULT = float(os.getenv("DISPLACEMENT_MULT", "1.5"))

SCORE_STRONG = int(os.getenv("SCORE_STRONG", "90"))
SCORE_MEDIUM = int(os.getenv("SCORE_MEDIUM", "70"))
MODEL_MIN_CONF = float(os.getenv("MODEL_MIN_CONF", "0.55"))

HISTORY_CANDLES = int(os.getenv("HISTORY_CANDLES", "150"))
MIN_HISTORY = int(os.getenv("MIN_HISTORY", str(SWING_EXTERNAL * 2 + 10)))
RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "30"))
ZONE_MAX_AGE = int(os.getenv("ZONE_MAX_AGE", "80"))

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")

PAIRS = [
    p.strip()
    for p in os.getenv(
        "PAIRS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc,AUDUSD_otc,USDCHF_otc,NZDUSD_otc"
    ).split(",")
]


def stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


def clean_text(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_telegram(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        print("[Telegram Disabled] Outputting to stdout:")
        print(text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        print("telegram error:", exc)


# ------------------------------------------------------- Quotex session fix
def _get_cookies(self):
    jar = {}
    for cookie in self._client.cookies.jar:
        jar[cookie.name] = cookie.value
    return "; ".join(f"{name}={value}" for name, value in jar.items())


def _make_get_settings(host: str):
    from pyquotex.config import update_session

    async def get_settings(self):
        cookies = self.get_cookies()
        token = None
        resp = await self.send_request(
            method="GET",
            url=f"https://{host}/api/v1/cabinets/digest",
            headers={
                "Cookie": cookies,
                "Referer": f"https://{host}/en/trade",
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        if resp.status_code == 200:
            token = resp.json().get("data", {}).get("token")
        if not token:
            match = re.search(r'"?token"?\s*[:=]\s*"([A-Za-z0-9]{32,})"', str(self.get_soup()))
            token = match.group(1) if match else None
        self.ssid = token
        self.api.session_data["cookies"] = cookies
        self.api.session_data["token"] = token
        self.api.session_data["user_agent"] = self.headers["User-Agent"]
        update_session(self.api.username, self.api.session_data)
        return self.response, {"token": token}

    return get_settings


def apply_session_patch(host: str):
    from pyquotex.network import navigator
    from pyquotex.network.login import Login

    navigator.Browser.get_cookies = _get_cookies
    Login.get_settings = _make_get_settings(host)


# ----------------------------------------------------------------------- per pair
class PairState:
    def __init__(self, pair: str, generator: QuotexSignalGenerator):
        self.pair = pair
        self.generator = generator
        self.entry = []
        self.struct = []
        self.trend = []
        self.seen_entry = set()
        self.model = None
        self.new_since_retrain = 0
        self.obs = []
        self.fvgs = []
        self.pending = None
        self.wins = 0
        self.losses = 0
        self.last_signal_idx = -1

    async def _fetch_batched(self, q, period: int, need: int):
        merged = {}
        end = time.time()
        attempts = 0
        while len(merged) < need and attempts < 20:
            batch = await q.get_candles(self.pair, end, 25 * period, period)
            batch = [c for c in (batch or []) if c.get("close") is not None]
            if not batch:
                break
            for c in batch:
                merged[c["time"]] = c
            end = min(c["time"] for c in batch) - period
            attempts += 1
            await asyncio.sleep(0.15)
        return sorted(merged.values(), key=lambda c: c["time"])

    async def bootstrap(self, q) -> bool:
        self.entry = await self._fetch_batched(q, PERIOD_ENTRY, HISTORY_CANDLES)
        self.struct = await self._fetch_batched(q, PERIOD_STRUCT, 60)
        self.trend = await self._fetch_batched(q, PERIOD_TREND, 40)
        if len(self.entry) < MIN_HISTORY:
            print(f"{self.pair}: not enough entry-TF history ({len(self.entry)}), skipping")
            return False
        self.seen_entry = {c["time"] for c in self.entry}
        self.model = train_model(self.entry)
        print(
            f"{self.pair}: entry={len(self.entry)} struct={len(self.struct)} "
            f"trend={len(self.trend)} candles loaded"
        )
        return True

    async def poll(self, q) -> None:
        try:
            fresh_entry = await q.get_candles(
                self.pair, time.time(), 10 * PERIOD_ENTRY, PERIOD_ENTRY
            )
            fresh_struct = await q.get_candles(
                self.pair, time.time(), 6 * PERIOD_STRUCT, PERIOD_STRUCT
            )
            fresh_trend = await q.get_candles(
                self.pair, time.time(), 4 * PERIOD_TREND, PERIOD_TREND
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{self.pair} fetch error:", type(exc).__name__, exc)
            return

        new = False
        for c in fresh_entry or []:
            if c["time"] in self.seen_entry or c.get("close") is None:
                continue
            self.seen_entry.add(c["time"])
            self.entry.append(c)
            self.entry = self.entry[-2000:]
            self.new_since_retrain += 1
            new = True

            if self.pending:
                went_up = c["close"] > self.pending["ref_close"]
                correct = went_up == (self.pending["dir"] == "UP")
                self.wins += int(correct)
                self.losses += int(not correct)
                total = self.wins + self.losses
                print(
                    f"    {self.pair} -> {'WIN ' if correct else 'LOSS'} (close {c['close']:.5f})  "
                    f"session {self.wins}/{total} ({self.wins/total*100:.0f}%)"
                )
                self.pending = None

        for target, fresh in ((self.struct, fresh_struct), (self.trend, fresh_trend)):
            seen_times = {c["time"] for c in target}
            for c in fresh or []:
                if c["time"] not in seen_times and c.get("close") is not None:
                    target.append(c)
            target[:] = sorted({c["time"]: c for c in target}.values(), key=lambda x: x["time"])[
                -500:
            ]

        if not new:
            return

        latest = self.entry[-1]
        print(
            f"  {stamp(latest['time'])}  {self.pair:12s} price={latest['close']:.5f}  "
            f"({len(self.entry)} candles)"
        )

        if self.new_since_retrain >= RETRAIN_EVERY:
            self.model = train_model(self.entry)
            self.new_since_retrain = 0

        cur_idx = len(self.entry) - 1
        if cur_idx - self.last_signal_idx < 3:
            return

        signal = self.generator.analyze(
            pair=self.pair,
            entry_candles=self.entry,
            struct_candles=self.struct,
            trend_candles=self.trend,
            ml_model=self.model,
            active_obs=self.obs,
            active_fvgs=self.fvgs,
        )

        if not signal:
            return

        self.last_signal_idx = cur_idx
        self.pending = {"dir": signal.direction, "ref_close": latest["close"]}

        total = self.wins + self.losses
        winrate = f"{self.wins/total*100:.0f}%" if total else "n/a"

        msg = f"{signal.formatted_message}\n\nSession win rate: {winrate} ({total} signals)"
        print(
            f"\n{stamp(latest['time'])}  {self.pair:12s} {signal.action}  "
            f"score={signal.score} ({signal.strength})  reasons={signal.reasons}"
            + (
                f"  model={signal.model_confidence*100:.0f}%"
                if signal.model_confidence
                else ""
            )
        )
        send_telegram(msg)


# ----------------------------------------------------------------------- main
async def run_once() -> None:
    email = os.getenv("QUOTEX_EMAIL")
    password = os.getenv("QUOTEX_PASSWORD")
    if not email or not password:
        print("QUOTEX_EMAIL and QUOTEX_PASSWORD environment variables are required to connect to Quotex live API.")
        return

    apply_session_patch(HOST)
    q = Quotex(email=email, password=password, lang="en", host=HOST)
    q.set_account_mode("PRACTICE")
    ok, reason = await q.connect()
    print("connected" if ok else f"failed: {reason}")
    if not ok:
        return

    print(f"balance: {await q.get_balance()}")
    print(f"loading {len(PAIRS)} pairs...\n")

    generator = QuotexSignalGenerator(
        {
            "swing_internal": SWING_INTERNAL,
            "swing_external": SWING_EXTERNAL,
            "equal_tol": EQUAL_TOL,
            "displacement_mult": DISPLACEMENT_MULT,
            "score_strong": SCORE_STRONG,
            "score_medium": SCORE_MEDIUM,
            "model_min_conf": MODEL_MIN_CONF,
            "zone_max_age": ZONE_MAX_AGE,
            "period_entry": PERIOD_ENTRY,
        }
    )

    states = []
    for pair in PAIRS:
        st = PairState(pair, generator)
        if await st.bootstrap(q):
            states.append(st)
        await asyncio.sleep(0.3)

    if not states:
        print("no pairs had enough history, aborting this attempt")
        return

    send_telegram(f"🤖 SMC Pro bot started — watching {len(states)} pairs")
    print(f"\n{len(states)} pairs ready — scanning, Ctrl+C to stop\n")

    cycle = 0
    while True:
        cycle += 1
        for st in states:
            await st.poll(q)
            await asyncio.sleep(0.5)
        if cycle % 20 == 0:
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] cycle {cycle} heartbeat")
        await asyncio.sleep(max(3, PERIOD_ENTRY // 10))


async def main() -> None:
    backoff = 5
    while True:
        try:
            for f in ("session.json", os.path.expanduser("~/.pyquotex/session.json")):
                if os.path.exists(f):
                    os.remove(f)
            await run_once()
        except Exception as exc:  # noqa: BLE001
            print(f"crashed: {type(exc).__name__} {exc}")
        print(f"reconnecting in {backoff}s...\n")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


# ------------------------------------------------------- Render keep-alive
def start_keepalive() -> None:
    port = int(os.getenv("PORT", "10000"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    threading.Thread(
        target=lambda: HTTPServer(("0.0.0.0", port), Handler).serve_forever(),
        daemon=True,
    ).start()


if __name__ == "__main__":
    start_keepalive()
    asyncio.run(main())
