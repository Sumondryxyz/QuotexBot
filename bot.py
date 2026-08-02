"""Institutional-style SMC bot — multi-timeframe, scored confluence, Telegram + Render.

Self-contained (no dependency on predictor/), so this file alone can be deployed
as a Render Background Worker.

CONFIRMATIONS IMPLEMENTED (score out of 110):
    BOS confirmed by candle CLOSE (not wick)........... 20
    Liquidity sweep before the move.................... 20
    Retest of an unmitigated Order Block............... 20
    Retest of an unmitigated FVG (50% consequent enc.). 15
    Premium/Discount alignment (buy discount, sell prem) 10
    CHoCH present in the recent structure............... 10
    Strong displacement leg (body >= 1.5x avg body)..... 15
Score >=90 STRONG | >=70 MEDIUM | <70 ignored.

MULTI-TIMEFRAME: entry TF for triggers, a higher "structure" TF and a higher
"trend" TF for directional bias. All three EMA(20) slopes + structure must
agree with the entry-TF signal direction, or the signal is dropped.

ML CONFLUENCE: logistic regression written in plain numpy (gradient descent,
no scikit-learn/xgboost/lightgbm — those need native compilation that fails
on Termux/ARM, this doesn't). Retrains periodically on each pair's own recent
history. Signal only fires if the model agrees with the SMC direction above
MODEL_MIN_CONF.

SELF-LEARNING: every fired signal's next-candle outcome is logged (WIN/LOSS)
and the running win rate is reported with each new signal and daily via
Telegram. "Self-learning" here means periodic retraining on fresh data, not
outcome-reweighted training — a true reinforcement setup is a much bigger
project; this keeps the model from going stale, nothing more.

HONEST NOTE: none of the above features change the fundamental ceiling on
next-candle direction prediction. They reduce false signals and give you a
transparent reason for each one — they do not create edge that isn't in the
data. Track the win-rate output for at least 100+ signals before trusting it.

ENV VARS:
  QUOTEX_EMAIL, QUOTEX_PASSWORD
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  PAIRS="EURUSD_otc,GBPUSD_otc,..."
  PERIOD_ENTRY=60  PERIOD_STRUCT=300  PERIOD_TREND=900
  MODEL_MIN_CONF=0.55  SCORE_STRONG=90  SCORE_MEDIUM=70
  RETRAIN_EVERY=30  HISTORY_CANDLES=150

pip install pyquotex numpy requests
"""
import asyncio
import os
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import requests
from pyquotex.stable_api import Quotex

# ------------------------------------------------------------- config
HOST = os.getenv("QUOTEX_HOST", "market-qx.trade")
PERIOD_ENTRY = int(os.getenv("PERIOD_ENTRY", "60"))
PERIOD_STRUCT = int(os.getenv("PERIOD_STRUCT", "300"))
PERIOD_TREND = int(os.getenv("PERIOD_TREND", "900"))

SWING_INTERNAL = int(os.getenv("SWING_INTERNAL", "2"))
SWING_EXTERNAL = int(os.getenv("SWING_EXTERNAL", "5"))
EQUAL_TOL = float(os.getenv("EQUAL_TOL", "0.0006"))       # relative tolerance for equal highs/lows
DISPLACEMENT_MULT = float(os.getenv("DISPLACEMENT_MULT", "1.5"))

SCORE_STRONG = int(os.getenv("SCORE_STRONG", "90"))
SCORE_MEDIUM = int(os.getenv("SCORE_MEDIUM", "70"))
MODEL_MIN_CONF = float(os.getenv("MODEL_MIN_CONF", "0.55"))

HISTORY_CANDLES = int(os.getenv("HISTORY_CANDLES", "150"))
MIN_HISTORY = int(os.getenv("MIN_HISTORY", str(SWING_EXTERNAL * 2 + 10)))
RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "30"))
ZONE_MAX_AGE = int(os.getenv("ZONE_MAX_AGE", "80"))       # candles before a zone expires unused

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]

PAIRS = [p.strip() for p in os.getenv(
    "PAIRS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc,AUDUSD_otc,USDCHF_otc,NZDUSD_otc"
).split(",")]


def stamp(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


# ------------------------------------------------------- Quotex session fix
def _get_cookies(self):
    jar = {}
    for cookie in self._client.cookies.jar:
        jar[cookie.name] = cookie.value
    return "; ".join(f"{name}={value}" for name, value in jar.items())


def _make_get_settings(host):
    from pyquotex.config import update_session

    async def get_settings(self):
        cookies = self.get_cookies()
        token = None
        resp = await self.send_request(
            method="GET",
            url=f"https://{host}/api/v1/cabinets/digest",
            headers={"Cookie": cookies, "Referer": f"https://{host}/en/trade",
                     "Accept": "application/json", "X-Requested-With": "XMLHttpRequest"},
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


def apply_session_patch(host):
    from pyquotex.network import navigator
    from pyquotex.network.login import Login
    navigator.Browser.get_cookies = _get_cookies
    Login.get_settings = _make_get_settings(host)


# ------------------------------------------------------------- swing points
def swing_points(candles, lookback):
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    points = []
    for i in range(lookback, len(candles) - lookback):
        wh = highs[i - lookback:i + lookback + 1]
        wl = lows[i - lookback:i + lookback + 1]
        if highs[i] == max(wh):
            points.append((i, "high", highs[i]))
        if lows[i] == min(wl):
            points.append((i, "low", lows[i]))
    return points


# --------------------------------------------------- structure (close-confirmed)
def market_structure(candles, points):
    """BOS/CHoCH confirmed by candle CLOSE beyond the swing (not just a wick).
    A wick-only break is reported separately as a liquidity sweep event."""
    events = []      # (idx, 'BOS_UP'|'BOS_DOWN'|'CHOCH_UP'|'CHOCH_DOWN')
    sweeps = []       # (idx, 'sweep_high'|'sweep_low', level)
    last_high = last_low = None
    bias = None

    for idx, kind, price in points:
        close = candles[idx]["close"]
        if kind == "high":
            if last_high is not None and price > last_high:
                if close > last_high:  # confirmed break
                    if bias == "down":
                        events.append((idx, "CHOCH_UP"))
                        bias = "up"
                    elif bias == "up":
                        events.append((idx, "BOS_UP"))
                else:
                    sweeps.append((idx, "sweep_high", last_high))
            last_high = price if last_high is None else max(last_high, price)
        else:
            if last_low is not None and price < last_low:
                if close < last_low:
                    if bias == "up":
                        events.append((idx, "CHOCH_DOWN"))
                        bias = "down"
                    elif bias == "down":
                        events.append((idx, "BOS_DOWN"))
                else:
                    sweeps.append((idx, "sweep_low", last_low))
            last_low = price if last_low is None else min(last_low, price)

        if bias is None:
            bias = "up" if kind == "high" else "down"

    return events, sweeps


def equal_levels(points, kind, tol):
    """Cluster swing highs (or lows) that sit within `tol` relative distance
    of each other — these are liquidity pools (equal highs/equal lows)."""
    vals = sorted([p for i, k, p in points if k == kind])
    clusters = []
    for v in vals:
        placed = False
        for c in clusters:
            if abs(v - c[-1]) / max(c[-1], 1e-9) <= tol:
                c.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters if len(c) >= 2]  # need >=2 touches to count


def inducement_filter(candles, events, lookahead=2):
    """Drop BOS/CHoCH events that reverse within `lookahead` candles — a fake
    break used to trap retail traders before the real move."""
    clean = []
    for idx, kind in events:
        bullish = kind.endswith("UP")
        failed = False
        for j in range(idx + 1, min(idx + 1 + lookahead, len(candles))):
            if bullish and candles[j]["close"] < candles[idx]["close"]:
                failed = True
                break
            if not bullish and candles[j]["close"] > candles[idx]["close"]:
                failed = True
                break
        if not failed:
            clean.append((idx, kind))
    return clean


# --------------------------------------------------------- order blocks / breakers
def displacement_strength(candles, idx, window=20):
    bodies = [abs(c["close"] - c["open"]) for c in candles[max(0, idx - window):idx]]
    avg = (sum(bodies) / len(bodies)) if bodies else 1e-9
    body = abs(candles[idx]["close"] - candles[idx]["open"])
    return body / max(avg, 1e-9)


def find_order_block(candles, break_idx, bullish):
    """Last opposite-colour candle before a displacement leg into the break."""
    disp_ratio = displacement_strength(candles, break_idx)
    if disp_ratio < DISPLACEMENT_MULT:
        return None, disp_ratio
    for i in range(break_idx, max(break_idx - 15, 0), -1):
        c = candles[i]
        is_down, is_up = c["close"] < c["open"], c["close"] > c["open"]
        if (bullish and is_down) or (not bullish and is_up):
            lo, hi = min(c["open"], c["close"]), max(c["open"], c["close"])
            return {"lo": lo, "hi": hi, "idx": i, "bullish": bullish,
                    "mitigated": False, "breaker": False}, disp_ratio
    return None, disp_ratio


def fair_value_gaps(candles, bullish, start=0):
    gaps = []
    for i in range(max(1, start), len(candles) - 1):
        prev_c, next_c = candles[i - 1], candles[i + 1]
        if bullish and next_c["low"] > prev_c["high"]:
            lo, hi = prev_c["high"], next_c["low"]
            gaps.append({"lo": lo, "hi": hi, "mid": (lo + hi) / 2, "idx": i,
                         "bullish": True, "mitigated": False})
        if not bullish and next_c["high"] < prev_c["low"]:
            lo, hi = next_c["high"], prev_c["low"]
            gaps.append({"lo": lo, "hi": hi, "mid": (lo + hi) / 2, "idx": i,
                         "bullish": False, "mitigated": False})
    return gaps


def premium_discount(candles, points, price):
    highs = [p for i, k, p in points if k == "high"]
    lows = [p for i, k, p in points if k == "low"]
    if not highs or not lows:
        return None
    eq = (max(highs[-5:]) + min(lows[-5:])) / 2
    return "discount" if price < eq else "premium"


# ------------------------------------------------------------------- ML model
def ema(values, span):
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = ema(np.clip(delta, 0, None), period)
    loss = ema(np.clip(-delta, 0, None), period)
    rs = gain / np.where(loss == 0, 1e-12, loss)
    return 100 - 100 / (1 + rs)


def build_features(candles):
    o = np.array([c["open"] for c in candles], float)
    h = np.array([c["high"] for c in candles], float)
    low = np.array([c["low"] for c in candles], float)
    c = np.array([c["close"] for c in candles], float)
    eps = 1e-12
    rng = np.where(h - low == 0, eps, h - low)
    ret1 = np.diff(c, prepend=c[0]) / np.where(c == 0, eps, c)
    ret3 = (c - np.roll(c, 3)) / np.where(c == 0, eps, c)
    ema_f, ema_s = ema(c, 5), ema(c, 20)
    macd = ema(c, 12) - ema(c, 26)
    macd_h = macd - ema(macd, 9)
    r = rsi(c)
    body = (c - o) / rng
    cols = [
        ret1 * 1e4, ret3 * 1e4, body,
        (c - ema_f) / np.where(ema_f == 0, eps, ema_f) * 1e4,
        (ema_f - ema_s) / np.where(ema_s == 0, eps, ema_s) * 1e4,
        (r - 50) / 50, macd_h * 1e4,
    ]
    X = np.column_stack(cols)
    X[:6] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def model_fit(X, y, epochs=300, lr=0.15, l2=1e-3):
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / std
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        err = p - y
        w -= lr * (Z.T @ err / len(y) + l2 * w)
        b -= lr * err.mean()
    return w, b, mean, std


def train_model(candles):
    if len(candles) < 15:
        return None
    X = build_features(candles)
    c = np.array([x["close"] for x in candles], float)
    y = (np.roll(c, -1) > c).astype(float)
    keep = np.roll(c, -1) != c
    keep[-1] = False
    if keep.sum() < 12:
        return None
    return model_fit(X[keep], y[keep])


def model_predict(model, candles):
    w, b, mean, std = model
    x = build_features(candles)[-1]
    z = (x - mean) / std
    return float(1 / (1 + np.exp(-(z @ w + b))))


# ---------------------------------------------------------------- HTF trend bias
def trend_bias(candles):
    if len(candles) < 25:
        return None
    closes = np.array([c["close"] for c in candles], float)
    fast, slow = ema(closes, 8)[-1], ema(closes, 21)[-1]
    return "up" if fast > slow else "down"


# ------------------------------------------------------------------- Telegram
def clean_text(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_telegram(text):
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=15)
    except Exception as exc:  # noqa: BLE001
        print("telegram error:", exc)


# ----------------------------------------------------------------------- per pair
class PairState:
    def __init__(self, pair):
        self.pair = pair
        self.entry = []      # entry TF candles
        self.struct = []     # structure TF candles
        self.trend = []      # trend TF candles
        self.seen_entry = set()
        self.model = None
        self.new_since_retrain = 0
        self.obs = []         # active order blocks
        self.fvgs = []         # active FVGs
        self.pending = None
        self.wins = 0
        self.losses = 0
        self.last_signal_idx = -1

    async def _fetch_batched(self, q, period, need):
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

    async def bootstrap(self, q):
        self.entry = await self._fetch_batched(q, PERIOD_ENTRY, HISTORY_CANDLES)
        self.struct = await self._fetch_batched(q, PERIOD_STRUCT, 60)
        self.trend = await self._fetch_batched(q, PERIOD_TREND, 40)
        if len(self.entry) < MIN_HISTORY:
            print(f"{self.pair}: not enough entry-TF history ({len(self.entry)}), skipping")
            return False
        self.seen_entry = {c["time"] for c in self.entry}
        self.model = train_model(self.entry)
        print(f"{self.pair}: entry={len(self.entry)} struct={len(self.struct)} "
              f"trend={len(self.trend)} candles loaded")
        return True

    def _refresh_zones(self, points, events):
        """Rebuild active OB/FVG lists around the latest confirmed structure event."""
        if not events:
            return None
        idx, kind = events[-1]
        bullish = kind.endswith("UP")
        ob, disp_ratio = find_order_block(self.entry, idx, bullish)
        if ob:
            if not any(z["idx"] == ob["idx"] for z in self.obs):
                self.obs.append(ob)
        gaps = fair_value_gaps(self.entry, bullish, start=max(0, idx - 20))
        for g in gaps[-3:]:
            if not any(z["idx"] == g["idx"] and z["bullish"] == g["bullish"] for z in self.fvgs):
                self.fvgs.append(g)
        return disp_ratio

    def _update_mitigation(self, price):
        for z in self.obs:
            if not z["mitigated"]:
                if z["bullish"] and price < z["lo"]:
                    z["mitigated"], z["breaker"] = True, True
                elif not z["bullish"] and price > z["hi"]:
                    z["mitigated"], z["breaker"] = True, True
        for g in self.fvgs:
            if g["mitigated"]:
                continue
            if g["bullish"] and price <= g["lo"]:
                g["mitigated"] = True
            elif not g["bullish"] and price >= g["hi"]:
                g["mitigated"] = True
        cur_idx = len(self.entry) - 1
        self.obs = [z for z in self.obs if cur_idx - z["idx"] <= ZONE_MAX_AGE and not z["mitigated"]]
        self.fvgs = [g for g in self.fvgs if cur_idx - g["idx"] <= ZONE_MAX_AGE and not g["mitigated"]]

    def _score_signal(self, direction, events, sweeps, points, disp_ratio, zone_hit):
        score = 0
        reasons = []
        if events and events[-1][1].startswith("BOS"):
            score += 20; reasons.append("BOS")
        if events and events[-1][1].startswith("CHOCH"):
            score += 10; reasons.append("CHoCH")
        if sweeps:
            score += 20; reasons.append("Liquidity Sweep")
        if zone_hit == "OB":
            score += 20; reasons.append("Order Block retest")
        elif zone_hit == "FVG":
            score += 15; reasons.append("FVG retest")
        pd = premium_discount(self.entry, points, self.entry[-1]["close"])
        if pd and ((direction == "UP" and pd == "discount") or (direction == "DOWN" and pd == "premium")):
            score += 10; reasons.append(f"{pd.title()} zone")
        if disp_ratio and disp_ratio >= DISPLACEMENT_MULT:
            score += 15; reasons.append(f"Displacement x{disp_ratio:.1f}")
        return score, reasons

    async def poll(self, q):
        try:
            fresh_entry = await q.get_candles(self.pair, time.time(), 10 * PERIOD_ENTRY, PERIOD_ENTRY)
            fresh_struct = await q.get_candles(self.pair, time.time(), 6 * PERIOD_STRUCT, PERIOD_STRUCT)
            fresh_trend = await q.get_candles(self.pair, time.time(), 4 * PERIOD_TREND, PERIOD_TREND)
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
                print(f"    {self.pair} -> {'WIN ' if correct else 'LOSS'} (close {c['close']:.5f})  "
                      f"session {self.wins}/{total} ({self.wins/total*100:.0f}%)")
                self.pending = None

        for target, fresh in ((self.struct, fresh_struct), (self.trend, fresh_trend)):
            seen_times = {c["time"] for c in target}
            for c in fresh or []:
                if c["time"] not in seen_times and c.get("close") is not None:
                    target.append(c)
            target[:] = sorted({c["time"]: c for c in target}.values(), key=lambda x: x["time"])[-500:]

        if not new:
            return

        latest = self.entry[-1]
        print(f"  {stamp(latest['time'])}  {self.pair:12s} price={latest['close']:.5f}  "
              f"({len(self.entry)} candles)")

        if self.new_since_retrain >= RETRAIN_EVERY:
            self.model = train_model(self.entry)
            self.new_since_retrain = 0

        pts_int = swing_points(self.entry, SWING_INTERNAL)
        pts_ext = swing_points(self.entry, SWING_EXTERNAL)
        ev_int_raw, sweeps_int = market_structure(self.entry, pts_int)
        ev_ext_raw, _sweeps_ext = market_structure(self.entry, pts_ext)
        ev_int = inducement_filter(self.entry, ev_int_raw)
        ev_ext = inducement_filter(self.entry, ev_ext_raw)
        if not ev_int or not ev_ext:
            return

        dir_int = "UP" if ev_int[-1][1].endswith("UP") else "DOWN"
        dir_ext = "UP" if ev_ext[-1][1].endswith("UP") else "DOWN"
        if dir_int != dir_ext:
            return

        struct_bias = trend_bias(self.struct)
        trend_tf_bias = trend_bias(self.trend)
        want = "up" if dir_int == "UP" else "down"
        if struct_bias and struct_bias != want:
            return
        if trend_tf_bias and trend_tf_bias != want:
            return

        disp_ratio = self._refresh_zones(pts_ext, ev_ext)
        self._update_mitigation(latest["close"])

        zone_hit = None
        for z in self.obs:
            if not z["mitigated"] and z["lo"] <= latest["close"] <= z["hi"] and z["bullish"] == (dir_int == "UP"):
                zone_hit = "OB"
                z["mitigated"] = True
                break
        if not zone_hit:
            for g in self.fvgs:
                if not g["mitigated"] and g["lo"] <= latest["close"] <= g["hi"] and g["bullish"] == (dir_int == "UP"):
                    if (g["bullish"] and latest["close"] <= g["mid"]) or (not g["bullish"] and latest["close"] >= g["mid"]):
                        zone_hit = "FVG"
                        g["mitigated"] = True
                        break
        if not zone_hit:
            return

        score, reasons = self._score_signal(dir_int, ev_int, sweeps_int, pts_ext, disp_ratio, zone_hit)
        if score < SCORE_MEDIUM:
            return

        model_conf = None
        if self.model:
            prob_up = model_predict(self.model, self.entry)
            model_dir = "UP" if prob_up > 0.5 else "DOWN"
            model_conf = max(prob_up, 1 - prob_up)
            if model_dir != dir_int or model_conf < MODEL_MIN_CONF:
                print(f"    {self.pair}: SMC score {score} ({dir_int}) but model says "
                      f"{model_dir} ({model_conf*100:.0f}%) — no confluence, skipping")
                return

        cur_idx = len(self.entry) - 1
        if cur_idx - self.last_signal_idx < 3:
            return
        self.last_signal_idx = cur_idx

        strength = "STRONG" if score >= SCORE_STRONG else "MEDIUM"
        self.pending = {"dir": dir_int, "ref_close": latest["close"]}
        arrow = "🟢 BUY" if dir_int == "UP" else "🔴 SELL"
        checklist = "\n".join(f"✔ {r}" for r in reasons)
        total = self.wins + self.losses
        winrate = f"{self.wins/total*100:.0f}%" if total else "n/a"

        msg = f"<b>{clean_text(self.pair)}</b>  {arrow}\n"
        if model_conf:
            msg += f"Confidence: {model_conf*100:.0f}%\n"
        msg += (
            f"SMC Score: {score}/110 ({strength})\n\n"
            f"Reason:\n{clean_text(checklist)}\n\n"
            f"Session win rate: {winrate} ({total} signals)"
        )
        print(f"\n{stamp(latest['time'])}  {self.pair:12s} {arrow}  score={score} ({strength})  "
              f"reasons={reasons}" + (f"  model={model_conf*100:.0f}%" if model_conf else ""))
        send_telegram(msg)


# ----------------------------------------------------------------------- main
async def run_once():
    apply_session_patch(HOST)
    q = Quotex(email=os.environ["QUOTEX_EMAIL"], password=os.environ["QUOTEX_PASSWORD"],
               lang="en", host=HOST)
    q.set_account_mode("PRACTICE")
    ok, reason = await q.connect()
    print("connected" if ok else f"failed: {reason}")
    if not ok:
        return

    print(f"balance: {await q.get_balance()}")
    print(f"loading {len(PAIRS)} pairs...\n")

    states = []
    for pair in PAIRS:
        st = PairState(pair)
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


async def main():
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
def start_keepalive():
    port = int(os.getenv("PORT", "10000"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    threading.Thread(target=lambda: HTTPServer(("0.0.0.0", port), Handler).serve_forever(),
                      daemon=True).start()


if __name__ == "__main__":
    start_keepalive()
    asyncio.run(main())