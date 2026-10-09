"""Quotex Signal Generator Engine.

Core signal generator module implementing Smart Money Concepts (SMC)
confluence scoring and machine learning directional filtering for Quotex/binary trading.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class Signal:
    pair: str
    direction: str  # "UP" (BUY) or "DOWN" (SELL)
    score: int  # Confluence score out of 110
    strength: str  # "STRONG" or "MEDIUM"
    reasons: List[str]
    price: float
    timestamp: int
    model_confidence: Optional[float] = None
    expiration_seconds: int = 60

    @property
    def action(self) -> str:
        return "BUY" if self.direction == "UP" else "SELL"

    @property
    def entry_time_utc(self) -> str:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%H:%M:%S UTC")

    @property
    def formatted_message(self) -> str:
        arrow = "🟢 BUY" if self.direction == "UP" else "🔴 SELL"
        checklist = "\n".join(f"✔ {r}" for r in self.reasons)
        conf_str = (
            f"Confidence: {self.model_confidence * 100:.0f}%\n"
            if self.model_confidence is not None
            else ""
        )
        return (
            f"<b>{self.pair}</b>  {arrow}\n"
            f"⏰ Entry Time: {self.entry_time_utc}\n"
            f"💵 Price: {self.price:.5f}\n"
            f"⏳ Expiration: {self.expiration_seconds}s (Next Candle)\n"
            f"{conf_str}"
            f"SMC Score: {self.score}/110 ({self.strength})\n\n"
            f"Reasons:\n{checklist}"
        )


def swing_points(candles: List[Dict[str, Any]], lookback: int) -> List[Tuple[int, str, float]]:
    """Identify swing highs and swing lows in candle data."""
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    points = []
    for i in range(lookback, len(candles) - lookback):
        wh = highs[i - lookback : i + lookback + 1]
        wl = lows[i - lookback : i + lookback + 1]
        if highs[i] == max(wh):
            points.append((i, "high", highs[i]))
        if lows[i] == min(wl):
            points.append((i, "low", lows[i]))
    return points


def market_structure(
    candles: List[Dict[str, Any]], points: List[Tuple[int, str, float]]
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str, float]]]:
    """Identify BOS / CHoCH structure breaks confirmed by candle close."""
    events = []  # (idx, 'BOS_UP' | 'BOS_DOWN' | 'CHOCH_UP' | 'CHOCH_DOWN')
    sweeps = []  # (idx, 'sweep_high' | 'sweep_low', level)
    last_high = None
    last_low = None
    bias = None

    for idx, kind, price in points:
        close = candles[idx]["close"]
        if kind == "high":
            if last_high is not None and price > last_high:
                if close > last_high:  # Close confirmed break
                    if bias == "down":
                        events.append((idx, "CHOCH_UP"))
                        bias = "up"
                    elif bias == "up":
                        events.append((idx, "BOS_UP"))
                else:  # Wick-only break
                    sweeps.append((idx, "sweep_high", last_high))
            last_high = price if last_high is None else max(last_high, price)
        else:
            if last_low is not None and price < last_low:
                if close < last_low:  # Close confirmed break
                    if bias == "up":
                        events.append((idx, "CHOCH_DOWN"))
                        bias = "down"
                    elif bias == "down":
                        events.append((idx, "BOS_DOWN"))
                else:  # Wick-only break
                    sweeps.append((idx, "sweep_low", last_low))
            last_low = price if last_low is None else min(last_low, price)

        if bias is None:
            bias = "up" if kind == "high" else "down"

    return events, sweeps


def equal_levels(points: List[Tuple[int, str, float]], kind: str, tol: float) -> List[float]:
    """Find clusters of equal highs or equal lows (liquidity pools)."""
    vals = sorted([p for _, k, p in points if k == kind])
    clusters: List[List[float]] = []
    for v in vals:
        placed = False
        for c in clusters:
            if abs(v - c[-1]) / max(c[-1], 1e-9) <= tol:
                c.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters if len(c) >= 2]


def inducement_filter(
    candles: List[Dict[str, Any]], events: List[Tuple[int, str]], lookahead: int = 2
) -> List[Tuple[int, str]]:
    """Filter out fake breaks (inducements) that immediately fail."""
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


def displacement_strength(
    candles: List[Dict[str, Any]], idx: int, window: int = 20
) -> float:
    """Calculate ratio of candle body size relative to recent average body size."""
    bodies = [abs(c["close"] - c["open"]) for c in candles[max(0, idx - window) : idx]]
    avg = (sum(bodies) / len(bodies)) if bodies else 1e-9
    body = abs(candles[idx]["close"] - candles[idx]["open"])
    return body / max(avg, 1e-9)


def find_order_block(
    candles: List[Dict[str, Any]], break_idx: int, bullish: bool, displacement_mult: float = 1.5
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Locate last opposite-color candle before a strong displacement break."""
    disp_ratio = displacement_strength(candles, break_idx)
    if disp_ratio < displacement_mult:
        return None, disp_ratio
    for i in range(break_idx, max(break_idx - 15, 0), -1):
        c = candles[i]
        is_down, is_up = c["close"] < c["open"], c["close"] > c["open"]
        if (bullish and is_down) or (not bullish and is_up):
            lo, hi = min(c["open"], c["close"]), max(c["open"], c["close"])
            return (
                {
                    "lo": lo,
                    "hi": hi,
                    "idx": i,
                    "bullish": bullish,
                    "mitigated": False,
                    "breaker": False,
                },
                disp_ratio,
            )
    return None, disp_ratio


def fair_value_gaps(
    candles: List[Dict[str, Any]], bullish: bool, start: int = 0
) -> List[Dict[str, Any]]:
    """Find unmitigated Fair Value Gaps (FVG)."""
    gaps = []
    for i in range(max(1, start), len(candles) - 1):
        prev_c, next_c = candles[i - 1], candles[i + 1]
        if bullish and next_c["low"] > prev_c["high"]:
            lo, hi = prev_c["high"], next_c["low"]
            gaps.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "mid": (lo + hi) / 2,
                    "idx": i,
                    "bullish": True,
                    "mitigated": False,
                }
            )
        if not bullish and next_c["high"] < prev_c["low"]:
            lo, hi = next_c["high"], prev_c["low"]
            gaps.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "mid": (lo + hi) / 2,
                    "idx": i,
                    "bullish": False,
                    "mitigated": False,
                }
            )
    return gaps


def premium_discount(
    candles: List[Dict[str, Any]], points: List[Tuple[int, str, float]], price: float
) -> Optional[str]:
    """Determine whether current price sits in Premium or Discount zone."""
    highs = [p for _, k, p in points if k == "high"]
    lows = [p for _, k, p in points if k == "low"]
    if not highs or not lows:
        return None
    eq = (max(highs[-5:]) + min(lows[-5:])) / 2
    return "discount" if price < eq else "premium"


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential Moving Average using numpy."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index using numpy."""
    delta = np.diff(close, prepend=close[0])
    gain = ema(np.clip(delta, 0, None), period)
    loss = ema(np.clip(-delta, 0, None), period)
    rs = gain / np.where(loss == 0, 1e-12, loss)
    return 100 - 100 / (1 + rs)


def build_features(candles: List[Dict[str, Any]]) -> np.ndarray:
    """Extract technical feature matrix for logistic regression ML model."""
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
        ret1 * 1e4,
        ret3 * 1e4,
        body,
        (c - ema_f) / np.where(ema_f == 0, eps, ema_f) * 1e4,
        (ema_f - ema_s) / np.where(ema_s == 0, eps, ema_s) * 1e4,
        (r - 50) / 50,
        macd_h * 1e4,
    ]
    X = np.column_stack(cols)
    X[:6] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def model_fit(
    X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.15, l2: float = 1e-3
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit pure numpy logistic regression model via gradient descent."""
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


def train_model(candles: List[Dict[str, Any]]) -> Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    """Train ML model on candle history to predict next candle direction."""
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


def model_predict(model: Tuple[np.ndarray, float, np.ndarray, np.ndarray], candles: List[Dict[str, Any]]) -> float:
    """Predict probability of next candle closing UP (higher than current close)."""
    w, b, mean, std = model
    x = build_features(candles)[-1]
    z = (x - mean) / std
    return float(1 / (1 + np.exp(-(z @ w + b))))


def trend_bias(candles: List[Dict[str, Any]]) -> Optional[str]:
    """Calculate EMA trend direction bias."""
    if len(candles) < 25:
        return None
    closes = np.array([c["close"] for c in candles], float)
    fast, slow = ema(closes, 8)[-1], ema(closes, 21)[-1]
    return "up" if fast > slow else "down"


class QuotexSignalGenerator:
    """Quotex Signal Generator combining SMC Confluence & ML Probability filters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.swing_internal = cfg.get("swing_internal", 2)
        self.swing_external = cfg.get("swing_external", 5)
        self.equal_tol = cfg.get("equal_tol", 0.0006)
        self.displacement_mult = cfg.get("displacement_mult", 1.5)
        self.score_strong = cfg.get("score_strong", 90)
        self.score_medium = cfg.get("score_medium", 70)
        self.model_min_conf = cfg.get("model_min_conf", 0.55)
        self.zone_max_age = cfg.get("zone_max_age", 80)
        self.period_entry = cfg.get("period_entry", 60)

    def analyze(
        self,
        pair: str,
        entry_candles: List[Dict[str, Any]],
        struct_candles: Optional[List[Dict[str, Any]]] = None,
        trend_candles: Optional[List[Dict[str, Any]]] = None,
        ml_model: Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray]] = None,
        active_obs: Optional[List[Dict[str, Any]]] = None,
        active_fvgs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Signal]:
        """Analyze candles and return a Signal if criteria and score thresholds are met."""
        min_candles = self.swing_external * 2 + 10
        if len(entry_candles) < min_candles:
            return None

        latest = entry_candles[-1]
        pts_int = swing_points(entry_candles, self.swing_internal)
        pts_ext = swing_points(entry_candles, self.swing_external)

        ev_int_raw, sweeps_int = market_structure(entry_candles, pts_int)
        ev_ext_raw, _sweeps_ext = market_structure(entry_candles, pts_ext)

        ev_int = inducement_filter(entry_candles, ev_int_raw)
        ev_ext = inducement_filter(entry_candles, ev_ext_raw)

        if not ev_int or not ev_ext:
            return None

        dir_int = "UP" if ev_int[-1][1].endswith("UP") else "DOWN"
        dir_ext = "UP" if ev_ext[-1][1].endswith("UP") else "DOWN"

        if dir_int != dir_ext:
            return None

        want = "up" if dir_int == "UP" else "down"

        if struct_candles and len(struct_candles) >= 25:
            sbias = trend_bias(struct_candles)
            if sbias and sbias != want:
                return None

        if trend_candles and len(trend_candles) >= 25:
            tbias = trend_bias(trend_candles)
            if tbias and tbias != want:
                return None

        # Rebuild order block / FVG zones
        obs = active_obs if active_obs is not None else []
        fvgs = active_fvgs if active_fvgs is not None else []

        idx, kind = ev_ext[-1]
        bullish = kind.endswith("UP")
        ob, disp_ratio = find_order_block(
            entry_candles, idx, bullish, self.displacement_mult
        )
        if ob and not any(z["idx"] == ob["idx"] for z in obs):
            obs.append(ob)

        gaps = fair_value_gaps(entry_candles, bullish, start=max(0, idx - 20))
        for g in gaps[-3:]:
            if not any(z["idx"] == g["idx"] and z["bullish"] == g["bullish"] for z in fvgs):
                fvgs.append(g)

        # Check zone retest
        zone_hit = None
        for z in obs:
            if not z.get("mitigated") and z["lo"] <= latest["close"] <= z["hi"] and z["bullish"] == (dir_int == "UP"):
                zone_hit = "OB"
                z["mitigated"] = True
                break

        if not zone_hit:
            for g in fvgs:
                if not g.get("mitigated") and g["lo"] <= latest["close"] <= g["hi"] and g["bullish"] == (dir_int == "UP"):
                    if (g["bullish"] and latest["close"] <= g["mid"]) or (
                        not g["bullish"] and latest["close"] >= g["mid"]
                    ):
                        zone_hit = "FVG"
                        g["mitigated"] = True
                        break

        if not zone_hit:
            return None

        # Score signal
        score = 0
        reasons = []
        if ev_int and ev_int[-1][1].startswith("BOS"):
            score += 20
            reasons.append("BOS (Break of Structure)")
        if ev_int and ev_int[-1][1].startswith("CHOCH"):
            score += 10
            reasons.append("CHoCH (Change of Character)")
        if sweeps_int:
            score += 20
            reasons.append("Liquidity Sweep")
        if zone_hit == "OB":
            score += 20
            reasons.append("Order Block Retest")
        elif zone_hit == "FVG":
            score += 15
            reasons.append("Fair Value Gap (50%) Retest")

        pd = premium_discount(entry_candles, pts_ext, latest["close"])
        if pd and ((dir_int == "UP" and pd == "discount") or (dir_int == "DOWN" and pd == "premium")):
            score += 10
            reasons.append(f"{pd.title()} Zone Alignment")

        if disp_ratio and disp_ratio >= self.displacement_mult:
            score += 15
            reasons.append(f"Strong Displacement Leg (x{disp_ratio:.1f})")

        if score < self.score_medium:
            return None

        # Machine Learning Confluence
        model_conf = None
        model = ml_model if ml_model is not None else train_model(entry_candles)
        if model:
            prob_up = model_predict(model, entry_candles)
            model_dir = "UP" if prob_up > 0.5 else "DOWN"
            model_conf = max(prob_up, 1 - prob_up)
            if model_dir != dir_int or model_conf < self.model_min_conf:
                return None

        strength = "STRONG" if score >= self.score_strong else "MEDIUM"

        return Signal(
            pair=pair,
            direction=dir_int,
            score=score,
            strength=strength,
            reasons=reasons,
            price=latest["close"],
            timestamp=latest["time"],
            model_confidence=model_conf,
            expiration_seconds=self.period_entry,
        )

    @staticmethod
    def generate_demo_candles(
        count: int = 150, trend: str = "up", seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Generate realistic synthetic OHLC candles for testing and CLI demonstration."""
        np.random.seed(seed)
        start_time = int(datetime.now(timezone.utc).timestamp()) - (count * 60)
        price = 1.0850
        candles = []
        drift = 0.00015 if trend == "up" else -0.00015

        for i in range(count):
            t = start_time + i * 60
            ret = np.random.normal(drift, 0.0004)
            close = max(0.5, price * (1 + ret))
            high = max(price, close) + abs(np.random.normal(0, 0.0002))
            low = min(price, close) - abs(np.random.normal(0, 0.0002))
            candles.append(
                {
                    "time": t,
                    "open": round(price, 5),
                    "high": round(high, 5),
                    "low": round(low, 5),
                    "close": round(close, 5),
                    "volume": int(np.random.randint(100, 1000)),
                }
            )
            price = close

        return candles
