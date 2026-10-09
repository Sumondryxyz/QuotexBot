"""Unit tests for Quotex Signal Generator core module."""

import unittest

from generator import (
    QuotexSignalGenerator,
    Signal,
    swing_points,
    market_structure,
    inducement_filter,
    find_order_block,
    fair_value_gaps,
    premium_discount,
    model_predict,
    train_model,
)


class TestQuotexSignalGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = QuotexSignalGenerator()

    def test_demo_candles_generation(self):
        candles = QuotexSignalGenerator.generate_demo_candles(count=50, trend="up")
        self.assertEqual(len(candles), 50)
        self.assertIn("open", candles[0])
        self.assertIn("high", candles[0])
        self.assertIn("low", candles[0])
        self.assertIn("close", candles[0])

    def test_swing_points(self):
        candles = [
            {"high": 1.0, "low": 0.5},
            {"high": 1.1, "low": 0.6},
            {"high": 1.5, "low": 0.7},  # Swing high at idx 2
            {"high": 1.2, "low": 0.6},
            {"high": 1.0, "low": 0.3},  # Swing low at idx 4
            {"high": 1.1, "low": 0.5},
            {"high": 1.2, "low": 0.6},
        ]
        points = swing_points(candles, lookback=2)
        self.assertTrue(any(p == (2, "high", 1.5) for p in points))
        self.assertTrue(any(p == (4, "low", 0.3) for p in points))

    def test_market_structure(self):
        candles = [
            {"close": 1.0},
            {"close": 1.2},
            {"close": 1.5},
            {"close": 1.1},
            {"close": 0.8},
            {"close": 1.6},  # Confirmed BOS_UP breaking 1.5
            {"close": 1.7},
        ]
        points = [(2, "high", 1.5), (4, "low", 0.8), (5, "high", 1.6)]
        events, sweeps = market_structure(candles, points)
        self.assertTrue(len(events) > 0)
        self.assertEqual(events[0][1], "BOS_UP")

    def test_inducement_filter(self):
        candles = [
            {"close": 1.0},
            {"close": 1.2},
            {"close": 1.5},
            {"close": 1.1},
            {"close": 1.6},  # Break at idx 4
            {"close": 1.0},  # Fails immediately at idx 5
        ]
        events = [(4, "BOS_UP")]
        clean = inducement_filter(candles, events, lookahead=1)
        self.assertEqual(len(clean), 0)

    def test_order_block_and_fvg(self):
        candles = QuotexSignalGenerator.generate_demo_candles(count=40, seed=123)
        ob, disp = find_order_block(candles, break_idx=20, bullish=True, displacement_mult=0.1)
        if ob:
            self.assertIn("lo", ob)
            self.assertIn("hi", ob)

        gaps = fair_value_gaps(candles, bullish=True, start=0)
        self.assertIsInstance(gaps, list)

    def test_premium_discount(self):
        points = [(2, "high", 2.0), (4, "low", 1.0)]
        candles = [{"close": 1.2}]
        zone = premium_discount(candles, points, price=1.2)
        self.assertEqual(zone, "discount")

        zone_prem = premium_discount(candles, points, price=1.8)
        self.assertEqual(zone_prem, "premium")

    def test_ml_model_training_and_prediction(self):
        candles = QuotexSignalGenerator.generate_demo_candles(count=100, seed=42)
        model = train_model(candles)
        self.assertIsNotNone(model)
        if model:
            prob = model_predict(model, candles)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_signal_formatting(self):
        sig = Signal(
            pair="EURUSD_otc",
            direction="UP",
            score=95,
            strength="STRONG",
            reasons=["BOS", "Order Block Retest"],
            price=1.0850,
            timestamp=1700000000,
            model_confidence=0.75,
        )
        self.assertEqual(sig.action, "BUY")
        self.assertIn("BUY", sig.formatted_message)
        self.assertIn("EURUSD_otc", sig.formatted_message)


if __name__ == "__main__":
    unittest.main()
