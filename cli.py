"""CLI Interface for Quotex Signal Generator.

Run Quotex signal analysis against simulated demo candles or JSON input files.
"""

import argparse
import json
import sys
from typing import List, Dict, Any

from generator import QuotexSignalGenerator, Signal


def load_candles_from_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "candles" in data:
            return data["candles"]
        else:
            raise ValueError("JSON file must contain a list of candles or object with 'candles' key.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quotex Signal Generator CLI - Institutional SMC Confluence Engine"
    )
    parser.add_argument(
        "--pair", type=str, default="EURUSD_otc", help="Asset symbol / pair name"
    )
    parser.add_argument(
        "--file", type=str, help="Path to JSON file containing candle data"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Generate signals using simulated demo candles"
    )
    parser.add_argument(
        "--trend",
        type=str,
        choices=["up", "down"],
        default="up",
        help="Trend bias for simulated demo candles",
    )
    parser.add_argument(
        "--candles-count",
        type=int,
        default=150,
        help="Number of demo candles to generate",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=70,
        help="Minimum signal confluence score threshold (default: 70)",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.55,
        help="Minimum ML model confidence threshold (default: 0.55)",
    )

    args = parser.parse_args()

    generator = QuotexSignalGenerator(
        {"score_medium": args.min_score, "model_min_conf": args.min_conf}
    )

    if args.file:
        print(f"Loading candles from '{args.file}'...")
        candles = load_candles_from_file(args.file)
    else:
        print(f"Generating {args.candles_count} simulated '{args.trend}' demo candles for {args.pair}...")
        candles = QuotexSignalGenerator.generate_demo_candles(
            count=args.candles_count, trend=args.trend
        )

    print(f"Loaded {len(candles)} candles. Running SMC Confluence & ML Model Analysis...\n")

    # Analyze full history progressively or on complete set
    signal: Signal = generator.analyze(pair=args.pair, entry_candles=candles)

    if signal:
        print("==========================================")
        print("  QUOTEX SIGNAL GENERATED SUCCESSFUL ")
        print("==========================================")
        print(f"Pair:            {signal.pair}")
        print(f"Action:          {signal.action} ({signal.direction})")
        print(f"Score:           {signal.score}/110 ({signal.strength})")
        print(f"Current Price:   {signal.price:.5f}")
        print(f"Expiration:      {signal.expiration_seconds}s (Next Candle)")
        if signal.model_confidence:
            print(f"ML Confidence:   {signal.model_confidence * 100:.1f}%")
        print("\nConfluence Reasons:")
        for reason in signal.reasons:
            print(f"  ✔ {reason}")
        print("==========================================")
    else:
        print(
            "No valid signal generated for the given candles under current risk & confluence thresholds."
        )


if __name__ == "__main__":
    main()
