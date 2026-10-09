"""CLI Interface for Quotex Signal Generator.

Run Quotex signal analysis against live market data, simulated demo candles, or launch Web Dashboard.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional

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


async def run_live_scanner(
    pairs: List[str], min_score: int, min_conf: float, web_port: Optional[int] = None
) -> None:
    """Connect to Quotex live API and continuously scan market pairs for signal opportunities."""
    email = os.getenv("QUOTEX_EMAIL")
    password = os.getenv("QUOTEX_PASSWORD")
    if not email or not password:
        print("Error: QUOTEX_EMAIL and QUOTEX_PASSWORD environment variables are required for live mode.")
        print("Example usage:")
        print("  QUOTEX_EMAIL='user@example.com' QUOTEX_PASSWORD='secret' python3 cli.py --live --pairs EURUSD_otc,USDJPY_otc")
        sys.exit(1)

    try:
        from pyquotex.stable_api import Quotex
        from bot import apply_session_patch, HOST
    except ImportError as e:
        print(f"Error importing pyquotex dependencies: {e}")
        sys.exit(1)

    if web_port:
        from app import run_web_server, add_signal_to_dashboard
        import threading
        web_server = run_web_server(web_port)
        threading.Thread(target=web_server.serve_forever, daemon=True).start()

    apply_session_patch(HOST)
    q = Quotex(email=email, password=password, lang="en", host=HOST)
    q.set_account_mode("PRACTICE")
    print(f"Connecting to Quotex live API host ({HOST})...")
    ok, reason = await q.connect()
    if not ok:
        print(f"Connection failed: {reason}")
        sys.exit(1)

    balance = await q.get_balance()
    print(f"Connected successfully! Practice Balance: ${balance:.2f}")
    print(f"Live Market Scanning active for pairs: {', '.join(pairs)}")
    print("Press Ctrl+C to stop scanning.\n")

    generator = QuotexSignalGenerator({"score_medium": min_score, "model_min_conf": min_conf})

    while True:
        for pair in pairs:
            try:
                raw_candles = await q.get_candles(pair, time.time(), 3000, 60)
                candles = [c for c in (raw_candles or []) if c.get("close") is not None]
                if len(candles) < 30:
                    continue

                signal: Optional[Signal] = generator.analyze(pair=pair, entry_candles=candles)
                if signal:
                    if web_port:
                        add_signal_to_dashboard(signal)

                    print("==========================================")
                    print(" 🔥 LIVE QUOTEX SIGNAL DETECTED 🔥")
                    print("==========================================")
                    print(f"Pair:            {signal.pair}")
                    print(f"Action:          {signal.action} ({signal.direction})")
                    print(f"Entry Time:      {signal.entry_time_utc}")
                    print(f"Entry Price:     {signal.price:.5f}")
                    print(f"Expiration:      {signal.expiration_seconds}s (Next Candle)")
                    print(f"Score:           {signal.score}/110 ({signal.strength})")
                    if signal.model_confidence:
                        print(f"ML Confidence:   {signal.model_confidence * 100:.1f}%")
                    print("\nConfluence Reasons:")
                    for reason in signal.reasons:
                        print(f"  ✔ {reason}")
                    print("==========================================")
            except Exception as err:  # noqa: BLE001
                print(f"Error fetching candles for {pair}: {err}")

            await asyncio.sleep(0.5)

        await asyncio.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quotex Signal Generator CLI - Institutional SMC Confluence Engine"
    )
    parser.add_argument(
        "--live", action="store_true", help="Connect to Quotex live market stream for real-time analysis"
    )
    parser.add_argument(
        "--web", action="store_true", help="Launch Web Dashboard server on http://localhost:8080"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for Web Dashboard server (default: 8080)"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="EURUSD_otc,USDJPY_otc,GBPUSD_otc,AUDUSD_otc",
        help="Comma-separated pairs for live scanning (e.g., EURUSD_otc,USDJPY_otc)",
    )
    parser.add_argument(
        "--pair", type=str, default="EURUSD_otc", help="Asset symbol / pair name for single analysis"
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

    if args.web and not args.live:
        from app import run_web_server, start_demo_signal_stream
        print(f"Starting Web Dashboard at http://localhost:{args.port}...")
        server = run_web_server(args.port)
        asyncio.run(start_demo_signal_stream())
        return

    if args.live:
        pair_list = [p.strip() for p in args.pairs.split(",")]
        web_port = args.port if args.web else None
        try:
            asyncio.run(run_live_scanner(pair_list, args.min_score, args.min_conf, web_port=web_port))
        except KeyboardInterrupt:
            print("\nLive scanning stopped by user.")
        return

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

    signal: Optional[Signal] = generator.analyze(pair=args.pair, entry_candles=candles)

    if signal:
        print("==========================================")
        print("  QUOTEX SIGNAL GENERATED SUCCESSFULLY ")
        print("==========================================")
        print(f"Pair:            {signal.pair}")
        print(f"Action:          {signal.action} ({signal.direction})")
        print(f"Entry Time:      {signal.entry_time_utc}")
        print(f"Entry Price:     {signal.price:.5f}")
        print(f"Expiration:      {signal.expiration_seconds}s (Next Candle)")
        print(f"Score:           {signal.score}/110 ({signal.strength})")
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
