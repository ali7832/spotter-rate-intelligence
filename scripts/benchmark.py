from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter_rate_intelligence.training import (
    run_cold_start_benchmark,
    run_fallback_benchmark,
    run_temporal_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.train, parse_dates=["date"])

    temporal, oof, calibration = run_temporal_benchmark(frame)
    temporal.to_csv(ROOT / "reports" / "temporal_benchmark.csv", index=False)
    oof.to_csv(ROOT / "reports" / "temporal_oof_predictions.csv", index=False)
    (ROOT / "reports" / "uncertainty_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )

    cold = run_cold_start_benchmark(frame)
    (ROOT / "reports" / "cold_start_benchmark.json").write_text(
        json.dumps(cold, indent=2), encoding="utf-8"
    )

    fallback = run_fallback_benchmark(frame)
    (ROOT / "reports" / "fallback_benchmark.json").write_text(
        json.dumps(fallback, indent=2), encoding="utf-8"
    )

    print(temporal.to_string(index=False))
    print("\nUncertainty calibration:", json.dumps(calibration, indent=2))
    print("\nCold-start benchmark:", json.dumps(cold, indent=2))
    print("\nFallback benchmark:", json.dumps(fallback, indent=2))


if __name__ == "__main__":
    main()
