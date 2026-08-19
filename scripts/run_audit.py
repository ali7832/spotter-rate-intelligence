from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter_rate_intelligence.training import create_data_audit, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default=str(ROOT / "reports" / "data_audit.json"))
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    audit = create_data_audit(train, validation)
    write_json(audit, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
