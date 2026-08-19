from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter_rate_intelligence.model import RateModelBundle
from spotter_rate_intelligence.inference import RatePredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--december", required=True)
    args = parser.parse_args()

    champion = RateModelBundle.load(ROOT / "artifacts" / "champion_model.joblib")
    challenger = RateModelBundle.load(ROOT / "artifacts" / "challenger_full_model.joblib")
    predictor = RatePredictor(champion, challenger)

    validation = pd.read_csv(args.validation)
    validation_result = predictor.predict(validation)
    pred_map = dict(zip(validation["load_id"], validation_result.predictions["predicted_rate"]))
    template = pd.read_csv(args.template)
    template["predicted_rate"] = template["load_id"].map(pred_map)
    template.to_csv(ROOT / "outputs" / "validation_predictions.csv", index=False)
    validation_result.diagnostics.assign(load_id=validation["load_id"]).to_csv(
        ROOT / "outputs" / "validation_diagnostics.csv", index=False
    )

    december = pd.read_csv(args.december)
    december_result = predictor.predict(december)
    december["predicted_rate"] = december_result.predictions["predicted_rate"].to_numpy()
    december.to_csv(ROOT / "outputs" / "december_chart_inputs.csv", index=False)
    december_result.diagnostics.to_csv(ROOT / "outputs" / "december_diagnostics.csv", index=False)
    print("Generated validation and December outputs with champion core model.")


if __name__ == "__main__":
    main()
