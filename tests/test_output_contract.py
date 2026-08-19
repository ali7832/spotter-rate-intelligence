from pathlib import Path

import pandas as pd


def test_validation_output_contract():
    path = Path(__file__).resolve().parents[1] / "outputs" / "validation_predictions.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["load_id", "predicted_rate"]
    assert len(frame) == 12000
    assert frame["load_id"].is_unique
    assert frame["predicted_rate"].notna().all()
    assert (frame["predicted_rate"] > 0).all()
