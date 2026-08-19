import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotter_rate_intelligence.data_quality import sanitize_inputs


def test_recoverable_weight_and_equipment():
    frame = pd.DataFrame({
        "pickup": ["Dallas"],
        "delivery": ["Atlanta"],
        "distance": [800],
        "equipment": ["dry van"],
        "weight": [-32000],
        "date": ["2025-12-01"],
    })
    clean, diagnostics = sanitize_inputs(frame)
    assert clean.loc[0, "equipment"] == "Dry Van"
    assert clean.loc[0, "weight"] == 32000
    assert diagnostics.loc[0, "status"] == "RECOVERED_WITH_WARNING"


def test_invalid_distance_is_rejected():
    frame = pd.DataFrame({
        "pickup": ["Dallas"], "delivery": ["Atlanta"], "distance": [0],
        "equipment": ["Dry Van"], "weight": [32000], "date": ["2025-12-01"],
    })
    _, diagnostics = sanitize_inputs(frame)
    assert diagnostics.loc[0, "status"] == "INVALID_INPUT"
