# Spotter Rate Intelligence

**ML Engineering Assessment Prototype for Spotter Labs**

A production-oriented freight-rate prediction system built around the exact assessment requirement: learn `posted_rate` from historical loads, validate honestly over time, predict every evaluation load, and generate the required December chart.

## Why this stands out

- Forward temporal backtesting instead of relying on a random split.
- Explicit cold-start evaluation for unseen cities and lanes.
- Freight-specific route, geography, distance, equipment, weight, and calendar features.
- Champion/challenger experiments using LightGBM + CatBoost ensembles.
- Feature-ablation evidence changed the production decision: `market_index` and `quote_signal` looked useful, but excluding them improved every future validation month, so the reduced-feature model is the production champion and the full-signal model remains an offline challenger.
- Empirical prediction intervals, OOD flags, and row-level data-quality diagnostics.
- One reusable inference path for assessment CSVs, REST/API serving, bulk prediction, and the demo UI.
- Automated tests plus the exact provided Spotter scorer.

## Measured ML evidence

Across Aug/Sep/Oct forward validation, mean performance was:

| Model | MAE | WAPE | RMSE |
|---|---:|---:|---:|
| Freight business baseline | 233.25 | 9.83% | 669.52 |
| Full-signal LGBM+CatBoost ensemble | 113.42 | 4.77% | 628.25 |
| **Selected core ensemble** | **101.32** | **4.26%** | **625.89** |

The most recent October fold improved from **$231.30 baseline MAE** to **$105.99 core-ensemble MAE**. Cold-start testing intentionally removes cities from the training categorical universe to measure performance when route names cannot simply be memorized.

## Key modeling judgment

The evaluation period is later than the labeled development period. `market_index` also shifts materially between development and evaluation. Rather than assuming more features are always better, the project uses forward ablation tests. The core model without `market_index` and `quote_signal` generalized better on August, September, and October, so it is selected as the champion. The full-signal model is retained as a challenger for future evidence, not forced into production.

This also gives the December prediction path a clean inference contract because the supplied December chart inputs do not contain those two optional market signals.

## Repository layout

```text
src/spotter_rate_intelligence/  reusable ML, validation, API and inference package
scripts/                        audit, benchmark, training and output scripts
tests/                          data-quality, API and submission-contract tests
reports/                        audit, temporal CV, ablation and OOD evidence
artifacts/                      serialized champion/challenger models
outputs/                        assessment predictions and scorer artifacts
static/                         working product-style demo UI
deploy/                         Cloud Run deployment notes
score.py                        provided Spotter scorer
```

## Reproduce locally

```bash
python -m pip install -r requirements.txt
make audit
make benchmark
make train
make predict
make score
make test
```

Run the working application:

```bash
PYTHONPATH=src uvicorn spotter_rate_intelligence.api:app --host 0.0.0.0 --port 8080
```

Then open `http://localhost:8080`.

## Productization principle

The UI is not a separate demo model. It calls the exact same tested inference package used to generate the official assessment CSV. Bad rows are isolated, recoverable issues are normalized, unsafe inputs are rejected rather than invented, and OOD traffic is surfaced as a warning instead of hidden.

The application is clearly labeled as an **ML Engineering Assessment Prototype**, not an official Spotter Labs product.
