# Engineering Strategy

## Goal
Build the assessment as a defensible machine-learning system first, then expose the exact same inference package through a production-style product surface.

## Priority order
1. Data audit and schema contract.
2. Forward temporal validation and cold-start stress testing.
3. Freight-aware feature engineering and business baselines.
4. Champion/challenger model experiments and error analysis.
5. Full-signal champion plus reduced-feature fallback model.
6. Uncertainty/OOD diagnostics and model metadata.
7. Exact assessment output generation and scorer validation.
8. FastAPI/Docker serving.
9. Cloud Run deployment and polished UI.
10. Monitoring, CI/CD, report, and Loom narrative.

## Non-negotiable ML evidence
- No random split as the primary validation strategy.
- Every feature available at inference time; no leakage from load_id or target-derived future information.
- Separate evaluation for unseen cities/lanes.
- Explicit treatment of missing/negative weight and missing optional signals.
- Multiple model families/baselines compared using MAE, WAPE, RMSE, MAPE, R2, median AE, and p95 AE.
- Error slices by equipment, distance band, time period, OOD state, and data-quality state.
- Final model selection documented from measured evidence, not model popularity.

## Productization principle
The API and UI are adapters around the tested ML package. They never implement separate cleaning or feature logic. Every batch and single prediction uses the same model router, diagnostics, and feature engineering as the assessment CSV generation path.
