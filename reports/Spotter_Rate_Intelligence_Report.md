# Spotter Rate Intelligence

## Machine Learning Engineer Assessment Report

**Project:** Spotter Rate Intelligence\
**Task:** Future freight-rate prediction\
**Target:** `posted_rate`\
**Development period:** 2025-01-01 to 2025-10-31\
**Evaluation period:** 2025-11-01 to 2025-12-31

------------------------------------------------------------------------

## 1. Executive Summary

This project develops a production-oriented machine learning system for
predicting freight load rates from historical transportation data.

The main engineering challenge is that the evaluation data is strictly
later in time than the labeled development data. In addition, the
evaluation set contains unseen cities and lanes, imperfect weight data,
changing missingness, and a substantial distribution shift in
`market_index`.

To reflect the real prediction setting, the project uses forward
temporal validation rather than a random train/test split. Multiple
tabular models were evaluated, including LightGBM, CatBoost, and a 50/50
ensemble.

The final champion is a reduced-feature LightGBM + CatBoost ensemble. It
excludes `market_index` and `quote_signal` from the core feature set
because the reduced-feature model generalized better across every
forward validation month and avoids dependence on unstable optional
signals.

Across August, September, and October forward validation, the champion
achieved:

  Metric         Mean
  -------- ----------
  MAE        \$101.32
  WAPE          4.26%
  RMSE       \$625.89

The business baseline achieved a mean MAE of \$233.25 and mean WAPE of
9.83%.

The resulting system includes reproducible data-quality checks, feature
engineering, temporal validation, cold-start evaluation, uncertainty
calibration, model artifacts, batch prediction, FastAPI serving,
automated tests, Docker support, and CI.

------------------------------------------------------------------------

## 2. Problem Understanding

The assessment requires using `data/train_test.csv` as labeled
development data, choosing a validation strategy, exploring and cleaning
the data, engineering features, training a model, and predicting every
load in `data/validation.csv`.

The validation set contains 12,000 future loads. The final prediction
file must contain:

``` text
load_id,predicted_rate
```

The assessment also requires predictions for a fixed 31-day December
scenario and a PDF/DOCX report containing the validation approach and
December prediction chart.

The project therefore treats this as a **future prediction problem**,
not an IID regression problem.

------------------------------------------------------------------------

## 3. Dataset Overview

The development dataset contains:

-   48,000 labeled rows
-   14 columns
-   Target: `posted_rate`
-   Date range: 2025-01-01 to 2025-10-31

The validation dataset contains:

-   12,000 rows
-   13 columns
-   No target column
-   Date range: 2025-11-01 to 2025-12-31

Important predictive fields include:

-   Pickup city
-   Delivery city
-   Pickup coordinates
-   Delivery coordinates
-   Distance
-   Equipment
-   Weight
-   Date
-   `market_index`
-   `quote_signal`

------------------------------------------------------------------------

## 4. Data Quality Analysis

### 4.1 Development Data

The audit identified:

  Issue                      Rows
  ------------------------ ------
  Missing weight              300
  Negative weight             292
  Missing `market_index`      374
  Missing `quote_signal`        0
  Invalid distance              0
  Unknown equipment             0
  Invalid date                  0
  Duplicate `load_id`           0

### 4.2 Validation Data

  Issue                      Rows
  ------------------------ ------
  Missing weight              165
  Negative weight             145
  Missing `market_index`      249
  Missing `quote_signal`        0
  Invalid distance              0
  Unknown equipment             0
  Invalid date                  0
  Duplicate `load_id`           0

### 4.3 Handling Strategy

Invalid weights are not blindly dropped. The feature pipeline preserves
the underlying observation while creating explicit data-quality
indicators.

This allows the model to distinguish between a normal weight, a missing
weight, and an invalid/negative weight.

The same principle is applied to optional market signals: missingness is
handled explicitly rather than silently replacing unavailable
information with fabricated values.

------------------------------------------------------------------------

## 5. Target Distribution

The training target is strongly right-skewed.

Key statistics:

  Statistic                                 Value
  --------------------------------- -------------
  Mean                                 \$2,373.98
  Median                               \$2,030.76
  1st percentile                         \$327.17
  99th percentile                      \$5,972.83
  99.9th percentile                   \$12,854.56
  Maximum                             \$25,533.00
  Median rate per mile                     \$2.15
  99.9th percentile rate per mile         \$10.07

The long tail is important because a small number of extreme rates have
a large effect on RMSE.

These observations were retained rather than being deleted without
evidence that they were invalid.

------------------------------------------------------------------------

## 6. Geographic Cold-Start Problem

The development data contains 64 cities while the evaluation data
contains 72.

The following eight validation cities are completely unseen during
training:

-   Allentown
-   Charlotte
-   Chicago
-   Jackson
-   Knoxville
-   Laredo
-   Norfolk
-   San Diego

Approximately:

-   12.06% of validation rows involve an unseen city.
-   12.17% involve an unseen lane.

This makes pure route memorization unsafe.

The solution therefore uses geographic coordinates, distance, route
features, and equipment information so that the model can generalize
beyond exact historical city/lane combinations.

------------------------------------------------------------------------

## 7. Distribution Shift

The largest detected numerical shift is in `market_index`.

  Feature            Train Mean   Validation Mean   Standardized Shift
  ---------------- ------------ ----------------- --------------------
  `market_index`         1.0834            0.9269               -0.931
  `distance`            1135.86           1141.77                0.008
  `weight`             31028.84          30506.64               -0.056
  `quote_signal`         2.0625            2.0513               -0.038

`market_index` also has higher missingness in validation:

-   Training missing: 0.78%
-   Validation missing: 2.08%

This is an important example of why historical predictive strength does
not automatically imply future robustness.

------------------------------------------------------------------------

## 8. Feature Engineering

The feature pipeline transforms raw freight fields into business-aware
numerical and categorical representations.

### Geographic Features

-   Pickup latitude/longitude
-   Delivery latitude/longitude
-   Latitude delta
-   Longitude delta
-   Haversine distance
-   Route relationships

### Distance Features

-   Raw distance
-   Log-transformed distance
-   Square-root distance

### Temporal Features

-   Month
-   Day of week
-   Week of year
-   Day of year
-   Weekend indicators
-   Cyclical date representations

### Data Quality Features

-   Missing weight indicator
-   Invalid weight indicator

### Categorical Features

-   Equipment
-   Pickup
-   Delivery
-   Lane-related features

The objective is to provide enough continuous and categorical structure
for the model to learn freight-rate relationships while retaining the
ability to generalize to unseen geography.

------------------------------------------------------------------------

## 9. Validation Strategy

Random cross-validation was not used as the primary model-selection
method.

Instead, the project uses forward temporal folds:

``` text
January–July  → August

January–August → September

January–September → October
```

This ensures that every validation observation occurs after the
corresponding training observations.

The final model is then trained on the full labeled development period
after model selection.

This strategy better approximates the actual assessment scenario, where
November and December loads must be predicted using January--October
history.

------------------------------------------------------------------------

## 10. Baseline

A transparent business baseline was established using an equipment-aware
historical median rate-per-mile approach.

Across the three temporal folds:

  Metric         Mean
  -------- ----------
  MAE        \$233.25
  WAPE          9.83%
  RMSE       \$669.52

This provides an interpretable benchmark for judging whether the machine
learning system adds meaningful predictive value.

------------------------------------------------------------------------

## 11. Model Experiments

The project evaluated:

-   LightGBM with L1 loss
-   CatBoost with MAE loss
-   50/50 LightGBM + CatBoost ensemble

Two feature families were also evaluated:

1.  Full-signal features, including `market_index` and `quote_signal`
2.  Core features that exclude these optional market signals

### Full-Signal Ensemble

Across August, September, and October:

  Metric         Mean
  -------- ----------
  MAE        \$113.42
  WAPE          4.77%
  RMSE       \$628.25

### Core Ensemble

Across August, September, and October:

  Metric         Mean
  -------- ----------
  MAE        \$101.32
  WAPE          4.26%
  RMSE       \$625.89

The core model was consistently stronger.

------------------------------------------------------------------------

## 12. Temporal Model Results

### August

The core ensemble achieved:

-   MAE: \$90.92
-   WAPE: 3.89%
-   RMSE: \$613.45
-   R²: 0.8270
-   Median absolute error: \$25.59
-   95th percentile absolute error: \$150.38

### September

The core ensemble achieved:

-   MAE: \$107.05
-   WAPE: 4.45%
-   RMSE: \$617.83
-   R²: 0.8355
-   Median absolute error: \$33.29
-   95th percentile absolute error: \$220.73

### October

The core ensemble achieved:

-   MAE: \$105.99
-   WAPE: 4.45%
-   RMSE: \$646.38
-   R²: 0.8212
-   Median absolute error: \$32.12
-   95th percentile absolute error: \$154.24

The results remain relatively stable across all three future folds,
supporting the choice of the core ensemble.

------------------------------------------------------------------------

## 13. Champion Model Selection

The final champion is:

``` text
core_ensemble_50_50
```

Architecture:

``` text
50% LightGBM L1
+
50% CatBoost MAE
```

Selection evidence:

  Model                        Mean MAE   Mean WAPE      Mean RMSE
  ---------------------- -------------- ----------- --------------
  Business baseline            \$233.25       9.83%       \$669.52
  Full-signal ensemble         \$113.42       4.77%       \$628.25
  Core ensemble            **\$101.32**   **4.26%**   **\$625.89**

The core model won despite using fewer signals.

The key reason is generalization: the reduced feature set avoids
dependence on the strongly shifted `market_index` and the optional
`quote_signal`.

------------------------------------------------------------------------

## 14. Cold-Start Evaluation

To test generalization beyond seen cities, complete cities were held out
during development.

Held-out cities:

-   Lexington
-   Bakersfield
-   Richmond
-   Oklahoma City
-   Atlanta
-   Mobile
-   Baton Rouge
-   Hartford

The test set contained 1,730 rows.

### Core Champion

-   MAE: \$138.04
-   WAPE: 5.98%
-   RMSE: \$820.85
-   R²: 0.7404
-   Median absolute error: \$37.19
-   95th percentile absolute error: \$168.84

### Full-Signal Challenger

-   MAE: \$157.26
-   WAPE: 6.81%
-   RMSE: \$821.05
-   R²: 0.7403
-   Median absolute error: \$51.94
-   95th percentile absolute error: \$233.54

The core champion again performs better, supporting its use for unseen
geography.

------------------------------------------------------------------------

## 15. Error Analysis

Performance varies by distance and target segment.

### Distance

  Distance Band          MAE    WAPE
  --------------- ---------- -------
  \<=500 miles       \$38.63   4.70%
  501--1000          \$68.63   4.14%
  1001--2000        \$132.72   4.45%
  \>2000            \$190.14   4.03%

Absolute error naturally increases for longer routes because the
underlying rates are larger.

### Equipment

  Equipment          MAE    WAPE
  ----------- ---------- -------
  Dry Van        \$98.56   4.35%
  Flatbed       \$104.98   4.32%
  Reefer        \$104.81   4.07%

### Extreme Target Segment

For the regular 99% of target observations:

-   MAE: \$65.17
-   WAPE: 2.82%
-   R²: 0.9712

For the top 1% highest-rate observations:

-   MAE: \$3,675.27
-   WAPE: 43.34%
-   R²: -1.51

This confirms that rare extreme-rate events dominate the hardest errors
and much of the RMSE.

------------------------------------------------------------------------

## 16. Uncertainty Calibration

Out-of-fold residuals were used to estimate empirical prediction
intervals.

Approximate absolute-error thresholds:

  Coverage     Absolute Error Threshold
  ---------- --------------------------
  80%                           \$76.19
  90%                          \$118.17
  95%                          \$172.94

These intervals allow the serving layer to expose more useful
information than a point prediction alone.

A production response can therefore provide:

-   Predicted rate
-   Empirical uncertainty range
-   Data-quality diagnostics
-   Model information

------------------------------------------------------------------------

## 17. December Prediction Scenario

The assessment provides a fixed December scenario.

Inputs are:

``` text
Pickup: Lexington
Delivery: Fort Wayne
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000 lb
Dates: 2025-12-01 through 2025-12-31
```

The December input file does not contain `market_index` or
`quote_signal`.

The core champion is therefore well suited to this scenario because it
does not depend on those optional signals.

The completed prediction file is:

``` text
outputs/december_chart_inputs.csv
```

The official scorer generates:

``` text
outputs/scorer_results/candidate_december.png
```

This chart should be included in the final submitted report.

------------------------------------------------------------------------

## 18. Submission Output Validation

The project generates:

``` text
outputs/validation_predictions.csv
```

The official scorer validates:

-   Exactly 12,000 prediction rows
-   Expected validation IDs
-   Correct column names and order
-   Numeric finite predictions
-   Positive predicted rates

The December file is also validated for:

-   31 unique dates
-   December 1--31, 2025
-   Lexington pickup
-   Fort Wayne delivery
-   360-mile distance
-   Dry Van equipment
-   32,000 lb weight
-   Positive predicted rates

The scorer successfully validates both prediction outputs and creates
the December chart.

------------------------------------------------------------------------

## 19. Production Architecture

The ML system is structured as a reusable Python package rather than a
single notebook.

``` text
Input Data
    |
    v
Data Quality Layer
    |
    v
Feature Engineering
    |
    v
Champion Model
    |
    +-------------------+
    |                   |
    v                   v
Batch Inference       FastAPI
    |                   |
    v                   v
CSV Outputs        Product Interface
```

The API exposes:

``` text
GET  /health
GET  /ready
GET  /v1/model/info

POST /v1/predict
POST /v1/predict/batch
POST /v1/predict/csv
```

The inference package is shared between batch prediction and API serving
so that production inference does not maintain a separate implementation
from the assessment pipeline.

------------------------------------------------------------------------

## 20. Reproducibility and Engineering

The repository includes:

-   Reusable ML package
-   Training scripts
-   Prediction scripts
-   Data audit
-   Benchmarking
-   Model artifacts
-   Automated tests
-   Docker configuration
-   CI workflow
-   Cloud Run deployment documentation

The repository is designed so that training, prediction, validation, and
testing can be executed through reproducible commands rather than manual
notebook steps.

------------------------------------------------------------------------

## 21. Limitations

The main remaining modeling limitation is extreme-rate prediction.

The model performs strongly on the bulk of freight observations but
struggles with the rarest high-rate events.

Other areas for future improvement include:

-   More explicit probabilistic modeling of extreme rates
-   Better calibrated conditional uncertainty
-   Production monitoring against actual realized rates
-   Automated drift alerts
-   Scheduled retraining
-   More extensive route-level historical statistics
-   Online champion/challenger evaluation

The current model does not attempt to fabricate missing market signals.
This is intentional because the validation data shows meaningful
distribution shift in `market_index`.

------------------------------------------------------------------------

## 22. Conclusion

The final system demonstrates a complete ML engineering approach to
freight-rate prediction.

The most important decisions were not simply algorithm selection. They
were:

1.  Treating the task as future prediction rather than IID regression.
2.  Using forward temporal validation.
3.  Designing features that generalize to unseen geography.
4.  Explicitly handling data-quality problems.
5.  Testing full-signal versus reduced-feature models.
6.  Selecting the core model because it generalized better under
    temporal and cold-start evaluation.
7.  Building reusable inference and serving components around the
    validated ML pipeline.

The final champion is a 50/50 LightGBM + CatBoost core-feature ensemble
with a mean forward-validation MAE of approximately **\$101.32** and
WAPE of approximately **4.26%**.

The solution is therefore positioned not only as an assessment
submission, but as a foundation for a production freight-rate
intelligence service.
