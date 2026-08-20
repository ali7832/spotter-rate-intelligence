# Spotter Rate Intelligence

## Production-Grade Freight Rate Prediction System

A machine learning system that predicts freight load rates using historical transportation data.  
The project was developed for the Spotter Machine Learning Engineer Assessment and designed beyond a simple model submission — as a reproducible, production-oriented ML pipeline with validation, inference, API serving, testing, and deployment considerations.

---

# 1. Problem Overview

Freight pricing is influenced by many interacting factors:

- Route geography
- Distance
- Equipment type
- Weight
- Time trends
- Market conditions
- Historical pricing patterns

The objective is to build a machine learning system that predicts:

posted_rate



for future freight loads.

The challenge is not only achieving good prediction accuracy but also building a system that can:

- Generalize to future time periods
- Handle unseen routes/cities
- Handle imperfect real-world data
- Produce reproducible predictions
- Support production inference

---

# 2. Solution Overview

The complete ML pipeline:


             Historical Freight Data
                      |
                      v
          Data Quality Validation Layer
                      |
                      v
          Feature Engineering Pipeline
                      |
                      v
         Temporal Validation Framework
                      |
                      v
          ML Model Training Pipeline
                      |
                      v
          Champion Rate Prediction Model
                      |
          +-----------+------------+
          |                        |
          v                        v
   Batch Predictions          API Inference
          |                        |
          v                        v
 CSV Submission Output      Product Interface


---

# 3. Key Engineering Decisions

## Temporal Validation Instead of Random Split

Freight pricing is a time-dependent problem.

A random train/test split would allow future information patterns to leak into validation.

Instead, the project uses forward validation:

Train:
January → July

Validate:
August

Train:
January → August

Validate:
September

Train:
January → September

Validate:
October



This better represents real deployment behavior:

Historical Loads
|
v
Future Rate Prediction



---

# 4. Dataset

The development dataset contains historical freight loads.

Features include:

| Feature | Description |
|---|---|
| pickup | Origin city |
| delivery | Destination city |
| pickup_lat/lon | Origin coordinates |
| delivery_lat/lon | Destination coordinates |
| distance | Route distance |
| equipment | Truck equipment type |
| weight | Load weight |
| date | Load date |
| market_index | Market signal |
| quote_signal | Pricing signal |

Target:

posted_rate



---

# 5. Data Quality Handling

Real freight data contains imperfect records.

The pipeline handles:

## Missing Values

Examples:

- Missing weight
- Missing market signals
- Missing geographic information

Strategy:

- Preserve missingness information
- Add missing indicators
- Apply safe preprocessing

---

## Invalid Weight Values

Detected:

- Missing weights
- Negative weights

Instead of deleting records:

Invalid Data
|
v
Quality Flags
|
v
Model Awareness



The model learns that data quality itself can be informative.

---

# 6. Feature Engineering

The feature layer creates business-aware features.

## Geographic Features

Added:

- Latitude difference
- Longitude difference
- Haversine distance
- Route relationships

These help generalize to unseen cities and lanes.

---

## Distance Features

Created:

- Raw distance
- Log distance
- Square-root distance

Freight pricing is nonlinear, so transformed distance improves learning.

---

## Time Features

Created:

- Month
- Day of week
- Week number
- Day of year
- Weekend indicator
- Cyclical date encoding

---

## Data Quality Features

Added:

- Weight missing flag
- Invalid weight flag

---

# 7. Cold Start Handling

A major challenge was unseen geography.

Training data contains fewer cities than validation data.

A model based only on:

pickup + delivery



would fail on new routes.

Therefore the system uses:

- Geographic coordinates
- Route features
- Equipment information
- Distance behavior

This allows prediction even when a lane has not appeared historically.

---

# 8. Model Strategy

Multiple approaches were evaluated.

## Baseline Model

Business-style baseline:

- Equipment-aware rate-per-mile estimation
- Median historical pricing behavior

Purpose:

Create an interpretable benchmark.

---

## Machine Learning Models

Evaluated:

- LightGBM
- CatBoost
- Ensemble approaches

Final architecture:

50% LightGBM
+
50% CatBoost



The ensemble combines:

- Gradient boosting performance
- Strong tabular learning
- Nonlinear relationship handling

---

# 9. Feature Ablation Experiment

Two model families were compared.

## Full Signal Model

Includes:

market_index
quote_signal



## Core Production Model

Uses:

Operational freight features
+
geography
+
time
+
equipment
+
distance



The core model performed better on future validation.

Decision:

The core model became the production champion because it:

- Generalizes better
- Handles missing optional signals
- Works with December inference data
- Avoids dependence on unstable market variables

---

# 10. Model Performance

The final champion model achieved:

| Metric | Result |
|---|---:|
| Mean MAE | ~$101 |
| Mean WAPE | ~4.26% |
| Mean RMSE | ~$626 |

October future validation:

| Metric | Result |
|---|---:|
| MAE | ~$106 |
| WAPE | ~4.45% |
| R² | ~0.82 |

The model significantly improves over the business baseline.

---

# 11. Error Analysis

Freight rates contain extreme pricing events.

The target distribution is long-tailed:

- Most loads are predictable
- Rare expensive loads create large RMSE errors

Analysis showed:

- Normal freight loads achieve strong accuracy
- Extreme high-rate events remain challenging

Instead of removing these cases, the project uses robust learning approaches.

---

# 12. Prediction Uncertainty

The system also estimates prediction confidence.

Example output:

json
{
  "predicted_rate": 1850,
  "confidence_range": {
      "90_percent": [
          1732,
          1968
      ]
  }
}

This provides more business value than returning only a single number.

# 13. Production Architecture

The ML package is designed for deployment.

Components:


                User / CSV
                    |
                    v
              FastAPI Service
                    |
                    v
          Inference Pipeline
                    |
                    v
           Champion ML Model
                    |
          +---------+---------+
          |                   |
          v                   v
 Prediction              Diagnostics
# 14. API Features
Available endpoints:

Health

GET /health
Model Information

GET /v1/model/info
Single Prediction

POST /v1/predict
Batch Prediction

POST /v1/predict/batch
CSV Prediction

POST /v1/predict/csv
# 15. Project Structure

spotter-rate-intelligence/

├── src/
│   └── spotter_rate_intelligence/
│       ├── api.py
│       ├── features.py
│       ├── inference.py
│       ├── training.py
│       ├── model.py
│       └── data_quality.py
│
├── scripts/
│   ├── train_models.py
│   ├── generate_outputs.py
│   ├── run_audit.py
│   └── benchmark.py
│
├── tests/
│
├── artifacts/
│   ├── champion_model.joblib
│   └── challenger_model.joblib
│
├── reports/
│
├── outputs/
│
├── docs/
│
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md

# 16. Running Locally
Create Environment
Bash

python -m venv .venv
Activate:

Windows:

PowerShell

.\.venv\Scripts\Activate.ps1
Linux/Mac:

Bash

source .venv/bin/activate
Install Dependencies
Bash

pip install -r requirements.txt
# 17. Train Models
Bash

python scripts/train_models.py \
--train data/train_test.csv
# 18. Generate Predictions
Bash

python scripts/generate_outputs.py
Outputs:


outputs/
├── validation_predictions.csv
└── december_chart_inputs.csv
# 19. Validate Submission
Run Spotter scorer:

Bash

python score.py \
--predictions outputs/validation_predictions.csv \
--december-predictions outputs/december_chart_inputs.csv
Expected:


Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart.
# 20. Testing
Run:

Bash

pytest -q
Expected:


6 passed
# 21. Docker
Build:

Bash

docker build -t spotter-rate-intelligence .
Run:

Bash

docker run -p 8000:8000 spotter-rate-intelligence
API:


http://localhost:8000
# 22. CI/CD
GitHub Actions automatically validates:

Environment setup

Dependency installation

Automated tests

Prediction contract checks

# 23. Future Improvements
Production expansion roadmap:

Monitoring
Data drift monitoring

Prediction drift

Error tracking

Platform Features
User authentication

Prediction history

Route analytics

Dashboard

Automated retraining

ML Improvements
More granular uncertainty modeling

Online learning

Champion/challenger deployment

Automated model promotion
