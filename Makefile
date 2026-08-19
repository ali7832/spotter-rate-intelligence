PYTHON ?= python
TRAIN ?= data/train_test.csv
VALIDATION ?= data/validation.csv
TEMPLATE ?= data/validation_predictions_template.csv
DECEMBER ?= data/december_chart_inputs.csv

.PHONY: audit benchmark train analyze predict score test serve

audit:
	PYTHONPATH=src $(PYTHON) scripts/run_audit.py --train $(TRAIN) --validation $(VALIDATION)

benchmark:
	PYTHONPATH=src $(PYTHON) scripts/benchmark.py --train $(TRAIN)

train:
	PYTHONPATH=src $(PYTHON) scripts/train_models.py --train $(TRAIN)

analyze:
	PYTHONPATH=src $(PYTHON) scripts/analyze_model.py

predict:
	PYTHONPATH=src $(PYTHON) scripts/generate_outputs.py --validation $(VALIDATION) --template $(TEMPLATE) --december $(DECEMBER)

score:
	$(PYTHON) score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv --output-dir outputs/scorer_results

test:
	PYTHONPATH=src pytest -q

serve:
	PYTHONPATH=src uvicorn spotter_rate_intelligence.api:app --host 0.0.0.0 --port 8080
