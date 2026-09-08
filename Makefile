.PHONY: data pipeline evaluate serve test lint typecheck build clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

data:  ## Download raw UCI dataset to data/raw/
	python scripts/fetch_data.py

pipeline:  ## Run full pipeline (preprocess → train → cluster → UMAP → artifacts)
	python scripts/run_pipeline.py

evaluate:  ## Run evaluation (reconstruction, cluster metrics, downstream benchmark)
	python scripts/evaluate.py

serve:  ## Launch FastAPI dev server on :8000
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:  ## Run test suite
	pytest tests/

lint:  ## Lint with ruff
	ruff check src/ app/ tests/ scripts/

typecheck:  ## Type-check with mypy
	mypy src/ app/

build:  ## Build Docker image
	docker compose build

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true