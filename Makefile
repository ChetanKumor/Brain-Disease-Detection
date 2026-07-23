.PHONY: help install install-dev test cov lint format typecheck check run docker-build docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies (incl. TensorFlow)
	pip install -r requirements.txt

install-dev: ## Install development/test dependencies (no TensorFlow)
	pip install -r requirements-dev.txt

test: ## Run the test suite
	pytest

cov: ## Run tests with a coverage report
	pytest --cov=brain_disease_detection --cov-report=term-missing

lint: ## Lint with ruff
	ruff check .

format: ## Auto-format with ruff
	ruff format .

typecheck: ## Static type-check with mypy
	mypy

check: lint typecheck test ## Run lint, type-check, and tests

run: ## Run the web app locally
	python wsgi.py

docker-build: ## Build the production Docker image
	docker build -t brain-disease-detection .

docker-run: ## Run the container (mounts ./models)
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models:ro brain-disease-detection

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist *.egg-info
