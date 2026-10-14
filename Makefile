.DEFAULT_GOAL := help

.PHONY: help sync lint format format-check typecheck test check docs docs-check docs-record docs-notebooks docs-mkdocs docs-serve docs-deploy build clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

sync: ## Create/refresh the dev environment (incl. docs deps)
	uv sync --group docs

lint: ## Lint with ruff
	uv run ruff check .

format: ## Auto-format and autofix with ruff
	uv run ruff format .
	uv run ruff check --fix .

format-check: ## Verify formatting without rewriting
	uv run ruff format --check .

typecheck: ## Type-check with mypy
	uv run mypy

test: ## Run the test suite
	uv run pytest

check: lint format-check typecheck test docs-check ## Lint + format check + type-check + test + docs check

docs: ## Build the Graphite docs site into site/ (v<version>/ plus the current root)
	uv run python -m scripts.docsite build

docs-check: ## Run every page program; check excerpts, recorded outputs and links
	uv run python -m scripts.docsite check
	uv run python -m scripts.docsite build --output build/docs-check
	uv run python -m scripts.docsite links build/docs-check

docs-record: ## Rerun page programs and write their output panels (review the diff)
	uv run python -m scripts.docsite record

docs-notebooks: ## Execute every learn and case-study notebook (release CI; slow)
	uv run --group docs python -m scripts.docsite notebooks

docs-mkdocs: ## Build the legacy MkDocs site (retires when the Graphite site goes live)
	uv run --group docs mkdocs build --strict

docs-serve: ## Serve the legacy MkDocs site locally with live reload
	uv run --group docs mkdocs serve

docs-deploy: ## Build and deploy the legacy MkDocs site to Cloud Run
	deploy/docs/deploy.sh

build: ## Build sdist + wheel
	uv build

clean: ## Remove caches and build artifacts
	rm -rf dist build site htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
