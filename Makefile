# CodeNova — common tasks. Everything runs through uv.
# Override pipeline vars on the command line, e.g.:
#   make ingest EXP=demo INPUT=data/raw_videos

.DEFAULT_GOAL := help

EXP         ?= demo
INPUT       ?= data/raw_videos
TOPK        ?= 20
QUERY       ?= a person riding a motorbike
RESUME      ?=          # set to 1 to resume an existing experiment (e.g. make pipeline RESUME=1)
# HOST/PORT empty -> fall back to CODENOVA_UI_HOST/PORT in .env (CLI defaults)
HOST        ?=
PORT        ?=
TN2_DIR     ?= external/TransNetV2/inference-pytorch
TN2_WEIGHTS ?= $(TN2_DIR)/transnetv2-pytorch-weights.pth

.PHONY: help sync install-hooks lint format check test pre-commit \
        qdrant-up qdrant-down qdrant-health \
        ingest detect-shots extract-frames embed-frames build-index pipeline \
        search serve-ui clean-runs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- Environment ---
sync: ## Install/sync dependencies with uv
	uv sync

install-hooks: ## Install the pre-commit git hook
	uv run pre-commit install

## --- Quality ---
lint: ## Lint with ruff
	uv run ruff check src tests

format: ## Auto-format with ruff
	uv run ruff format src tests

check: lint ## Lint + format-check (CI gate)
	uv run ruff format --check src tests

test: ## Run unit tests
	uv run pytest -q

pre-commit: ## Run all pre-commit hooks
	uv run pre-commit run --all-files

## --- Qdrant (vector DB) ---
qdrant-up: ## Start Qdrant via docker compose
	docker compose up -d qdrant

qdrant-down: ## Stop Qdrant
	docker compose down

qdrant-health: ## Check Qdrant health
	curl -sf http://localhost:6333/healthz && echo

## --- Offline indexing pipeline ---
ingest: ## Discover videos (INPUT, EXP; RESUME=1 to reuse an existing run)
	uv run codenova ingest --input $(INPUT) --experiment-name $(EXP) $(if $(RESUME),--resume)

detect-shots: ## Detect shots with TransNetV2 (EXP)
	uv run codenova detect-shots --experiment-name $(EXP) \
		--transnetv2-module-dir $(TN2_DIR) \
		--transnetv2-weights $(TN2_WEIGHTS)

extract-frames: ## Extract keyframes (EXP)
	uv run codenova extract-frames --experiment-name $(EXP)

embed-frames: ## Embed keyframes with CLIP (EXP)
	uv run codenova embed-frames --experiment-name $(EXP)

build-index: ## Build the Qdrant index (EXP); needs Qdrant running
	uv run codenova build-index --experiment-name $(EXP)

pipeline: ingest detect-shots extract-frames embed-frames build-index ## Run the full offline pipeline

## --- Online retrieval ---
search: ## Search by text (EXP, QUERY, TOPK)
	uv run codenova search --experiment-name $(EXP) --top-k $(TOPK) "$(QUERY)"

serve-ui: ## Serve the retrieval UI (EXP; HOST/PORT override .env)
	uv run codenova serve-ui --experiment-name $(EXP) $(if $(HOST),--host $(HOST)) $(if $(PORT),--port $(PORT))

## --- Housekeeping ---
clean-runs: ## Remove a run directory (EXP)
	rm -rf runs/$(EXP)

clean: ## Remove caches (.venv, __pycache__, .ruff_cache)
	rm -rf .venv .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
