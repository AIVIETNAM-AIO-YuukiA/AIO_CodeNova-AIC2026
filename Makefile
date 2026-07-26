# CodeNova — các lệnh dùng chung. Mọi thứ chạy qua uv.
# Ghi đè biến pipeline trên dòng lệnh, ví dụ:
#   make ingest EXP=demo INPUT=data/raw_videos

.DEFAULT_GOAL := help

EXP         ?= demo
INPUT       ?= data/raw_videos
TOPK        ?= 20
QUERY       ?= a person riding a motorbike
RESUME      ?=          # set 1 để resume 1 experiment đã có (vd: make pipeline RESUME=1)
# HOST/PORT để trống -> lấy mặc định từ CODENOVA_UI_HOST/PORT trong .env
HOST        ?=
PORT        ?=
TN2_DIR     ?= external/TransNetV2/inference-pytorch
TN2_WEIGHTS ?= $(TN2_DIR)/transnetv2-pytorch-weights.pth

.PHONY: help sync install-hooks lint format check test pre-commit precommit \
        qdrant-up qdrant-down qdrant-health \
        elasticsearch-up elasticsearch-health \
        atlas-index-up atlas-index-down atlas-index-health \
        agent-up agent-down agent-health \
        ingest detect-shots extract-frames embed-frames build-index extract-text export-text import-text pipeline \
        search serve-ui clean-runs clean

help: ## Hiện danh sách lệnh này
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- Môi trường ---
sync: ## Cài/đồng bộ dependency bằng uv
	uv sync

install-hooks: ## Cài pre-commit git hook
	uv run pre-commit install

## --- Kiểm tra chất lượng code ---
lint: ## Lint bằng ruff
	uv run ruff check src tests

format: ## Tự động format bằng ruff
	uv run ruff format src tests

check: lint ## Lint + kiểm tra format (dùng cho CI gate)
	uv run ruff format --check src tests

test: ## Chạy unit test
	uv run pytest -q

pre-commit: ## Chạy toàn bộ pre-commit hook
	uv run pre-commit run --all-files

precommit: pre-commit ## Alias của pre-commit

## --- Qdrant (vector DB) ---
qdrant-up: ## Khởi động Qdrant qua docker compose
	docker compose up -d qdrant

qdrant-down: ## Dừng Qdrant
	docker compose down

qdrant-health: ## Kiểm tra Qdrant có sống không
	curl -sf http://localhost:6333/healthz && echo

## --- Elasticsearch (full-text index cho OCR/ASR) ---
elasticsearch-up: ## Khởi động Elasticsearch + Elasticvue qua docker compose
	docker compose up -d elasticsearch elasticvue

elasticsearch-health: ## Kiểm tra Elasticsearch có sống không
	curl -sf http://localhost:8882 && echo

## --- Atlas captioning/OCR (Qwen3.6-35B-A3B NVFP4, chỉ DGX Spark/GB10, port 8881) ---
atlas-index-up: ## Khởi động Atlas cho captioning/OCR offline
	docker compose --profile atlas-index up -d atlas-index

atlas-index-down: ## Dừng atlas-index
	docker compose stop atlas-index
	docker compose rm -f atlas-index

atlas-index-health: ## Kiểm tra atlas-index có sống không
	curl -sf http://localhost:8881/v1/models && echo

## --- Agent LLM (Qwen3.5-4B, port 8888; engine chọn theo phần cứng) ---
agent-up: ## Khởi động agent LLM: Atlas nếu là DGX Spark/GB10, ngược lại llama.cpp
	@if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GB10; then \
		echo "GB10 detected -> Atlas (NVFP4 4-bit)"; \
		docker compose --profile atlas-agent up -d atlas-agent; \
	else \
		echo "Non-GB10 GPU -> llama.cpp (GGUF Q4_K_M)"; \
		docker compose --profile llamacpp-agent up -d llamacpp-agent; \
	fi

agent-down: ## Dừng agent LLM (cả 2 engine)
	docker compose stop atlas-agent llamacpp-agent 2>/dev/null || true
	docker compose rm -f atlas-agent llamacpp-agent 2>/dev/null || true

agent-health: ## Kiểm tra agent LLM có sống không
	curl -sf http://localhost:8888/v1/models && echo

## --- Pipeline index offline ---
ingest: ## Quét video (INPUT, EXP; RESUME=1 để dùng lại experiment đã có)
	uv run codenova ingest --input $(INPUT) --experiment-name $(EXP) $(if $(RESUME),--resume)

detect-shots: ## Phát hiện shot bằng TransNetV2 (EXP)
	uv run codenova detect-shots --experiment-name $(EXP) \
		--transnetv2-module-dir $(TN2_DIR) \
		--transnetv2-weights $(TN2_WEIGHTS)

extract-frames: ## Trích keyframe (EXP)
	uv run codenova extract-frames --experiment-name $(EXP)

embed-frames: ## Embed keyframe bằng BEiT-3 large (EXP; EMBEDDING_MODELS để thêm siglip2/vietnamese)
	uv run codenova embed-frames --experiment-name $(EXP)

build-index: ## Build index Qdrant (EXP); cần Qdrant đang chạy
	uv run codenova build-index --experiment-name $(EXP)

extract-text: ## Chạy OCR + ASR, index vào Elasticsearch (EXP); cần Elasticsearch + `make atlas-index-up` đang chạy
	uv run codenova extract-text --experiment-name $(EXP)

export-text: ## Xuất document OCR/ASR từ Elasticsearch ra runs/<exp>/manifests/text.jsonl (EXP)
	uv run codenova export-text --experiment-name $(EXP)

import-text: ## Nạp runs/<exp>/manifests/text.jsonl vào Elasticsearch (EXP); cần Elasticsearch đang chạy
	uv run codenova import-text --experiment-name $(EXP)

pipeline: ingest detect-shots extract-frames embed-frames build-index ## Chạy toàn bộ pipeline offline (chỉ vector index; extract-text chạy riêng cho OCR/ASR)

## --- Retrieval online ---
search: ## Tìm kiếm bằng text (EXP, QUERY, TOPK)
	uv run codenova search --experiment-name $(EXP) --top-k $(TOPK) "$(QUERY)"

serve-ui: ## Chạy UI retrieval local (EXP; HOST/PORT ghi đè .env)
	uv run codenova serve-ui --experiment-name $(EXP) $(if $(HOST),--host $(HOST)) $(if $(PORT),--port $(PORT))

## --- Dọn dẹp ---
clean-runs: ## Xóa 1 thư mục run (EXP)
	rm -rf runs/$(EXP)

clean: ## Xóa cache (.venv, __pycache__, .ruff_cache)
	rm -rf .venv .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
