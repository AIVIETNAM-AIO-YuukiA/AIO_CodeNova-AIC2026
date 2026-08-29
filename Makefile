.DEFAULT_GOAL := help

EXP         ?= demo
INPUT       ?= data/raw_videos
TOPK        ?= 20
QUERY       ?= a person riding a motorbike
RESUME      ?=          # set 1 để resume 1 experiment đã có (vd: make pipeline RESUME=1)
WITH_TEXT   ?=          # set 1 để chạy thêm OCR/ASR trong offline-index
# HOST/PORT để trống -> lấy mặc định từ CODENOVA_UI_HOST/PORT trong .env
HOST        ?=
PORT        ?=
TN2_DIR     ?= external/TransNetV2/inference-pytorch
TN2_WEIGHTS ?= $(TN2_DIR)/transnetv2-pytorch-weights.pth

.PHONY: help setup sync install-hooks lint format check test pre-commit precommit \
        qdrant-up qdrant-down qdrant-health \
        elasticsearch-up elasticsearch-down elasticsearch-health \
        vllm-reranker-up vllm-reranker-down vllm-reranker-health \
        infra-up infra-status \
        preflight-index validate-index repair-manifest offline-index \
        ingest detect-shots extract-frames embed-frames build-index extract-text extract-asr extract-ocr drop-ocr-watermarks export-text import-text pipeline \
        search serve-ui clean-runs clean

help: ## Hiện danh sách lệnh này
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- Môi trường ---
setup: sync ## Cài TẤT CẢ: dependency + GPU extras + external repo (TransNetV2, gipformer)
	@echo "=== Optional GPU extras (onnxruntime, TensorRT) ==="
	uv pip install onnxruntime onnx onnxscript tensorrt || \
		echo "  (bỏ qua — pipeline vẫn chạy được với PyTorch)"
	@echo "=== torchaudio (Silero-VAD cần; khớp index cu128 của torch) ==="
	uv pip install torchaudio --index-url https://download.pytorch.org/whl/cu128 || \
		echo "  (bỏ qua — ASR sẽ cắt cố định 30s thay vì theo câu nói)"
	@echo "=== External repos + weights (một lần, vài phút) ==="
	uv run python -c "import sys; sys.path.insert(0, 'src'); \
		from core.external_setup import ensure_transnetv2, ensure_gipformer; \
		print('TransNetV2:', ensure_transnetv2()); print('gipformer:', ensure_gipformer())"
	@echo "=== Xong. Chạy 'make qdrant-up elasticsearch-up' rồi 'make pipeline EXP=... INPUT=...' ==="

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
	uv run pytest -v

pre-commit: ## Chạy toàn bộ pre-commit hook
	uv run pre-commit run --all-files

precommit: pre-commit ## Alias của pre-commit

## --- Backend hạ tầng: tự nhận diện docker (máy Windows/Ubuntu có Docker
## Desktop/Engine) hay supervisorctl (instance Vast.ai này, không có Docker).
## Ép buộc thủ công nếu cần: make qdrant-up INFRA_BACKEND=docker|supervisor
INFRA_BACKEND ?= $(shell command -v docker >/dev/null 2>&1 && echo docker || echo supervisor)

## --- Qdrant (vector DB) ---
qdrant-up: ## Khởi động Qdrant (docker compose hoặc supervisor, xem INFRA_BACKEND)
ifeq ($(INFRA_BACKEND),docker)
	docker compose up -d qdrant
else
	supervisorctl start qdrant
endif

qdrant-down: ## Dừng Qdrant
ifeq ($(INFRA_BACKEND),docker)
	docker compose stop qdrant
else
	supervisorctl stop qdrant
endif

qdrant-health: ## Kiểm tra Qdrant có sống không
	curl -sf http://localhost:6333/healthz && echo

## --- Elasticsearch (full-text index cho OCR/ASR) ---
elasticsearch-up: ## Khởi động Elasticsearch (docker compose hoặc supervisor)
ifeq ($(INFRA_BACKEND),docker)
	docker compose up -d elasticsearch elasticvue
else
	supervisorctl start elasticsearch
endif

elasticsearch-down: ## Dừng Elasticsearch
ifeq ($(INFRA_BACKEND),docker)
	docker compose stop elasticsearch elasticvue
else
	supervisorctl stop elasticsearch
endif

elasticsearch-health: ## Kiểm tra Elasticsearch có sống không
	curl -sf http://localhost:8882 && echo

## --- vLLM reranker (Qwen3-VL-Reranker-2B) — chỉ cần khi RERANKER_BACKEND=qwen-vl-vllm ---
vllm-reranker-up: ## Khởi động vLLM reranker (docker compose hoặc supervisor)
ifeq ($(INFRA_BACKEND),docker)
	docker compose up -d vllm-reranker
else
	supervisorctl start vllm-reranker
endif

vllm-reranker-down: ## Dừng vLLM reranker
ifeq ($(INFRA_BACKEND),docker)
	docker compose stop vllm-reranker
else
	supervisorctl stop vllm-reranker
endif

vllm-reranker-health: ## Kiểm tra vLLM reranker có sống không
	curl -sf http://localhost:8884/health && echo

## --- Gộp: start toàn bộ infra đang cài (Qdrant + Elasticsearch) + kiểm tra ---
infra-up: qdrant-up elasticsearch-up ## Khởi động Qdrant + Elasticsearch, rồi health-check cả hai
	@echo "--- Backend: $(INFRA_BACKEND) — chờ service sẵn sàng ---"
	@sleep 3
	@$(MAKE) qdrant-health elasticsearch-health

infra-status: ## Xem trạng thái các service infra (docker compose hoặc supervisor)
	@echo "--- Backend: $(INFRA_BACKEND) ---"
ifeq ($(INFRA_BACKEND),docker)
	-docker compose ps
else
	-supervisorctl status qdrant elasticsearch vllm-reranker
endif

## --- Pipeline index offline ---
preflight-index: ## In inventory và tạo plan; thêm APPROVE=1 để phê duyệt plan (EXP, INPUT)
	uv run codenova preflight-index --input $(INPUT) --experiment-name $(EXP) \
		$(if $(APPROVE),--approve)

validate-index: ## Chạy quality gate và ghi readiness.json (EXP)
	uv run codenova validate-index --experiment-name $(EXP)

repair-manifest: ## Dry-run repair JSONL (EXP, MANIFEST); thêm APPLY=1 để sửa + backup
	uv run codenova repair-manifest $(MANIFEST) --experiment-name $(EXP) \
		$(if $(APPLY),--apply)

migrate-frame-paths: ## Dry-run chuẩn hóa frame path (EXP, LEGACY_ROOT); APPLY=1 để ghi
	uv run codenova migrate-frame-paths --experiment-name $(EXP) \
		--legacy-root $(LEGACY_ROOT) $(if $(APPLY),--apply)

offline-index: ## Một lệnh: preflight được duyệt -> vector stages -> quality gate (EXP, INPUT)
	uv run codenova offline-index --input $(INPUT) --experiment-name $(EXP) --approve \
		$(if $(RESUME),--resume) $(if $(WITH_TEXT),--with-text)

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

extract-text: ## Chạy OCR + ASR, index vào Elasticsearch (EXP); cần Elasticsearch đang chạy
	uv run codenova extract-text --experiment-name $(EXP)

extract-asr: ## Chỉ chạy ASR, index vào Elasticsearch (EXP); cần Elasticsearch đang chạy
	uv run codenova extract-text --experiment-name $(EXP) --skip-ocr

extract-ocr: ## Chỉ chạy OCR, index vào Elasticsearch (EXP); cần Elasticsearch đang chạy
	uv run codenova extract-text --experiment-name $(EXP) --skip-asr

drop-ocr-watermarks: ## Lọc dòng watermark/logo đài lặp lại khỏi OCR đã trích (EXP); chạy sau khi extract-ocr xong hẳn
	uv run codenova drop-ocr-watermarks --experiment-name $(EXP)

export-text: ## Xuất document OCR/ASR từ Elasticsearch ra runs/<exp>/manifests/text.jsonl (EXP)
	uv run codenova export-text --experiment-name $(EXP)

import-text: ## Nạp runs/<exp>/manifests/text.jsonl vào Elasticsearch (EXP); cần Elasticsearch đang chạy
	uv run codenova import-text --experiment-name $(EXP)

pipeline: offline-index ## Alias tương thích cho offline-index có preflight và quality gate

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
