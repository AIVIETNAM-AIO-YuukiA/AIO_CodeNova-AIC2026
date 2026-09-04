# Hướng dẫn cài mới và khởi động lại CodeNova trên GPU server

Tài liệu này dùng cho hai tình huống:

1. **Server mới**: chưa có source, dependency, Docker hoặc artifact.
2. **Server đã từng chạy**: chỉ cần khởi động lại Qdrant, Elasticsearch và UI.

Các lệnh bên dưới giả định Ubuntu/Linux và bạn đang là `root`. Nếu dùng user
thường, thêm `sudo` vào các lệnh hệ thống.

> Không dùng `curl -I` để kiểm tra UI: server UI chỉ hỗ trợ `GET`, nên `HEAD`
> có thể trả `501` dù UI hoàn toàn bình thường.

## 0. Những thứ cần có

- GPU NVIDIA và driver hoạt động: `nvidia-smi` phải hiển thị GPU.
- Ít nhất 100 GB trống nếu tải đầy đủ `runs/result`, video gốc và model cache.
- Một port public, ví dụ `39000`, nếu muốn mở UI từ máy cá nhân.
- GitHub access để clone/pull source.
- OpenRouter API key và một **model hỗ trợ ảnh** nếu dùng caption/OCR/Grounded
  VQA. VQA nhiều frame không hoạt động với model chỉ nhận text.

## 1. Server mới: kiểm tra GPU và cài công cụ cơ bản

```bash
nvidia-smi

apt-get update
apt-get install -y aria2 ca-certificates curl git wget unzip make
```

Nếu `nvidia-smi` báo `Driver/library version mismatch`, không cài lại driver
ngay. Thường server vừa cập nhật driver/kernel; reboot instance rồi kiểm tra
lại:

```bash
reboot
```

Sau khi SSH vào lại, chạy `nvidia-smi` một lần nữa. Chỉ tiếp tục khi lệnh này
thành công.

## 2. Docker và GPU trong container

Kiểm tra trước:

```bash
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Lệnh cuối phải nhìn thấy GPU. Nếu Docker báo không tìm được NVIDIA runtime,
cài `nvidia-container-toolkit` từ repository NVIDIA đã được template server
cung cấp, sau đó cấu hình Docker:

```bash
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
```

Khởi động lại Docker theo loại server:

```bash
# Ubuntu/server có systemd
systemctl restart docker

# Một số container template không có systemd
service docker restart
```

Sau đó chạy lại lệnh `docker run ... nvidia-smi` ở trên. Không cần GPU trong
container để chạy Qdrant/Elasticsearch; nó chỉ cần khi chọn reranker vLLM.

## 3. Cài `uv`, clone source và cài Python dependency

Cài `uv` một lần trên mỗi server. Installer sẽ tự thêm `uv` vào shell profile;
nạp lại PATH cho terminal hiện tại rồi kiểm tra version:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Nếu `source` báo không tìm thấy file, đóng SSH và kết nối lại, sau đó chạy
`uv --version` lần nữa. Không dùng `pip install uv` cho bước này.

Sau đó clone source và tạo môi trường Python của project:

```bash
cd ~
git clone https://github.com/AIVIETNAM-AIO-YuukiA/AIO_CodeNova-AIC2026.git
cd AIO_CodeNova-AIC2026

# Nếu repo đã có sẵn thay vì clone
# git pull

uv sync
```

Lần đầu cần chạy full offline indexing mới dùng:

```bash
make setup
```

`make setup` tải external repo/weight cho TransNetV2 và gipformer. Nếu chỉ
khôi phục artifact đã có sẵn frame + embedding, `uv sync` là đủ; không cần chạy
`make setup` trước khi phục vụ UI.

## 4. Tạo `.env`

```bash
cp .env.example .env
nano .env
```

Các giá trị tối thiểu để chạy UI với Qdrant/Elasticsearch local:

```dotenv
QDRANT_URL=http://localhost:6333
ELASTIC_URL=http://localhost:8882

CODENOVA_UI_HOST=0.0.0.0
CODENOVA_UI_PORT=39000

OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=your_vision_capable_model
# Để trống sẽ dùng OPENROUTER_MODEL; vẫn phải là model nhận ảnh.
VQA_OPENROUTER_MODEL=
VQA_RETRIEVAL_POOL=500
VQA_VIDEO_HITS_PER_EVENT=8
VQA_MOMENT_POOL=20
VQA_CANDIDATE_COUNT=5
VQA_BEAM_WIDTH=20
VQA_DEBUG_TRACE=0
```

Với Grounded VQA, `VQA_OPENROUTER_MODEL` hoặc `OPENROUTER_MODEL` phải nhận
được ảnh. Query planning có thể là text-only, nhưng bước xác minh đáp án gửi
4–6 ảnh cho mỗi candidate.

Kiểm tra key đã được nạp mà không in secret:

```bash
set -a
source .env
set +a

test -n "$OPENROUTER_API_KEY" && echo "OPENROUTER_API_KEY: OK"
echo "VQA model: ${VQA_OPENROUTER_MODEL:-$OPENROUTER_MODEL}"
```

## 5. Tải và giải nén artifact `runs/result` bằng aria2c

### 5.1 Chuẩn bị cookie Google Drive

Trên máy cá nhân, đăng nhập đúng Google account có quyền xem file Drive. Export
cookie của `drive.google.com` theo định dạng **Netscape cookies.txt** (các browser
extension như “Get cookies.txt LOCALLY” có thể export đúng định dạng). Upload file
này lên server, ví dụ vào `/root/drive-cookies.txt`.

Cookie là secret tương đương phiên đăng nhập. Không commit, không gửi vào chat,
không để trong repository và giới hạn quyền đọc:

```bash
chmod 600 /root/drive-cookies.txt
```

Đặt `FILE_ID` là phần nằm giữa `/d/` và `/view` trong link Drive, ví dụ
`https://drive.google.com/file/d/FILE_ID/view`.

> Cookie giúp aria2c dùng đúng session/quyền Drive và resume download. Nó **không
> vượt được quota tải của Google Drive**. Nếu Drive báo “Too many users have
> viewed or downloaded this file recently”, hãy tạo bản copy trong Drive của bạn,
> dùng mirror/link khác, hoặc chờ quota mở lại.

### 5.2 Archive `.zip` từ Google Drive

```bash
cd ~/AIO_CodeNova-AIC2026
mkdir -p runs

aria2c \
  --continue=true \
  --file-allocation=none \
  --max-connection-per-server=1 \
  --split=1 \
  --min-split-size=1M \
  --load-cookies=/root/drive-cookies.txt \
  --user-agent='Mozilla/5.0' \
  --out=result.zip \
  "https://drive.usercontent.google.com/download?id=FILE_ID&export=download&confirm=t"

unzip result.zip -d runs

test -f runs/result/config.json && echo "result artifact: OK"
```

Archive phải chứa thư mục `result/` ở cấp đầu; vì vậy giải nén vào `runs/` sẽ
tạo đúng `runs/result/`. Không giải nén vào `runs/result/`, nếu không sẽ thành
`runs/result/result/`.

### 5.3 Archive `.tar.gz`

```bash
cd ~/AIO_CodeNova-AIC2026
mkdir -p runs
tar -xzf codenova_runs.tar.gz -C runs

test -f runs/result/config.json && echo "result artifact: OK"
```

Cảnh báo `Ignoring unknown extended header keyword 'SCHILY.fflags'` khi giải
nén tar từ macOS không phải lỗi; có thể bỏ qua.

### 5.4 Resume download và khi đầy ổ

Kiểm tra dung lượng trước/sau khi tải:

```bash
df -h .
du -sh runs .venv qdrant_storage 2>/dev/null
```

Chạy lại **đúng lệnh aria2c** ở mục 5.2 để resume file chưa hoàn tất. Không đổi
tên `--out=result.zip` giữa các lần chạy.

Nếu Drive báo quá nhiều lượt download thì phải chờ quota mở lại, tạo bản copy
vào Drive của bạn, hoặc dùng link/download host khác; cookie/aria2c không thể
vượt quota của Google.

## 6. Video gốc và frames: cần gì?

- `runs/result/frames/` là **bắt buộc** để UI hiển thị ảnh và Grounded VQA gửi
  evidence frame cho OpenRouter.
- `data/raw_videos/` là bắt buộc nếu muốn `validate-index` xác nhận artifact
  hoàn chỉnh, chạy ASR/OCR mới, hoặc mở video gốc.
- Artifact cũ có thể có frame/embedding nhưng không có raw video. UI có thể
  tìm được vector nếu index đã có, nhưng validation sẽ báo `MISSING_VIDEO_FILE`.
  Cách đúng là tải video về `data/raw_videos/` đúng tên trong manifest, không
  tạo file fake.

Cấu trúc mong đợi:

```text
data/
  raw_videos/
    L21_V001.mp4
    ...
runs/
  result/
    config.json
    frames/
    embeddings/
    manifests/
```

### 6.1 Sửa path video từ manifest Windows sau khi chuyển server

Artifact tạo trên Windows thường ghi path như
`data\\raw_videos\\L21_V001.mp4`. Trên Linux, dấu `\\` không phải path
separator nên `validate-index` có thể báo `MISSING_VIDEO_FILE` dù video đã có
trong `data/raw_videos/`.

Chỉ chạy script này **sau khi đã tải video gốc đúng tên** vào
`data/raw_videos/`. Script chỉ đổi những record mà nó tìm thấy file tương ứng,
tự tạo backup trước khi ghi và không sửa frame/embedding:

```bash
cd ~/AIO_CodeNova-AIC2026

python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json

root = Path.cwd()
manifest = root / "runs" / "result" / "manifests" / "videos.jsonl"
raw_root = root / "data" / "raw_videos"

if not manifest.is_file():
    raise SystemExit(f"Không thấy manifest: {manifest}")
if not raw_root.is_dir():
    raise SystemExit(f"Không thấy thư mục video: {raw_root}")

rows = []
changed = []
missing = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    old = str(row.get("path", ""))
    filename = Path(old.replace("\\", "/")).name
    candidate = raw_root / filename
    if candidate.is_file():
        new = candidate.relative_to(root).as_posix()
        if old != new:
            row["path"] = new
            changed.append((old, new))
    else:
        missing.append(filename or old)
    rows.append(row)

if changed:
    backup = manifest.with_suffix(
        manifest.suffix + f".before-path-fix-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    backup.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(manifest)
    print(f"Đã sửa {len(changed)} path. Backup: {backup}")
else:
    print("Không có path nào cần sửa.")

if missing:
    print(f"Chưa tìm thấy {len(missing)} video, ví dụ: {missing[:10]}")
PY

uv run codenova validate-index --experiment-name result
```

Không chạy script nếu bạn chưa có raw video: nó không thể tạo video bị thiếu.
Sau khi path được sửa, `validate-index` sẽ cập nhật lại `readiness.json`.

## 7. Khởi động Qdrant và Elasticsearch

```bash
cd ~/AIO_CodeNova-AIC2026
make infra-up
```

Kiểm tra riêng nếu cần:

```bash
make qdrant-health
make elasticsearch-health
```

Khi Docker không tồn tại nhưng server có Supervisor, Makefile tự dùng
`supervisorctl`. Có thể xem backend/trạng thái:

```bash
make infra-status
```

## 8. Khôi phục index từ artifact

### 8.1 Qdrant mới hoặc trống

`runs/result/embeddings/*.npz` không tự động xuất hiện trong Qdrant. Nếu vừa
đổi server/Qdrant chưa có collection, build lại Qdrant từ artifact:

```bash
uv run codenova build-index --experiment-name result --force
```

Lệnh này sử dụng embedding sẵn có; không chạy lại caption, OCR, ASR hay embed
toàn bộ frame.

### 8.2 Elasticsearch mới hoặc trống

Nếu `runs/result/manifests/text.jsonl` đã có OCR/ASR, import nó vào ES:

```bash
uv run codenova import-text --experiment-name result --include-captions
```

`--include-captions` nạp thêm `captions.jsonl` nếu file tồn tại. Nếu ES báo lỗi
kết nối, chạy lại `make elasticsearch-up` rồi kiểm tra `make elasticsearch-health`.

### 8.3 Validation

Chỉ chạy khi raw video và frame đầy đủ:

```bash
mkdir -p runs/result/logs
VALIDATE_LOG="runs/result/logs/validate-index-$(date -u +%Y%m%dT%H%M%SZ).log"

# Vừa xem output trên terminal, vừa lưu cả INFO/WARNING/ERROR vào log.
uv run codenova validate-index --experiment-name result 2>&1 | tee "$VALIDATE_LOG"

echo "Log validation: $VALIDATE_LOG"
echo "===== readiness.json ====="
python3 -m json.tool < runs/result/readiness.json
```

`serve-ui` cần `runs/result/readiness.json` có trạng thái hợp lệ. Nếu validation
báo thiếu video/frame, khôi phục file dữ liệu bị thiếu trước; không sửa thủ công
`readiness.json` để bỏ qua lỗi.

Để chỉ xem lỗi/warning trong log gần nhất:

```bash
grep -E 'VALIDATION_(ISSUE|COMPLETED)| ERROR | WARNING ' "$VALIDATE_LOG"
```

## 9. Reranker: tùy chọn

Mặc định reranker BLIP-2 chạy trong process và tốn VRAM. Nếu server chỉ có một
GPU 24 GB nhưng đã phải chạy embedder/VQA, nên tắt để UI ổn định:

```bash
DISABLE_RERANKER=1 nohup uv run codenova serve-ui \
  --experiment-name result --host 0.0.0.0 --port 39000 \
  > ui.log 2>&1 &
```

Nếu muốn dùng reranker vLLM riêng, đặt `RERANKER_BACKEND=qwen-vl-vllm` trong
`.env`, sau đó:

```bash
make vllm-reranker-up
make vllm-reranker-health
```

Không bật checkbox reranker trong UI nếu bạn khởi động UI với
`DISABLE_RERANKER=1`; backend sẽ vẫn tắt nó.

## 10. Khởi động UI

Trước khi chạy, nạp `.env` để OpenRouter/VQA nhận key:

```bash
cd ~/AIO_CodeNova-AIC2026
set -a
source .env
set +a

nohup uv run codenova serve-ui \
  --experiment-name result \
  --host 0.0.0.0 \
  --port 39000 \
  > ui.log 2>&1 &

tail -f ui.log
```

Khi log có dòng `Serving retrieval UI at http://0.0.0.0:39000`, kiểm tra local
trên server bằng GET:

```bash
curl -s http://127.0.0.1:39000 | head
```

Từ máy cá nhân, mở:

```text
http://PUBLIC_IP_CUA_SERVER:39000
```

Port `39000` phải được nhà cung cấp GPU mở ra. Với Vast.ai, mở port trong phần
network/ports; với server cloud khác, kiểm tra firewall/security group.

## 11. Khởi động lại sau khi reboot server

Nếu source, `.env`, `runs/result` và dữ liệu Docker còn nguyên thì chỉ cần:

```bash
cd ~/AIO_CodeNova-AIC2026
make infra-up

set -a
source .env
set +a

nohup uv run codenova serve-ui \
  --experiment-name result --host 0.0.0.0 --port 39000 \
  > ui.log 2>&1 &

tail -f ui.log
```

Không cần `import-text`, `build-index`, `validate-index`, OCR, ASR, captioning
hoặc embedding lại chỉ vì reboot. Chỉ chạy `build-index`/`import-text` khi
Qdrant/Elasticsearch mới, trống, hoặc đã mất Docker volume/storage.

## 12. Test Grounded VQA sau khi deploy code mới

Grounded VQA sử dụng `pipeline_mode="grounded"` mặc định. Test qua API:

```bash
curl -s http://127.0.0.1:39000/api/vqa-search \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"Một cô gái đeo tạp dề màu trắng đặt bốn con X lên đĩa, sau đó cầm hai con X và trao đổi về món ăn.",
    "question":"Hỏi X là con gì?",
    "top_k":20,
    "enabled_models":["jina-clip-v2","siglip2-so400m","vietnamese-embedding"],
    "use_reranker":true,
    "use_llm":true,
    "pipeline_mode":"grounded"
  }' | python3 -m json.tool
```

Để đo định lượng, tạo qrels riêng từ mẫu rồi chạy evaluator:

```bash
cp -n eval/vqa_qrels.example.jsonl eval/vqa_qrels.jsonl
uv run python -m evaluation.vqa \
  --experiment-name result \
  --qrels eval/vqa_qrels.jsonl \
  --top-k 20,50 \
  --pipeline-mode grounded
```

## 13. Lỗi thường gặp

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `Failed to initialize NVML` | Driver/kernel vừa cập nhật nhưng chưa reboot | Reboot server, rồi chạy `nvidia-smi`. |
| Docker không thấy GPU | Thiếu/cấu hình sai NVIDIA Container Toolkit | Làm lại bước 2 rồi kiểm tra bằng CUDA container. |
| UI `Connection refused` | Process UI đã exit | `tail -n 100 ui.log` để xem lỗi. |
| UI báo thiếu `readiness.json` | Chưa validate hoặc artifact thiếu data | Khôi phục raw video/frame, chạy validate-index. |
| `MISSING_VIDEO_FILE` | Chưa tải `data/raw_videos` | Tải đúng video/name theo manifest. |
| `FRAME_FILE_MISSING` | Archive chưa đủ frames hoặc giải nén sai cấp thư mục | Kiểm tra `runs/result/frames/...`; giải nén lại đúng vào `runs/`. |
| Elasticsearch `413` khi import | Bulk request quá lớn/ES cũ | Pull code mới có batch import, restart ES, chạy lại import-text. |
| `index_not_found_exception` cho text | ES mới nhưng chưa import text | Chạy `uv run codenova import-text --experiment-name result --include-captions`. |
| OpenRouter `401 Missing Authentication` | `.env` chưa được nạp hoặc key rỗng | `set -a; source .env; set +a`, rồi kiểm tra biến không in key. |
| VQA không trả lời/abstain | Model VQA không nhận ảnh hoặc evidence chưa có | Dùng vision model cho `VQA_OPENROUTER_MODEL`; kiểm tra frames và OpenRouter log. |
| Google Drive `quota exceeded` | File bị tải quá nhiều | Cookie không vượt quota; chờ, tạo bản copy Drive, hoặc dùng mirror/link khác. |
