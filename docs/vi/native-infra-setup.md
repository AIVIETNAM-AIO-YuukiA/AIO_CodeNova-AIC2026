# Cài infra không dùng Docker (Vast.ai instance)

`docker-compose.yml` ở gốc repo định nghĩa 4 service: Qdrant, Elasticsearch,
Elasticvue, vllm-reranker. Instance Vast.ai hiện tại chạy trong container
**không đặc quyền** (unprivileged), không hỗ trợ Docker-in-Docker, nên
`docker compose up` không chạy được ở đây. File này ghi lại cách đã cài native
binary tương đương, quản lý bằng **supervisor** để tự khởi động lại và log
gộp vào portal — không cần cấu hình `.env` khác gì so với dùng Docker vì
mọi service đều bind vào `127.0.0.1` với đúng port mà `.env.example` đã kỳ vọng.

**Trạng thái trên instance này:** Qdrant và Elasticsearch đã cài và chạy qua
supervisor (mục 5 xác nhận health check). Elasticvue và vllm-reranker chưa
cài — `RERANKER_BACKEND` trong `.env` hiện là `blip2` (chạy in-process) nên
vllm-reranker không cần thiết.

Tổng quan port (khớp `.env.example`, chạy nội bộ `127.0.0.1`, không cần mở
port ra ngoài vì app Python trong repo gọi trực tiếp qua localhost):

| Service        | Port  | Biến `.env` liên quan       | Trạng thái |
|----------------|-------|------------------------------|------------|
| Qdrant (REST)  | 6333  | `QDRANT_URL`                 | ✅ chạy |
| Qdrant (gRPC)  | 6334  | —                             | ✅ chạy |
| Elasticsearch  | 8882  | `ELASTIC_URL`                 | ✅ chạy |
| Elasticvue     | 8883  | — (chỉ là UI, không bắt buộc) | chưa cài |
| vllm-reranker  | 8884  | `QWEN_VL_RERANKER_URL`        | chưa cần (backend=blip2) |

Chỉ cài vllm-reranker nếu chuyển sang `RERANKER_BACKEND=qwen-vl-vllm`.

---

## 1. Qdrant

Static binary, không cần build.

```bash
mkdir -p /opt/qdrant "${WORKSPACE}/AIO_CodeNova-AIC2026/qdrant_storage"
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz \
  | tar -xz -C /opt/qdrant
```

Wrapper script `/opt/supervisor-scripts/qdrant.sh`:

```bash
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

export QDRANT__STORAGE__STORAGE_PATH="${WORKSPACE:-/workspace}/AIO_CodeNova-AIC2026/qdrant_storage"
export QDRANT__SERVICE__HTTP_PORT=6333
export QDRANT__SERVICE__GRPC_PORT=6334
pty /opt/qdrant/qdrant 2>&1
```

`chmod +x /opt/supervisor-scripts/qdrant.sh`.

Supervisor config `/etc/supervisor/conf.d/qdrant.conf`:

```ini
[program:qdrant]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/qdrant.sh
autostart=true
autorestart=unexpected
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
```

## 2. Elasticsearch

Cần Java — bundle theo distro tar của ES là đủ, không cần cài JDK riêng.

```bash
curl -L -o /tmp/es.tar.gz \
  https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.1.0-linux-x86_64.tar.gz
mkdir -p /opt/elasticsearch
tar -xzf /tmp/es.tar.gz -C /opt/elasticsearch --strip-components=1
# ES không cho chạy bằng root -> tạo user riêng
useradd -m -s /bin/bash esuser || true
mkdir -p "${WORKSPACE}/AIO_CodeNova-AIC2026/elasticsearch_data"
chown -R esuser:esuser /opt/elasticsearch "${WORKSPACE}/AIO_CodeNova-AIC2026/elasticsearch_data"
```

**Quan trọng — dùng `elasticsearch.yml`, không dùng cờ `-E` trên CLI:**
`-E http.cors.allow-origin="*"` bị lỗi khi wrapper đi qua `pty` → `unbuffer -p`
→ `runuser -u esuser -- ...`: giá trị `*` mất dấu ngoặc kép qua các lớp quoting
đó và YAML parser của ES hiểu `*` là cú pháp alias, ES crash-loop ngay khi
khởi động (`MarkedYAMLException: while scanning an alias`). Cách chạy được:
ghi toàn bộ cấu hình vào `config/elasticsearch.yml` (append), để script chỉ
gọi `./bin/elasticsearch` không kèm tham số nào.

Append vào `/opt/elasticsearch/config/elasticsearch.yml`:

```yaml
discovery.type: single-node
xpack.security.enabled: false
http.port: 8882
http.host: 127.0.0.1
http.cors.enabled: true
http.cors.allow-origin: "/.*/"
http.cors.allow-headers: X-Requested-With,Content-Type,Content-Length,Authorization
path.data: /workspace/AIO_CodeNova-AIC2026/elasticsearch_data
```

(`chown esuser:esuser` file này sau khi sửa.)

Wrapper `/opt/supervisor-scripts/elasticsearch.sh`:

```bash
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

export ES_JAVA_OPTS="-Xms1g -Xmx1g"
cd /opt/elasticsearch
# Cấu hình (port, cors, path.data...) nằm trong config/elasticsearch.yml,
# không truyền qua -E để tránh lỗi quoting khi đi qua pty/runuser.
pty runuser -u esuser -- ./bin/elasticsearch 2>&1
```

`chmod +x /opt/supervisor-scripts/elasticsearch.sh`.

Supervisor config `/etc/supervisor/conf.d/elasticsearch.conf`: giống mẫu
`qdrant.conf` ở trên (đổi `command` sang `elasticsearch.sh`), thêm
`startsecs=15` vì JVM khởi động chậm hơn Qdrant.

Lưu ý về `vm.max_map_count`: ES khuyến nghị `>= 262144`, container này báo
`65530` và sysctl là read-only (`sysctl -w` báo lỗi). Trên instance này việc
đó **chỉ là WARN, không chặn khởi động**, vì ES chỉ strict-enforce bootstrap
check này khi bind ra ngoài loopback ("production mode"); bind `127.0.0.1`
(single-node/dev) thì bỏ qua. Nếu sau này đổi `http.host` ra địa chỉ khác
127.0.0.1, cần báo host/Vast hỗ trợ nâng sysctl hoặc dùng máy khác.

## 3. Elasticvue (UI, tuỳ chọn — chưa cài trên instance này)

Elasticvue chỉ là SPA tĩnh — dùng bản CLI Node thay vì Docker image:

```bash
. /opt/nvm/nvm.sh
npm install -g elasticvue-cli   # hoặc: npx serve trên bản build tĩnh tải từ GitHub release
```

Wrapper `/opt/supervisor-scripts/elasticvue.sh`:

```bash
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Elasticvue"

. /opt/nvm/nvm.sh
pty elasticvue-cli --port 8883 --host 127.0.0.1 2>&1
```

Vì đây là UI phụ trợ (không bắt buộc cho app chạy), có thể bỏ qua — dùng
`curl` hoặc REST client để kiểm tra ES trực tiếp thay vì UI.

## 4. vllm-reranker (chỉ cần khi `RERANKER_BACKEND=qwen-vl-vllm` — chưa cài)

`vllm` cài thẳng vào venv có sẵn, không cần container GPU riêng vì GPU đã
pass-through vào container này.

```bash
source /venv/main/bin/activate
uv pip install vllm
```

Wrapper `/opt/supervisor-scripts/vllm-reranker.sh`:

```bash
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

source /venv/main/bin/activate
pty vllm serve Qwen/Qwen3-VL-Reranker-2B \
  --task score \
  --host 127.0.0.1 \
  --port 8884 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 8192 \
  --dtype bfloat16 2>&1
```

Supervisor config tương tự mẫu trên, `command` trỏ vào `vllm-reranker.sh`.

Trước khi bật service này, kiểm tra VRAM còn trống (embedder jina/siglip2
chạy trong process Python khác có thể đã chiếm phần lớn 24GB của RTX 3090):

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 5. Áp dụng & kiểm tra

```bash
supervisorctl reread && supervisorctl update
supervisorctl status
curl http://127.0.0.1:6333/healthz            # Qdrant OK
curl http://127.0.0.1:8882                    # Elasticsearch OK
curl http://127.0.0.1:8884/v1/models          # vllm-reranker OK (nếu bật)
```

Log từng service: `tail -f /var/log/portal/<qdrant|elasticsearch|vllm-reranker>.log`.

**Bẫy đã gặp — process cũ giữ port khi supervisor báo BACKOFF:** nếu từng
test chạy service bằng tay (`&` nền) rồi `kill` PID cha, tiến trình Java con
có thể sống sót (detach), vẫn giữ port. Lúc đó `supervisorctl start` sẽ báo
`BACKOFF`/`spawn error` dù `curl` health check vẫn trả OK (đang trả lời từ
process orphan, không phải process supervisor quản lý). Xử lý:
`pkill -9 -f elasticsearch` (hoặc tên service tương ứng), xác nhận port trống
bằng `ss -tlnp | grep <port>`, rồi `supervisorctl start <service>` lại.

## 6. Dữ liệu có persist không?

`${WORKSPACE}/AIO_CodeNova-AIC2026/qdrant_storage` và
`.../elasticsearch_data` chỉ sống sót qua **recycle/destroy** nếu
`${WORKSPACE}` là host volume thật:

```bash
vast-capabilities | jq '.instance.workspace_is_volume'
```

Nếu `false`, dữ liệu index sẽ mất khi instance bị recycle/destroy (nhưng vẫn
giữ nguyên qua stop/start bình thường) — cân nhắc backup định kỳ (rclone,
snapshot) nếu index tốn nhiều công build lại.
