# Qdrant — Hướng dẫn sử dụng

> Bản tiếng Anh: [../qdrant.md](../qdrant.md)

Vector index của project dùng **Qdrant**, một vector database chạy như một service
riêng. Trước khi chạy `build-index` hoặc `search`, Qdrant phải đang chạy và biến môi
trường trong `.env` phải trỏ đúng vào nó.

## 1. Khởi động Qdrant

Cách đơn giản nhất là Docker Compose (đã có sẵn `docker-compose.yml`):

```bash
docker compose up -d qdrant      # chạy nền
docker compose ps                # kiểm tra container
docker compose logs -f qdrant    # xem log
docker compose down              # dừng (dữ liệu vẫn giữ trong ./qdrant_storage)
```

Dữ liệu được lưu vào `./qdrant_storage/` (đã gitignore) nên restart không mất index.

Kiểm tra Qdrant sống:

```bash
curl http://localhost:6333/healthz        # -> "healthz check passed"
```

Dashboard web: mở http://localhost:6333/dashboard để xem collection và điểm dữ liệu.

## 2. Cấu hình `.env`

File `.env` ở thư mục gốc (được `python-dotenv` tự load khi chạy `codenova`):

```bash
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=codenova_frames
QDRANT_API_KEY=                  # để trống khi chạy local; điền khi dùng Qdrant Cloud
```

Mỗi experiment dùng một collection riêng, đặt tên `{QDRANT_COLLECTION}__{experiment_name}`,
nên nhiều run dùng chung một Qdrant không đè lên nhau.

## 3. Luồng chạy

```bash
# Offline indexing (qua Makefile)
make pipeline EXP=demo INPUT=data/raw_videos   # ingest → ... → build-index

# Online retrieval
make search   EXP=demo QUERY="a person riding a motorbike"
make serve-ui EXP=demo                          # UI tại http://127.0.0.1:7860
```

Chi tiết từng bước và chuẩn bị TransNetV2: xem [README.md](README.md) và
[transnetv2.md](transnetv2.md).

`build-index` luôn **recreate** collection rồi upsert lại toàn bộ embeddings, nên chạy
lại là idempotent (không cần xoá tay).

## 4. Một số thao tác kiểm tra nhanh

```bash
# Liệt kê các collection
curl http://localhost:6333/collections

# Xem thông tin một collection (đổi tên cho đúng experiment)
curl http://localhost:6333/collections/codenova_frames__demo

# Xoá một collection nếu muốn build lại từ đầu
curl -X DELETE http://localhost:6333/collections/codenova_frames__demo
```

## 5. Dùng Qdrant Cloud thay vì local

Tạo cluster trên Qdrant Cloud, lấy URL + API key, rồi sửa `.env`:

```bash
QDRANT_URL=https://<your-cluster>.qdrant.io:6333
QDRANT_API_KEY=<your-api-key>
```

Không cần đổi code — `index/factory.py` đọc thẳng các biến này.

## 6. Lỗi thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `Connection refused` / `Failed to connect` | Qdrant chưa chạy → `docker compose up -d qdrant` |
| `Install qdrant-client before using the Qdrant index` | Thiếu dependency → `uv sync` |
| Search trả về rỗng | Chưa `build-index`, hoặc sai `--experiment-name` (khác collection) |
| `Cannot build a Qdrant index with zero embeddings` | Chưa chạy `embed-frames` hoặc không có frame nào |
