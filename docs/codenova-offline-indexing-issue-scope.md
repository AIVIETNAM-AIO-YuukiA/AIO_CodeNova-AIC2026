# CodeNova — Phạm vi issue có thể xử lý từ offline indexing

Cập nhật: 2026-08-14 (Asia/Ho_Chi_Minh)

## 1. Mục đích

Tài liệu này tách các phát hiện trong
[`codenova-retrieval-issue-report.md`](./codenova-retrieval-issue-report.md)
theo phạm vi của người quản lý offline indexing. Việc một issue xuất hiện ở
đây không có nghĩa toàn bộ issue thuộc indexing; một số issue chỉ có thể xử lý
phần nguyên nhân dữ liệu hoặc cung cấp validation/readiness cho retrieval.

Baseline đang được báo cáo là branch `version3_10/8/2026`, commit
`d2a4497f62d7833abf7d2eca862644cbe6022b2a`. Khi baseline thay đổi, cần chạy
lại test và rà lại line anchor trước khi sử dụng tài liệu để phân công.

## 2. Kết luận nhanh

| Nhóm | Issue | Phạm vi phù hợp |
|---|---|---|
| Có thể sửa chính từ indexing | CN-05 | Manifest, metadata, prerequisite, quality gate |
| Có thể sửa chính từ indexing | CN-09 | Model registry, unknown model fail-fast, provenance |
| Có thể sửa phần nền tảng | CN-01 | Persist và khôi phục config định nghĩa artifact |
| Có thể sửa một phần | CN-03 | Alignment/coverage của embedding artifacts |
| Có thể sửa một phần | CN-10 | **DONE:** canonical frame path, validation, migration |
| Có thể sửa một phần | CN-06 | **DONE tổng thể:** offline readiness + online warmup health |
| Ngoài phạm vi indexing | CN-02, CN-04, CN-07, CN-08, CN-11, CN-12 | UI, request handling, query processing, reranking hoặc rendering |

Thứ tự thực hiện đề xuất:

1. CN-05 — làm cho output của indexing đầy đủ và kiểm chứng được.
2. CN-09 — ngăn tạo artifact bằng model được suy đoán sai.
3. CN-01 — khóa provenance bằng persisted experiment config.
4. CN-03 — kiểm tra alignment và coverage giữa các model.
5. CN-10 — kiểm tra metadata và khả năng resolve frame path.
6. CN-06 — xuất offline readiness report cho bước activation.

## 3. Issue có thể sửa chính từ offline indexing

### 3.1. CN-05 — Manifest và metadata không bảo đảm đầy đủ

**Trạng thái triển khai: DONE (2026-08-14).** Trạng thái này áp dụng cho
failure mode CN-05 trên branch hiện tại; CN-01/CN-03/CN-09 vẫn được theo dõi
riêng dù một phần nền tảng liên quan đã được gia cố cùng thay đổi này.

Bằng chứng hoàn thành:

- `offline-index` tạo/verify preflight plan, chạy các stage và luôn chạy quality
  gate cuối tại [`src/cli/main.py`](../src/cli/main.py).
- JSONL được inspect nghiêm ngặt; partition được consolidate bằng temporary
  file + atomic replace; repair là command riêng, mặc định dry-run, có backup
  và audit JSON tại [`src/indexing/manifest.py`](../src/indexing/manifest.py)
  và [`src/cli/main.py`](../src/cli/main.py).
- Validator kiểm tra ID/reference, video source checksum/size, frame file,
  embedding ID/vector/coverage, OCR/ASR/caption và failed jobs tại
  [`src/indexing/validation.py`](../src/indexing/validation.py).
- UI chỉ đọc `readiness.json`, kiểm tra config hash và artifact fingerprint;
  không tạo hoặc repair manifest tại [`src/ui/server.py`](../src/ui/server.py).
- Regression/unit/integration/CLI tests nằm trong `tests/unit/` và
  `tests/integration/`.

#### Vì sao thuộc phạm vi indexing

Offline pipeline là nơi tạo metadata gốc:

- Discovery tính checksum, `video_id` và kích thước file tại
  [`src/video/discovery.py:13-37`](../src/video/discovery.py#L13-L37).
- Ingest ghi video manifest tại
  [`src/indexing/ingest.py:16-45`](../src/indexing/ingest.py#L16-L45).
- Shot stage ghi shot records tại
  [`src/indexing/shots.py:37-76`](../src/indexing/shots.py#L37-L76).
- Frame stage ghi frame records tại
  [`src/indexing/frames.py:27-53`](../src/indexing/frames.py#L27-L53).
- Embedding stage tiêu thụ frame metadata tại
  [`src/indexing/embeddings.py:175-215`](../src/indexing/embeddings.py#L175-L215).

Nếu các artifact này thiếu hoặc không nhất quán, retrieval không thể khôi phục
metadata thật một cách đáng tin cậy. Serve path không nên phải suy đoán lại
`video_id`, FPS, checksum hoặc shot structure.

#### Hạng mục có thể thực hiện

1. Kiểm tra prerequisite trước từng stage; manifest thiếu/rỗng không được coi
   là success hợp lệ nếu stage cần dữ liệu đó.
2. Kiểm tra uniqueness và referential integrity của `video_id`, `shot_id` và
   `frame_id`.
3. Ghi manifest theo cơ chế an toàn hơn: temporary file, validate rồi atomic
   replace; không để JSONL dở dang được coi là output hoàn chỉnh.
4. Ghi structured failure report cho file/video/frame bị bỏ qua.
5. Chạy quality gate sau indexing để đối chiếu videos, shots, frames,
   embeddings, captions, OCR và ASR.
6. Chỉ đánh dấu experiment ready khi quality gate thỏa policy đã chốt.
7. Tách repair thành lệnh chủ động có `--dry-run`, backup và audit log; serve
   path chỉ validate, không tự sinh metadata suy đoán.

#### Kết quả cần bàn giao

- Validated manifests.
- Coverage report theo từng stage và từng model.
- Exit code phản ánh partial/failed indexing.
- Readiness status có nguyên nhân rõ ràng.
- Test cho missing prerequisite, record trùng, JSONL hỏng và partial failure.

### 3.2. CN-09 — Unknown model bị ánh xạ sang SigLIP

**Trạng thái triển khai: DONE (2026-08-14).** Factory và preflight dùng chung
strict registry; tên có marker `siglip` hợp lệ vẫn được hỗ trợ, còn unknown/typo
raise `EmbeddingError`/`PreflightError` trước khi tạo artifact. Embedding
manifest lưu requested name, backend, resolved model ID, revision,
preprocessing identity và dimension. Quality gate cùng `build_retriever()` đối
chiếu provenance offline với encoder được resolve tại runtime. Regression tests
nằm tại `tests/unit/test_embedding_registry.py`,
`tests/unit/test_embed_incremental.py` và
`tests/unit/test_index_quality_gate.py`.

#### Vì sao thuộc phạm vi indexing

Ở baseline trước sửa, `build_embedder()` kiểm tra Jina, BEiT-3 và Vietnamese marker, sau đó trả
`SiglipEmbedder` cho mọi tên còn lại tại
[`src/modules/embedding/__init__.py:17-50`](../src/modules/embedding/__init__.py#L17-L50).
Offline embedding gọi factory này theo model trong experiment tại
[`src/indexing/embeddings.py:194-199`](../src/indexing/embeddings.py#L194-L199).

Chuỗi rủi ro:

```text
model name bị typo hoặc chưa hỗ trợ
→ factory mặc định chọn SigLIP
→ indexing có thể tạo artifact bằng backend không đúng intent
→ artifact key và encoder provenance không đáng tin cậy
```

#### Hạng mục có thể thực hiện

1. Tạo registry/alias rõ ràng cho model được hỗ trợ.
2. Chỉ nhận diện SigLIP khi alias hợp lệ hoặc model ID thực sự chứa `siglip`.
3. Unknown model phải raise typed configuration error trước khi ghi artifact.
4. Ghi resolved backend, model ID, revision, dimension và preprocessing identity
   vào metadata/readiness report.
5. Test alias hợp lệ, Hugging Face ID hợp lệ, typo và unknown model.

Warning có thể được ghi để hỗ trợ chẩn đoán, nhưng không nên warning rồi tiếp
tục indexing bằng một kiến trúc được đoán.

## 4. Issue có thể sửa phần nền tảng

### 4.1. CN-01 — Persisted experiment config không được dùng khi mở lại

**Trạng thái triển khai: DONE (2026-08-14).** `config.json` schema v1 tách
`artifact_config` khỏi `runtime_defaults`; hash chỉ bao phủ artifact-defining
fields. `Experiment.open()` phục hồi persisted artifact config, validate
schema/name/hash/type và chỉ cho `runs_dir`, `device`, `top_k` thay đổi lúc
runtime. Artifact flag được truyền explicit nhưng khác persisted value sẽ raise
`ExperimentConfigError` chứa field cùng hai giá trị. `create(..., resume=True)`
dùng cùng loader và không rewrite metadata. Legacy unversioned metadata được
đọc tương thích nhưng không tự sửa; metadata thiếu/hỏng/future schema fail rõ.
CLI log active hash/config/runtime controls và UI hiển thị active experiment +
persisted model set. Regression tests nằm tại
`tests/unit/test_runtime_config.py` và `tests/integration/test_index_cli.py`.

#### Phần indexing có thể xử lý

Ở baseline trước sửa, `Experiment.create()` ghi config vào metadata, nhưng issue report đã chứng minh
luồng mở experiment có thể gắn runtime config thay vì khôi phục config đã dùng
để tạo artifact. Phân tích và characterization test đầy đủ nằm trong
[`codenova-retrieval-issue-report.md`](./codenova-retrieval-issue-report.md#cn-01--existing-experiment-bỏ-qua-persisted-config).

Các hạng mục nền tảng có thể thực hiện:

1. Persist đầy đủ field định nghĩa indexing artifact.
2. Khi mở existing experiment, đọc và validate persisted config.
3. Tách `artifact-defining config` khỏi runtime-only option như device hoặc
   request `top_k`.
4. Từ chối override làm thay đổi model/preprocessing identity của artifact.
5. Ghi config hash cùng resolved model provenance.
6. Thêm regression test: config runtime khác config offline không được âm thầm
   thay thế artifact config.

Phần lựa chọn experiment trên UI và cách hiển thị model vẫn nằm ngoài thay đổi
offline thuần túy, nhưng retrieval có thể dựa vào persisted config chỉ khi lớp
experiment cung cấp source of truth đúng.

## 5. Issue chỉ có thể xử lý một phần từ indexing

### 5.1. CN-03 — Alignment giữa embedding artifacts

**Trạng thái triển khai: DONE (2026-08-14).** Offline `build_index()` kiểm tra
vector/ID count, uniqueness, finite vectors và yêu cầu mọi model có cùng tập
`frame_id`; row order khác nhau được join về canonical order theo ID. Quality
gate xuất `embedding_alignment` gồm policy, canonical/common counts, set/order
status. Online `wsf_fuse()` độc lập validate từng model, reject missing/extra/
duplicate/count/dimension mismatch bằng `FusionError`, rồi align score theo
`frame_id`. `load_temporal_data()` kiểm tra artifact và dùng `frame_id` làm
tie-breaker deterministic. Regression tests nằm tại
`tests/unit/test_build_index.py`, `tests/unit/test_fusion.py`,
`tests/unit/test_temporal_search.py` và `tests/unit/test_index_quality_gate.py`.

Ở baseline trước sửa, fusion xảy ra trong retrieval nên offline indexing không thể đóng toàn bộ
CN-03. Tuy nhiên indexing có thể bảo đảm đầu vào có thể fusion an toàn:

1. Mỗi vector phải đi kèm `frame_id` có provenance rõ.
2. Số vector phải bằng số ID của artifact tương ứng.
3. Không có duplicate `frame_id`, vector `NaN` hoặc `Inf`.
4. Báo coverage, phần giao và phần thiếu giữa các model.
5. Nếu pipeline yêu cầu row alignment, tạo và validate canonical frame order.
6. Không đánh dấu ready khi alignment không thỏa policy.

Retrieval vẫn cần validate hoặc join/fusion theo `frame_id`; quality gate phía
offline không thay thế kiểm tra tại consumer boundary.

### 5.2. CN-10 — Frame path hoặc frame metadata không hợp lệ

**Trạng thái triển khai: DONE (2026-08-14)** cho phạm vi đã chốt. Offline frame
extraction lưu path tương đối với run; validator báo reason code và coverage;
activation recheck file trước warmup. Embedding/OCR/temporal/hydration/UI dùng
resolver theo experiment. Legacy run có command `migrate-frame-paths` mặc định
dry-run, cập nhật cả manifest tổng và partition khi `--apply`, đồng thời backup,
audit và invalid readiness.

Source chính: [`src/core/paths.py`](../src/core/paths.py),
[`src/indexing/frames.py`](../src/indexing/frames.py),
[`src/indexing/validation.py`](../src/indexing/validation.py),
[`src/indexing/frame_paths.py`](../src/indexing/frame_paths.py),
[`src/retrieval/hydrator.py`](../src/retrieval/hydrator.py) và
[`src/retrieval/search.py`](../src/retrieval/search.py).

Không thay đổi WSF, rerank, `top_k`, fusion pool hay backfill. Backfill vẫn chỉ
là đề xuất riêng cho team retrieval nếu muốn bảo đảm số lượng output sau một
sự cố file xảy ra giữa activation và request.

### 5.3. CN-06 — Readiness của experiment và model

**Trạng thái tổng thể: DONE (2026-08-14).** Offline indexing cung cấp readiness,
provenance, vector coverage và artifact fingerprint. Online activation tại
[`src/ui/server.py`](../src/ui/server.py) kiểm tra từng embedder độc lập, chặn
activation nếu model bắt buộc lỗi và degrade có kiểm soát nếu reranker tùy chọn
lỗi. Runtime reranker fallback nằm tại
[`src/retrieval/search.py`](../src/retrieval/search.py).

Phần indexing có thể cung cấp là readiness report, tối thiểu gồm:

```json
{
  "experiment": "competition-v3",
  "config_hash": "...",
  "embedding_models": {
    "model-name": {
      "status": "ready",
      "vector_count": 0,
      "dimension": 0,
      "frame_coverage": 0.0,
      "model_id": "...",
      "revision": "..."
    }
  },
  "ocr": {"status": "ready", "document_count": 0},
  "asr": {"status": "ready", "document_count": 0}
}
```

Report offline vẫn chỉ chứng minh artifact sẵn sàng; `WarmupReport` online mới
chứng minh component runtime đã load. Hai lớp kiểm tra không thay thế nhau.

## 6. Issue ngoài phạm vi offline indexing

| Issue | Lý do |
|---|---|
| CN-02 | Empty model selection và API behavior thuộc UI/retrieval request |
| CN-04 | Model checkbox hard-code và dynamic capabilities thuộc UI |
| CN-07 | Kiểu dữ liệu boolean trong JSON request thuộc API validation |
| CN-08 | Retry/circuit state của LLM query processor thuộc online retrieval |
| CN-11 | Candidate reranking là đề xuất rà soát benchmark online, chưa xác định là bug |
| CN-12 | Caption đã có trong payload nhưng chưa được UI render |

Các issue này vẫn nên giữ trong báo cáo tích lũy để team xem xét, nhưng không
nên đưa vào acceptance criteria của gói sửa offline indexing.

## 7. Definition of done cho gói offline indexing

Gói thay đổi offline có thể coi là hoàn thành khi:

1. Existing experiment khôi phục đúng artifact-defining config đã persist.
2. Unknown embedding model thất bại trước khi tạo artifact.
3. Manifests có validation về schema, uniqueness và referential integrity.
4. Mọi embedding artifact có model provenance, dimension, vector count và
   frame-ID coverage.
5. Alignment/cross-model coverage được báo rõ.
6. Frame paths được kiểm tra bằng một quy tắc resolver thống nhất.
7. OCR/ASR/caption coverage xuất hiện trong readiness report nếu các stage được
   cấu hình.
8. Partial failure không bị báo thành success hoàn toàn.
9. Serve path có thể đọc readiness report nhưng không cần tự sửa hoặc suy đoán
   indexing metadata.
10. Unit/integration tests chứng minh các invariant trên và lưu được kết quả để
    tái kiểm tra sau khi branch thay đổi.

### Trạng thái CN-05 theo checklist

Các tiêu chí trực tiếp của CN-05 đã hoàn thành: manifest an toàn, partial
failure có trạng thái, quality gate/readiness, source metadata validation,
explicit repair và serve read-only. Model provenance/unknown backend thuộc
CN-09 và fusion policy thuộc CN-03 nên tiếp tục có issue/status riêng.
