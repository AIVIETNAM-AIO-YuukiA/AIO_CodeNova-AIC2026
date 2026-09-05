# CodeNova Retrieval — Báo cáo lỗi và rủi ro tích lũy

Cập nhật: 2026-08-11 (Asia/Ho_Chi_Minh)

## 1. Mục đích và quy ước

Tài liệu này là báo cáo gửi team về các lỗi/rủi ro có khả năng ảnh hưởng hệ
thống retrieval phục vụ cuộc thi. Mỗi issue phải có:

1. Branch và commit đã kiểm tra.
2. Mức độ xác nhận.
3. Bằng chứng source hoặc test có thể tái kiểm tra.
4. Chuỗi nguyên nhân dẫn tới biểu hiện lỗi.
5. Phạm vi ảnh hưởng và điều kiện xảy ra.
6. Định hướng sửa, test hồi quy và tiêu chí nghiệm thu.

Các nhãn dùng trong báo cáo:

- **Đã tái hiện:** test đã chạy và quan sát được hành vi nêu trong issue.
- **Đã chứng minh bằng source:** control flow thể hiện trực tiếp trong code,
  nhưng chưa đo tần suất/hậu quả trên deployment thật.
- **Hậu quả có điều kiện:** chỉ xảy ra khi runtime/data thỏa điều kiện đã nêu.
- **Chưa xác minh:** cần test hoặc artifact bổ sung.

Không xem một suy luận là sự cố production đã xảy ra nếu chưa có log/runtime
evidence. Không xem test characterization đang pass là bằng chứng hệ thống
đúng; test loại này có thể đang chủ động xác nhận hành vi lỗi hiện tại.

## 2. Danh sách issue

| ID | Tiêu đề | Target | Mức độ | Trạng thái |
|---|---|---|---|---|
| CN-01 | Existing experiment bỏ qua persisted config, dùng config runtime | `version3_10/8/2026` | Critical | **DONE (2026-08-14)** |

---

## CN-01 — Existing experiment bỏ qua persisted config

> **Trạng thái hiện tại: DONE trên branch `version3_10/8/2026` (2026-08-14).**
> Phân tích từ mục 2.1 đến 2.14 mô tả baseline trước sửa và được giữ làm lịch
> sử nguyên nhân. Implementation hiện dùng versioned metadata, tách
> artifact/runtime config, restore + hash/type validation, explicit mismatch
> error, legacy read-only compatibility, downstream provenance/readiness gate
> và UI hiển thị active persisted identity. Test được tổng hợp tại
> [`tests/unit/test_runtime_config.py`](../tests/unit/test_runtime_config.py) và
> [`tests/integration/test_index_cli.py`](../tests/integration/test_index_cli.py).

### 2.1. Tóm tắt để báo cáo

Ở baseline trước sửa, khi tạo experiment, CodeNova lưu cấu hình offline vào
`runs/<experiment>/config.json`. Tuy nhiên, khi mở experiment cho các stage sau
hoặc serve retrieval, CodeNova không đọc lại file này. Nó dựng một
`PipelineConfig` mới từ CLI/environment hiện tại và gắn config đó vào
`Experiment`.

Vì vậy cùng một `--experiment-name` có thể được offline index bằng model set A
nhưng online retrieval lại load/query bằng model set B. Hành vi nhận config B
đã được tái hiện bằng unit test. Việc deployment thật có gặp exception, dùng
thiếu model hay trả ranking sai còn phụ thuộc A/B và artifact hiện có.

### 2.2. Phạm vi xác nhận

| Nội dung | Giá trị |
|---|---|
| Branch làm baseline báo cáo | `version3_10/8/2026` |
| Commit đã kiểm tra | `d2a4497f62d7833abf7d2eca862644cbe6022b2a` |
| Working tree khi chạy test cuối | `version3_10/8/2026` |
| Kết quả | Test characterization pass; CN-01 còn tồn tại |

Toàn bộ nhận định, line anchor, test và hướng sửa từ đây chỉ áp dụng cho
`version3_10/8/2026@d2a4497`. Không dùng branch cũ làm bằng chứng cho issue.
Khi version3 có commit mới, cần chạy lại test và rà lại line anchor trước khi
gửi báo cáo.

### 2.3. Bằng chứng source: persisted config được ghi

[`PipelineConfig`](../src/config/settings.py#L41-L53) chứa các thuộc tính định
nghĩa experiment, bao gồm `embedding_models`, `frame_sampling`,
`index_backend` và `keyframe_percentiles`.

Khi tạo experiment:

- [`Experiment.create()`](../src/config/settings.py#L96-L121) gắn config được
  truyền vào object rồi gọi `write_metadata()`.
- [`Experiment.write_metadata()`](../src/config/settings.py#L132-L145) ghi
  `config_hash` và toàn bộ `config` vào `runs/<experiment>/config.json`.

Do đó nhận định “offline config có metadata để khôi phục” được chứng minh trực
tiếp bởi source; đây không phải giả định.

### 2.4. Bằng chứng source baseline: persisted config không được đọc khi mở

[`Experiment.open()`](../src/config/settings.py#L123-L130):

```python
@classmethod
def open(cls, config: PipelineConfig, name: str) -> "Experiment":
    experiment_name = validate_experiment_name(name)
    run_dir = config.runs_dir / experiment_name
    if not run_dir.exists():
        raise ExperimentNameError(...)
    return cls(name=experiment_name, run_dir=run_dir, config=config)
```

Hàm chỉ kiểm tra `run_dir` tồn tại và trả `Experiment(..., config=config)`.
Không có lệnh đọc `run_dir / "config.json"`, không so `config_hash` và không
validate runtime config với persisted config.

### 2.5. Bằng chứng source: config runtime đến từ CLI/environment

Nguồn `embedding_models` của CLI được định nghĩa tại
[`add_config_args()`](../src/cli/main.py#L132-L153):

```python
default=os.environ.get("EMBEDDING_MODELS", "jina-clip-v2")
```

Thứ tự giá trị thực tế là:

```text
--embedding-models explicit
→ nếu thiếu, EMBEDDING_MODELS trong environment/.env
→ nếu thiếu, literal default "jina-clip-v2"
```

[`config_from_args()`](../src/cli/main.py#L167-L183) chuyển các CLI value này
thành một `PipelineConfig` mới.

[`load_experiment()`](../src/cli/main.py#L219-L225) sau đó gọi:

```python
config = config_from_args(args)
experiment = Experiment.open(config=config, name=args.experiment_name)
```

Cuối cùng, [`handle_serve_ui()`](../src/cli/main.py#L310-L336) dùng chính
`load_experiment(args)` trước khi khởi động UI. Vì vậy chọn đúng experiment name
hiện chưa đồng nghĩa dùng đúng persisted config của experiment.

### 2.6. Bằng chứng downstream trên retrieval version3

Trên version3, [`build_retriever()`](../src/retrieval/search.py#L163-L189) tạo
một embedder cho từng tên trong:

```python
experiment.config.embedding_models
```

Sau đó [`Retriever.search()`](../src/retrieval/search.py#L93-L109) dùng mỗi
active model để:

```text
embed query
→ load_temporal_data(experiment, model_name)
→ đọc vector/frame-ID artifact theo model_name
→ WSF
```

Có thể khóa việc kiểm tra vào đúng commit báo cáo:

```bash
git show d2a4497:src/retrieval/search.py | nl -ba | sed -n '60,125p;163,190p'
git show d2a4497:src/retrieval/temporal_search.py | nl -ba | sed -n '247,280p'
```

[`load_temporal_data()`](../src/retrieval/temporal_search.py#L247-L274) xây
đường dẫn artifact từ `model_name`:

```python
embeddings_path = vectors_path(embeddings_dir, model_name)
frame_ids_json_path = frame_ids_path(embeddings_dir, model_name)
```

và raise `FileNotFoundError` nếu hai file không tồn tại. Do đó nếu runtime B có
model chưa được offline index trong A, “không tìm thấy artifact model B” là hậu
quả trực tiếp có điều kiện. Nếu B chỉ là subset của A, retrieval có thể chạy
nhưng âm thầm bỏ source đã index; đây cũng là hậu quả có điều kiện.

### 2.7. Chuỗi nguyên nhân

```text
Offline stage dựng config A
→ Experiment.create(A) ghi A vào config.json
→ vector/index artifact được tạo theo A

Serve hoặc existing-run stage dựng config B từ CLI/.env/default hiện tại
→ load_experiment() gọi Experiment.open(B, same_name)
→ Experiment.open() không đọc config.json, gắn B
→ Retriever tạo encoder và chọn artifact/named vector theo B

Nếu A != B:
├── B có model không được index trong A
│   → thiếu embedding file/named vector, request/startup có thể lỗi
├── B là subset của A
│   → model đã index nhưng không được load/search
└── B dùng cùng tên nhưng khác resolved checkpoint/revision
    → có nguy cơ offline/online embedding space không tương thích
```

Nhánh cuối về checkpoint là **rủi ro provenance**, chưa được unit test CN-01
chứng minh. Nó cần model revision/hash evidence riêng; không được coi là sự cố
đã xảy ra chỉ từ test config này.

### 2.8. Test tái hiện đã chạy

Test: [`tests/unit/test_runtime_config.py`](../tests/unit/test_runtime_config.py#L1-L44).

Fixture tạo experiment bằng:

```python
embedding_models=("jina-clip-v2", "siglip2-so400m")
```

rồi mở cùng experiment bằng:

```python
embedding_models=("beit3",)
```

Hai assertion cuối xác nhận hành vi lỗi hiện tại:

```python
assert reopened.config.embedding_models == ("beit3",)
assert reopened.config.embedding_models != persisted_models
```

Lệnh đã chạy trên working tree `version3_10/8/2026@d2a4497`:

```bash
.venv/bin/pytest tests/unit/test_runtime_config.py -vv
```

Kết quả quan sát:

```text
tests/unit/test_runtime_config.py::test_open_accepts_runtime_config_different_from_persisted_config PASSED
1 passed in 0.02s
```

Giải thích: test pass không có nghĩa implementation đúng. Đây là
characterization test được viết để chứng minh `Experiment.open()` chấp nhận B
và bỏ qua persisted A.

### 2.9. Mức độ và ảnh hưởng

**Severity đề xuất: Critical** đối với competition retrieval vì lỗi phá vỡ
invariant cơ bản:

```text
online query encoder/config phải tương thích với offline indexed artifacts
```

Ảnh hưởng đã chứng minh:

- `Experiment.open()` có thể trả config khác persisted metadata.
- Retriever downstream lấy model list từ config khác đó.

Ảnh hưởng có điều kiện, cần integration/runtime evidence để biết trường hợp cụ
thể:

- Startup/request lỗi vì thiếu embedding artifact hoặc Qdrant named vector.
- Search thiếu model đã offline index.
- Ranking thay đổi giữa các lần serve cùng experiment name.
- Tải model/GPU khác với cấu hình mà operator nghĩ đang phục vụ.

### 2.10. Định hướng sửa được khuyến nghị

#### Quy tắc nguồn sự thật

Khi operator chọn một existing experiment để phục vụ retrieval:

```text
experiment name
→ runs/<experiment>/config.json
→ persisted index-defining config
→ validate artifact/schema
→ build Retriever
→ UI hiển thị active experiment và model set thực tế
```

Không dùng lại toàn bộ config mới dựng từ CLI làm identity của existing
experiment.

#### Phân loại config

Các field định nghĩa artifact/index phải lấy từ persisted metadata và không
được override âm thầm:

- `embedding_models`;
- `frame_sampling`;
- `keyframe_percentiles`;
- `index_backend`;
- `data_dir` nếu artifact phụ thuộc dataset/path này.

Các field chỉ ảnh hưởng runtime có thể cho phép override có chủ đích:

- `device`;
- `top_k` nếu chỉ là output/candidate setting;
- UI host/port;
- logging và worker/performance knobs không đổi embedding space.

Danh sách này là **đề xuất thiết kế**. Trước khi implement, team cần chốt field
nào thật sự tham gia experiment identity; `PipelineConfig.config_hash()` hiện
đang hash cả `top_k` và `device` tại
[`settings.py:55-70`](../src/config/settings.py#L55-L70), nên nếu muốn coi chúng
là runtime override thì phải điều chỉnh semantics hash/version metadata tương
ứng.

#### Hành vi khi mismatch

Khuyến nghị fail-fast, không âm thầm chọn một bên:

```text
Persisted embedding_models: jina-clip-v2,siglip2-so400m
Runtime embedding_models:   beit3

Refuse to open: runtime model set does not match indexed experiment.
```

Nếu UI cho chọn experiment, dropdown chỉ chọn `experiment name`; sau activation,
backend phải đọc persisted config và sinh model controls từ active Retriever.
CLI `--experiment-name` vẫn hữu ích cho automation/default activation nhưng
không phải nguồn retrieval model config.

### 2.11. Các bước implement đề xuất

1. Thêm parser/loader có versioning cho `runs/<experiment>/config.json`.
2. Dựng lại `PipelineConfig` với đúng kiểu (`Path`, tuple, float, int).
3. Tách persisted index-defining fields khỏi runtime override fields.
4. Sửa `Experiment.open()` hoặc thêm API rõ nghĩa như
   `Experiment.open_persisted(runs_dir, name, runtime_overrides=...)`.
5. Validate persisted model set với embedding artifacts và/hoặc Qdrant schema
   trước khi warmup/serve.
6. Log active experiment, persisted config hash, resolved model IDs/revisions
   và runtime overrides.
7. Sinh UI model picker từ `retriever.embedders`, không từ HTML hard-code.
8. Có migration/error rõ cho experiment cũ thiếu hoặc hỏng `config.json`.

### 2.12. Test hồi quy cần có

Characterization test hiện tại nên được giữ trong commit tái hiện hoặc đổi tên
rõ ràng. Sau khi sửa, test chính phải mô tả hành vi mong muốn.

#### A. Restore persisted config

```python
def test_open_uses_persisted_index_config(tmp_path):
    # create bằng A, open existing experiment
    # expected: embedding_models vẫn là A
    ...
```

#### B. Reject explicit mismatch

```python
def test_open_rejects_runtime_embedding_model_mismatch(tmp_path):
    # persisted A, explicit runtime B
    # expected: typed config error chứa field/value mismatch
    ...
```

#### C. Allow runtime-only override

```python
def test_open_allows_device_override_without_changing_model_set(tmp_path):
    # expected: persisted model set giữ nguyên; device override có hiệu lực
    ...
```

#### D. Missing/corrupt metadata

```python
def test_open_fails_clearly_when_config_json_is_missing(tmp_path): ...
def test_open_fails_clearly_when_config_json_is_invalid(tmp_path): ...
```

#### E. Downstream Retriever

Mock model loader để chứng minh `build_retriever()` chỉ dựng đúng persisted
model set; không cần tải model thật.

#### F. Integration artifact/schema

Tạo experiment A, cố serve bằng B và xác nhận lỗi xuất hiện ở bước validation
trước khi nhận search request, thay vì lỗi muộn trong Qdrant/WSF.

### 2.13. Acceptance criteria

CN-01 chỉ được coi là resolved khi:

- Cùng experiment name luôn khôi phục cùng index-defining config từ metadata.
- Runtime không thể âm thầm đổi `embedding_models` của existing experiment.
- Mismatch trả lỗi có field, persisted value và runtime value rõ ràng.
- Runtime-only override đã được định nghĩa và test.
- Retriever model set khớp persisted config và artifact/schema.
- UI hiển thị active experiment/config/model set thực tế.
- Test hồi quy chạy pass trên branch mục tiêu `version3_10/8/2026` hoặc branch
  kế nhiệm được team chốt.

### 2.14. Quyết định cần team chốt

1. `config.json` có phải nguồn sự thật duy nhất cho existing experiment không?
2. Field nào là index-defining, field nào được runtime override?
3. Khi metadata cũ thiếu/hỏng: fail, migrate hay cho compatibility flag?
4. Chọn experiment ở UI sẽ activate một Retriever duy nhất hay cache nhiều
   Retriever?
5. Competition mode có khóa việc đổi experiment sau khi server Ready không?

---

## 3. Backlog issue cần kiểm chứng và sửa dần

Các mục dưới đây được phát hiện bằng static audit trên
`version3_10/8/2026@d2a4497`. Chúng chưa có trạng thái “đã tái hiện” cho đến
khi test tương ứng được viết và chạy. Thứ tự ưu tiên đề xuất dựa trên khả năng
làm sai ranking, làm pipeline không khởi động hoặc khiến UI khác hành vi thật.

| ID | Phát hiện | Mức độ hiện tại | Ưu tiên | Test |
|---|---|---|---|---|
| CN-02 | Bỏ chọn tất cả model lại dùng tất cả model | Đã chứng minh bằng source | High | Chưa viết |
| CN-03 | WSF cộng theo row nhưng không validate frame alignment | **DONE (2026-08-14)** | — | Build/quality-gate/WSF/loader regression tests |
| CN-04 | UI hard-code model list | Đã chứng minh bằng source | High | Chưa viết |
| CN-05 | UI server tự sinh manifest bằng metadata suy đoán | **DONE (2026-08-14)** | — | Unit + integration + CLI + real-video smoke |
| CN-06 | Warmup lỗi nhưng vẫn báo tất cả model ready | **DONE (2026-08-14)** | — | Per-component health + activation/fallback tests |
| CN-07 | Chuỗi `"false"` bị ép thành `True` | Rủi ro có điều kiện, xác suất thấp với UI hiện tại | Low | Chưa viết |
| CN-08 | Query processor tự disable cho cả session sau một lỗi | Đã chứng minh bằng source | Medium | Chưa viết |
| CN-09 | Tên model lạ fallback sang SigLIP | **DONE (2026-08-14)** | — | Registry + offline + retrieval provenance tests |
| CN-10 | Result có frame path thiếu/sai bị loại khỏi output | **DONE (2026-08-14)** | — | Resolver + gate + diagnostics + migration tests |
| CN-11 | Reranker limit hard-code 50 | Đề xuất rà soát; chưa xác định là lỗi | Low | Chưa viết |
| CN-12 | Caption có trong backend nhưng UI không render | Đã chứng minh bằng source | Medium | Chưa viết |

### CN-02 — Bỏ chọn tất cả model lại dùng tất cả model

**Bằng chứng:** UI có thể tạo list rỗng từ các checkbox; endpoint đổi list đó
thành `None` tại [`server.py:490-513`](../src/ui/server.py#L490-L513). Retriever
hiểu `None` là toàn bộ embedder tại
[`search.py:129-137`](../src/retrieval/search.py#L129-L137).

```text
enabled_models=[]
→ payload.get(...) or None
→ enabled_models=None
→ return self.embedders
```

**Expected behavior đã chốt với team:** nếu người dùng không chọn model nào,
hệ thống phải dừng trước retrieval và hiển thị:

```text
Please select at least one available embedding model.
```

Không chạy zero-model search và không tự bật lại tất cả model.

**Hướng sửa đã thống nhất:** áp dụng validation ở cả hai lớp:

1. UI lấy kết quả `getEnabledModels()` trước khi gọi `fetch()`. Nếu list rỗng,
   hiển thị thông báo trên và không gửi request.
2. Backend không dùng `payload.get("enabled_models") or None`, vì biểu thức này
   làm mất khác biệt giữa list rỗng và field thiếu.
3. Backend nhận `enabled_models=[]` phải trả HTTP 400 với cùng thông báo, phòng
   trường hợp API được gọi trực tiếp mà không qua UI.
4. `Retriever._select_embedders([])` cũng phải raise `ValueError` rõ nghĩa, tạo
   lớp bảo vệ cuối cùng trước khi embed query/load frame embedding.

Semantics đề xuất:

| Input | Hành vi |
|---|---|
| Field `enabled_models` bị thiếu | Dùng tất cả model của active experiment để giữ backward compatibility |
| `enabled_models: null` | Dùng tất cả model, hoặc reject nếu team muốn API strict; cần chốt nhất quán |
| `enabled_models: []` | HTTP 400; không chạy retrieval |
| List có ít nhất một model hợp lệ | Chỉ chạy các model được chọn |
| List chỉ chứa model không tồn tại | HTTP 400 và liệt kê available models |

**Test cần viết:**

1. `Retriever._select_embedders([])` raise với message
   `Please select at least one available embedding model.`
2. `_select_embedders(None)` trả toàn bộ configured embedders.
3. List chỉ có model không tồn tại raise và liệt kê available models.
4. API nhận `enabled_models=[]` trả HTTP 400 và spy/mock xác nhận
   `retriever.search()` không được gọi.
5. UI không gọi `fetch()` khi mọi checkbox bị bỏ chọn và hiển thị đúng message.

**Acceptance criteria:**

- Không còn phép chuyển `[] → None`.
- Không model nào được embed/search khi selection rỗng.
- UI và API trả cùng một thông báo dễ hiểu.
- Field thiếu vẫn giữ semantics backward-compatible đã chốt.
- Các test Retriever, API và UI phía trên đều pass.

### CN-03 — WSF không validate frame alignment giữa model

> **Trạng thái hiện tại: DONE trên branch `version3_10/8/2026` (2026-08-14).**
> Phần bằng chứng dưới đây mô tả baseline trước sửa. Implementation hiện yêu
> cầu cùng unique frame-ID set, cho phép row order khác nhau, join theo
> `frame_id` ở cả `build_index()` và `wsf_fuse()`, đồng thời ghi alignment
> evidence vào readiness. Test/bằng chứng triển khai được tổng hợp tại
> [`codenova-offline-indexing-issue-scope.md`](./codenova-offline-indexing-issue-scope.md#51-cn-03--alignment-giữa-embedding-artifacts).

**Bằng chứng baseline:** [`wsf_fuse()`](../src/retrieval/fusion.py#L120-L165) lấy
`frame_records` và `n_frames` từ model đầu, sau đó gán similarity của từng model
vào cùng ma trận theo row:

```python
_, frame_records, _ = model_data[first_model]
n_frames = len(frame_records)
...
scores[:, i] = frame_embs @ q
```

Comment tại dòng 130 thừa nhận giả định mọi model có cùng số frame và cùng
`frame_records`, nhưng code không validate giả định đó.

```text
khác số frame → NumPy shape/broadcast error
cùng số frame, khác thứ tự → cộng score của hai frame khác nhau, có thể sai âm thầm
```

**Test cần viết:** ba fixture: khác số frame; cùng IDs nhưng đảo thứ tự; một
model thiếu ID. Test phải kiểm tra bằng `frame_id`, không chỉ vector shape.

**Hướng sửa:** align mọi model bằng `frame_id` trước fusion hoặc build một
canonical common-ID order có validation. Fail-fast nếu duplicate/missing ID
không đúng policy. Không fusion bằng row position chưa xác minh.

### CN-04 — UI hard-code model list

**Mức độ xác nhận:** đã chứng minh bằng source trên
`version3_10/8/2026@d2a4497`; chưa có UI/API test tự động.

**Tóm tắt:** model picker hiện không lấy model từ experiment đang phục vụ.
Ba checkbox Jina/SigLIP/Vietnamese được viết cố định trong HTML tại
[`server.py:841-847`](../src/ui/server.py#L841-L847). JavaScript chỉ đọc các
checkbox có sẵn này tại [`server.py:1039-1044`](../src/ui/server.py#L1039-L1044),
nên browser không có bước nào hỏi backend về model set thật.

Trong khi đó backend dựng Retriever từ
`experiment.config.embedding_models` tại
[`search.py:163-189`](../src/retrieval/search.py#L163-L189). `serve_ui()` đã có
active `experiment` và tạo active Retriever trước khi build handler tại
[`server.py:165-190`](../src/ui/server.py#L165-L190), nhưng dữ liệu này chưa
được dùng để sinh model picker.

Hai nguồn hiện độc lập:

```text
Model backend thực tế
→ persisted config của active experiment (sau khi CN-01 được sửa)
→ build/validate Retriever
→ retriever.embedders

Model hiển thị trên UI hiện tại
→ ba checkbox hard-code trong INDEX_HTML
```

**Quá trình dẫn đến lỗi:** giả sử active experiment chỉ có BEiT-3:

```text
experiment.config.embedding_models = ("beit3",)
→ backend dựng retriever.embedders = {"beit3": ...}
→ UI vẫn hiển thị Jina, SigLIP, Vietnamese
→ UI không có checkbox BEiT-3
→ user không thể chọn đúng model thật của experiment
→ request có thể gửi toàn các tên không thuộc active Retriever
```

Ngược lại, experiment chỉ có Jina nhưng UI vẫn hiển thị SigLIP và Vietnamese.
Vì vậy checkbox checked không chứng minh model thuộc experiment, embedder đã
load thành công hay artifact của model tồn tại.

**Expected behavior đã chốt:** model phải bắt nguồn từ experiment được chọn,
nhưng UI nên nhận danh sách cuối cùng từ active Retriever đã build/validate:

```text
user chọn experiment
→ backend đọc persisted config của experiment
→ validate embedding artifacts/model configuration
→ build active Retriever
→ lấy retriever.embedders.keys()
→ trả model capabilities cho UI
→ UI render checkbox động
```

Không tạo thêm model ngoài experiment chỉ vì tên model được hard-code trên UI.
Checkbox chỉ được phép chọn subset của model set hợp lệ trong active Retriever.

**Hướng sửa đã thống nhất:** sau CN-01, backend cung cấp active experiment và
model capabilities, ví dụ:

```json
{
  "active_experiment": "competition-final",
  "models": [
    {"name": "jina-clip-v2", "status": "ready"},
    {"name": "siglip2-so400m", "status": "ready"}
  ]
}
```

UI xóa ba checkbox model viết cứng và render từ response trên. Nguồn model
cuối cùng nên là `retriever.embedders.keys()` sau validation; persisted config
là nguồn gốc, còn active Retriever là tập backend thực sự có thể phục vụ.

**Test cần viết:**

1. Active Retriever BEiT-3-only → UI/API chỉ trả checkbox BEiT-3.
2. Active Retriever Jina-only → không hiển thị SigLIP/Vietnamese.
3. Experiment có nhiều model → UI options khớp chính xác
   `retriever.embedders.keys()`.
4. Model validation/load thất bại → không đánh dấu `ready`; competition mode
   fail activation hoặc hiển thị trạng thái lỗi theo policy của CN-06.
5. Model name chứa ký tự đặc biệt phải được render/escape và round-trip đúng
   trong `enabled_models`.

**Acceptance criteria:**

- Không còn checkbox model hard-code trong `INDEX_HTML`.
- UI model set bằng đúng active Retriever model set.
- Đổi/activate experiment cập nhật model picker tương ứng.
- UI không thể gửi model ngoài active experiment.
- Bỏ chọn tất cả vẫn tuân theo CN-02: báo lỗi và không chạy retrieval.
- UI hiển thị active experiment cùng trạng thái model để người vận hành kiểm
  chứng pipeline đang phục vụ.

**Quan hệ phụ thuộc:** thực hiện sau hoặc cùng CN-01. Nếu `Experiment.open()`
vẫn dùng config runtime sai, model picker động chỉ hiển thị chính xác một
Retriever được dựng từ sai nguồn config.

### CN-05 — Tự sinh manifest bằng metadata suy đoán

> **Trạng thái hiện tại: DONE trên branch `version3_10/8/2026` (2026-08-14).**
> Phần phân tích bên dưới mô tả failure mode của baseline trước khi sửa và được
> giữ làm lịch sử nguyên nhân. Implementation hiện tại đã xóa auto-generation
> khỏi UI, yêu cầu fresh `READY`, thêm guarded `offline-index`, strict/atomic
> manifest, explicit repair và cross-stage quality gate. Bằng chứng triển khai
> và test được tổng hợp tại
> [`codenova-offline-indexing-issue-scope.md`](./codenova-offline-indexing-issue-scope.md#31-cn-05--manifest-và-metadata-không-bảo-đảm-đầy-đủ).

**Mức độ xác nhận ở baseline trước sửa:** source chứng minh UI có thể tự ghi
metadata suy đoán. Production experiment cụ thể có từng bị ảnh hưởng hay không
vẫn cần đối chiếu artifact/log lịch sử của experiment đó.

**Bằng chứng về hành vi baseline:** `_ensure_manifests()` từng được gọi khi serve tại
[`server.py:57-174`](../src/ui/server.py#L57-L174). Nếu `frames.jsonl` thiếu hoặc
rỗng, code:

- đọc file frame-ID đầu tiên tìm thấy rồi suy `video_id` bằng
  `fid.split("_")[0]` ([`server.py:69-105`](../src/ui/server.py#L69-L105));
- hoặc quét thư mục frame và gán mọi frame vào shot `s0`
  ([`server.py:111-131`](../src/ui/server.py#L111-L131));
- tính timestamp bằng `frame_number / 25.0`;
- ghi lại cả `frames.jsonl` và `videos.jsonl`, trong đó video path là convention
  suy đoán, checksum là `"dummy_checksum"` và size bằng 0
  ([`server.py:134-151`](../src/ui/server.py#L134-L151)).

```text
manifest thiếu
→ suy video/shot/frame metadata từ tên file
→ hard-code 25 FPS
→ ghi manifest mới
→ metadata có thể sai với video/naming convention thật
```

#### Metadata chuẩn vốn được tạo ở đâu?

Pipeline offline đã có nguồn dữ liệu đáng tin hơn cơ chế phục hồi của UI:

| Artifact | Stage tạo | Bằng chứng về nguồn dữ liệu |
|---|---|---|
| `videos.jsonl` | `ingest` | [`discover_videos()`](../src/video/discovery.py#L13-L37) tính SHA-256 và file size thật; [`ingest_videos()`](../src/indexing/ingest.py#L16-L45) ghi `VideoRecord` |
| `shots.jsonl` | `detect-shots` | [`detect_shots()`](../src/indexing/shots.py#L37-L76) decode video và ghi output của detector |
| `frames.jsonl` | `extract-frames` | [`extract_frames()`](../src/indexing/frames.py#L27-L53) ghi `FrameRecord` do extractor trả về |
| `timestamp_sec` | `FFmpegFrameExtractor` | [`video/frames.py:40-79`](../src/video/frames.py#L40-L79) probe FPS thật rồi tính `frame_index / fps` |

Vì vậy hướng sửa không phải làm parser suy đoán trong UI thông minh hơn. Hướng
đúng là bảo đảm offline metadata hoàn chỉnh, kiểm chứng nó trước activation và
để retrieval chỉ đọc.

#### Tại sao indexing vẫn có thể thiếu metadata?

1. `discover_videos()` tính checksum ngoài `try` theo từng video của
   `ingest_videos()` ([`discovery.py:22-37`](../src/video/discovery.py#L22-L37),
   [`ingest.py:26-43`](../src/indexing/ingest.py#L26-L43)). Một file không đọc
   được có thể làm discovery raise trước khi video đó được đánh dấu `FAILED`.
2. Shot decode/detection lỗi được đánh dấu `FAILED` rồi pipeline tiếp tục video
   sau ([`shots.py:54-75`](../src/indexing/shots.py#L54-L75)). Vì vậy
   `videos.jsonl` có thể chứa video không có shot.
3. `extract_frames()` chỉ warning và skip video không có shot; lỗi extractor
   cũng được ghi `FAILED` rồi tiếp tục
   ([`frames.py:32-52`](../src/indexing/frames.py#L32-L52)). Vì vậy video/shot
   có thể không có frame metadata.
4. JSONL được append trực tiếp, không atomic; reader bỏ qua dòng JSON hỏng
   ([`manifest.py:20-56`](../src/indexing/manifest.py#L20-L56)). Process bị kill
   có thể để manifest partial mà lần đọc sau vẫn tiếp tục.
5. `embed_frames()` gặp manifest frame rỗng chỉ warning và trả 0
   ([`embeddings.py:117-126`](../src/indexing/embeddings.py#L117-L126)), nên
   command có thể kết thúc không exception dù chưa có dữ liệu để embed.
6. Vietnamese embedding chủ động skip frame chưa có caption
   ([`embeddings.py:162-179`](../src/indexing/embeddings.py#L162-L179)). Đây
   không làm thiếu `frames.jsonl`, nhưng tạo coverage embedding nhỏ hơn metadata.
7. `build_index()` lấy giao frame-ID của mọi model
   ([`build_index.py:63-78`](../src/indexing/build_index.py#L63-L78)). Nó bảo đảm
   point đã build có đủ vector, nhưng có thể build thành công trên một subset
   mà không chứng minh toàn bộ video/frame metadata hoàn chỉnh.

#### Expected behavior đã đề xuất

```text
offline indexing tạo metadata từ nguồn thật
→ validate-experiment đối chiếu toàn pipeline
→ chỉ experiment đạt quality gate mới được đánh dấu READY
→ serve-ui chạy validator read-only
→ hợp lệ: build Retriever và serve
→ không hợp lệ: từ chối activation, báo chính xác stage/record bị thiếu
```

`serve-ui` không tự tạo, overwrite hay “repair” manifest.

#### Định hướng sửa chi tiết

##### Bước 1 — Làm discovery báo lỗi theo từng file

**Lý do:** hiện checksum được tính tại
[`discovery.py:22-37`](../src/video/discovery.py#L22-L37) trước `try` ghi record
của [`ingest.py:36-43`](../src/indexing/ingest.py#L36-L43).

**Thay đổi đề xuất:** discovery trả/yield từng candidate hoặc một kết quả có
`path + VideoRecord/error`; bắt `OSError`/checksum failure theo file, ghi log và
job/inventory failure. Cuối stage trả summary `discovered/completed/failed`, và
competition indexing phải có exit code khác 0 nếu còn failure chưa được waive.

##### Bước 2 — Ngăn duplicate và làm manifest write an toàn hơn

**Lý do:** `JsonlManifest.append/extend()` chỉ append
([`manifest.py:20-29`](../src/indexing/manifest.py#L20-L29)); nếu state chưa
`COMPLETED` nhưng record đã tồn tại, ingest có thể append lại.

**Thay đổi đề xuất:** enforce uniqueness theo primary key (`video_id`,
`shot_id`, `frame_id`) trong validator; với rewrite/compaction dùng temp file +
atomic rename. Resume phải reconcile manifest với `jobs.sqlite`, không xem một
trong hai nguồn là đủ khi chúng mâu thuẫn.

##### Bước 3 — Fail rõ khi prerequisite manifest thiếu/rỗng

**Lý do:** shot/frame/embed stage hiện có các đường trả 0 hoặc skip
([`shots.py:37-50`](../src/indexing/shots.py#L37-L50),
[`frames.py:40-43`](../src/indexing/frames.py#L40-L43),
[`embeddings.py:123-126`](../src/indexing/embeddings.py#L123-L126)).

**Thay đổi đề xuất:** phân biệt hai trường hợp:

```text
không còn item pending vì tất cả đã COMPLETED → return 0 hợp lệ
input manifest thiếu/rỗng hoặc upstream incomplete → raise typed error
```

Mỗi stage phải kiểm tra prerequisite trước khi làm việc và in command cần chạy
để phục hồi upstream.

##### Bước 4 — Thêm validator read-only cho experiment

**Lý do:** build-index hiện chủ yếu validate embedding artifacts và lấy giao
frame-ID, chưa kiểm tra completeness xuyên toàn pipeline
([`build_index.py:43-78`](../src/indexing/build_index.py#L43-L78)).

**Thay đổi đề xuất:** thêm một core validator dùng được bởi CLI và serve, đối
chiếu tối thiểu:

```text
videos.jsonl video_id unique, path tồn tại, checksum/size hợp lệ
shots.jsonl video_id ⊆ videos; shot_id unique; boundary hợp lệ
frames.jsonl video_id/shot_id resolve được; frame_id unique; file tồn tại
embedding frame_ids mỗi model ↔ vector row count
embedding frame_ids ⊆ frames.jsonl
WSF models có frame-ID set/order đáp ứng policy CN-03
jobs.sqlite không còn FAILED/PENDING bắt buộc
```

Validator trả structured report và exit code, không sửa file.

##### Bước 5 — Định nghĩa quality gate và readiness artifact

**Lý do:** hiện một stage có thể partial-success nhưng chưa có một bằng chứng
tổng hợp rằng experiment đủ điều kiện serve.

**Thay đổi đề xuất:** command ví dụ:

```bash
codenova validate-experiment --experiment-name competition-final
```

chỉ tạo `validation-report.json`/trạng thái `READY` sau khi tất cả invariant bắt
buộc pass. Report phải chứa counts và coverage, ví dụ:

```text
videos discovered / with shots / with frames
manifest frames / existing frame files
embedded frames per model / common retrieval frames
missing, duplicate, corrupt and failed-item counts
```

##### Bước 6 — Retrieval/UI chỉ validate và fail-fast

**Lý do:** `serve_ui()` hiện gọi `_ensure_manifests()` trước khi dựng Retriever
([`server.py:162-174`](../src/ui/server.py#L162-L174)).

**Thay đổi đề xuất:** thay auto-recovery bằng validator read-only. Nếu fail,
không warmup model, không mở trạng thái `Ready`, và trả lỗi có action:

```text
Experiment is not ready:
- 2 videos have no shots
- 17 frame IDs have no metadata
- 1 embedding file has row-count mismatch
Run: codenova validate-experiment ...
```

##### Bước 7 — Nếu thật sự cần repair, tách thành CLI explicit

**Lý do:** repair là hành động ghi dữ liệu và cần provenance; startup UI không
phải nơi phù hợp để làm việc này.

**Thay đổi đề xuất:** một command riêng có `--dry-run`, backup, atomic write và
chỉ dùng nguồn thật. Không hard-code FPS, không dùng `split("_")` để đoán schema,
không ghi checksum giả. Record không thể phục hồi phải được báo lỗi thay vì chế
dữ liệu mặc định.

#### Test cần viết theo từng lớp

1. Discovery: một video không đọc được không làm mất inventory của video khác;
   failure được report và exit status phản ánh partial failure.
2. Manifest: process/write interruption hoặc corrupt line được validator phát
   hiện; duplicate ID bị reject.
3. Shot/frame prerequisites: manifest thiếu/rỗng raise typed error, không chỉ
   return 0.
4. Frame metadata: FPS thật từ `probe_fps()` được giữ; video/shot ID lấy trực
   tiếp từ record, không suy từ filename.
5. Cross-manifest: video thiếu shot, shot thiếu video, frame thiếu shot, file
   ảnh thiếu đều được report đúng count/ID.
6. Embedding: vector row count khớp frame-ID count; IDs tồn tại trong
   `frames.jsonl`; coverage từng model được báo.
7. Serve: experiment invalid không làm thay đổi filesystem, không build/warmup
   Retriever và không báo `Ready`.
8. Repair CLI: `--dry-run` không ghi; repair thật backup trước khi atomic replace.

#### Acceptance criteria

- Offline pipeline tạo `videos/shots/frames` metadata từ nguồn thật.
- Mọi partial failure xuất hiện trong structured validation report và exit code.
- Không stage downstream âm thầm coi prerequisite rỗng là success hợp lệ.
- Experiment chỉ được serve sau khi quality gate pass.
- Khởi động/đóng UI không thay đổi manifest hoặc artifact indexing.
- Không còn timestamp hard-code 25 FPS, checksum giả hay ID suy đoán trong
  retrieval path.
- Coverage metadata/embedding của từng model được đo và truy vết được.

### CN-06 — Warmup lỗi nhưng vẫn báo tất cả model ready

#### Tóm tắt và mức độ xác nhận

**Trạng thái triển khai: DONE (2026-08-14).** Warmup ghi health theo từng
component. Mọi embedder thuộc config của experiment được xem là bắt buộc: từng
model vẫn được kiểm tra độc lập, nhưng chỉ cần một model lỗi thì activation bị
chặn trước khi bind HTTP socket. Reranker là tầng tùy chọn: load lỗi sẽ disable
component và startup ở trạng thái `DEGRADED`; inference lỗi trả ranking trước
rerank rồi giữ reranker ở trạng thái disabled.

#### Bằng chứng triển khai

1. `WarmupComponentHealth` và `WarmupReport` lưu `READY/FAILED`, component và
   error tại [`src/ui/server.py`](../src/ui/server.py).
2. Mỗi embedder có `try/except` riêng. Sau khi kiểm tra hết, danh sách
   `failed_embedders` làm raise `RetrievalError`, nên `serve_ui()` chưa tạo
   `ThreadingHTTPServer` và chưa mở listener.
3. UI reranker và reranker nội bộ của `Retriever` đều được warmup. Load lỗi làm
   chúng được thay bằng `None`; không còn đường request âm thầm gọi lại model
   đã fail load.
4. `_FailOpenReranker` xử lý reranker tùy chọn của UI: inference lỗi được log
   bằng event `RERANKER_DEGRADED`, trả input ranking và không gọi backend đó ở
   request sau. `Retriever.search()` áp dụng cùng policy cho reranker nội bộ tại
   [`src/retrieval/search.py`](../src/retrieval/search.py).
5. Offline readiness, artifact fingerprint và frame-file validation vẫn chạy
   trước khi dựng/warmup model; health online không thay thế quality gate.

#### Chuỗi dẫn đến lỗi

```text
Warmup embedder/reranker raise exception
→ exception bị xem là non-fatal
→ không lưu health state và không vô hiệu hóa thành phần lỗi
→ startup vẫn log tất cả model ready
→ request sau vẫn có thể gọi lại thành phần đã warmup thất bại
→ request lỗi hoặc trạng thái vận hành khác với thông báo startup
```

Với embedder, một lỗi còn làm các embedder đứng sau không được kiểm tra vì cả
vòng lặp nằm trong cùng một `try`. Với reranker, lỗi có thể xuất hiện ở startup
hoặc chỉ xuất hiện khi query thật xử lý batch ảnh, chẳng hạn thiếu dependency,
checkpoint không tải được, ảnh hỏng hoặc hết VRAM.

#### Policy đã áp dụng

```text
offline readiness không READY → từ chối activation
embedder trong experiment fail warmup → kiểm tra tiếp model còn lại, sau đó từ chối activation
reranker fail load → disable reranker, status DEGRADED
reranker fail inference → trả ranking trước rerank, disable cho request sau
mọi component pass → status READY và mới bind HTTP socket
```

#### Đề xuất mở rộng, không phải phạm vi sửa chính

- Intelligent Search có thể được phát triển để lựa chọn và fusion KIS/OCR/ASR
  theo query và trạng thái khả dụng của từng nguồn. Code hiện có tại
  [`intelligent_search.py:27-87`](../src/retrieval/intelligent_search.py#L27-L87),
  nhưng việc hoàn thiện cơ chế này không phải điều kiện đóng CN-06.
- Có thể bổ sung competition profile trong tương lai để khai báo model/nguồn
  bắt buộc. Profile chỉ nên quyết định policy readiness; không thay thế health
  check và validation artifact.

#### Test hồi quy

[`tests/unit/test_warmup_health.py`](../tests/unit/test_warmup_health.py) kiểm
tra model phía sau vẫn warmup khi model giữa lỗi, mandatory embedder chặn
activation, cả hai reranker bị disable khi load lỗi, và inference failure chỉ
được gọi một lần rồi trả kết quả pre-rerank. Quality-gate/readiness tests nằm ở
[`tests/unit/test_index_quality_gate.py`](../tests/unit/test_index_quality_gate.py).

#### Acceptance criteria

- Không còn trường hợp exception warmup đi kèm thông báo tất cả model ready.
- Mỗi embedder/reranker có trạng thái và nguyên nhân lỗi truy vết được.
- Lỗi một model không ngăn health check các model còn lại.
- Thành phần đã warmup thất bại không bị gọi lại âm thầm trong request.
- Artifact và coverage offline được kiểm tra độc lập trước khi experiment được
  activate.

### CN-07 — Boolean string được parse sai

**Mức độ xác nhận và khả năng xảy ra:** đây là lỗi validation API có điều
kiện, không phải lỗi được kỳ vọng xảy ra trong luồng UI hiện tại. UI đọc
checkbox qua thuộc tính JavaScript `.checked` tại
[`server.py:1098-1104`](../src/ui/server.py#L1098-L1104), nên giá trị được gửi
bởi UI là JSON boolean `true/false`, không phải string `"true"/"false"`.

Khả năng xảy ra hiện tại vì thế được đánh giá là **thấp** nếu endpoint chỉ được
gọi bởi UI này. Rủi ro vẫn tồn tại khi một script, client tích hợp, form hoặc
tầng trung gian gọi API và serialize boolean thành string.

**Bằng chứng cho failure mode:** endpoint gọi `bool(value)` cho
`use_reranker` và `use_llm` tại
[`server.py:490-505`](../src/ui/server.py#L490-L505). Trong Python, mọi string
không rỗng đều truthy, nên `bool("false") is True`.

```text
Client gửi {"use_reranker": "false"}
→ JSON parser tạo Python string "false"
→ bool("false") trả về True
→ reranker được bật trái với ý định của client
```

**Test cần viết:** request lần lượt với JSON boolean `false`, string `"false"`,
`null` và field thiếu; chốt expected schema cho từng trường hợp.

**Hướng sửa:** chưa cần xem đây là lỗi ưu tiên cao, nhưng vẫn nên xử lý để làm
rõ contract của API. Dùng schema validation và chỉ nhận JSON boolean; trả HTTP
400 cho string thay vì tự coercion mơ hồ. JSON boolean `true/false` tiếp tục
được chấp nhận; field thiếu hoặc `null` phải có hành vi mặc định được quy định
rõ trong schema.

### CN-08 — Query processor tự disable sau một lỗi

#### Tóm tắt và mức độ xác nhận

`LlmQueryProcessor` có cơ chế fallback có chủ ý: nếu LLM phụ trợ không hoạt
động, retrieval vẫn dùng raw query thay vì làm hỏng toàn bộ request. Tuy nhiên,
chỉ một exception trong `process()` cũng đặt `_disabled=True`; mọi request sau
trên cùng object `Retriever` sẽ bỏ qua LLM cho đến khi process được khởi động
lại.

Issue này đang ở mức **đã chứng minh bằng source**. Chưa có runtime evidence để
kết luận deployment thực tế từng bị suy giảm theo failure mode này, và chưa có
thông tin đủ để khẳng định policy disable cả session có phải chủ đích cuối cùng
của nhóm retrieval hay không.

#### Bằng chứng source

1. `process()` trả pass-through ngay khi `_disabled` hoặc `use_llm=False` tại
   [`query_processor.py:95-99`](../src/retrieval/query_processor.py#L95-L99).
2. Toàn bộ thao tác gọi LLM, parse JSON và dựng `ProcessedQuery` nằm trong một
   `try`; mọi `Exception` đều đi vào cùng một nhánh tại
   [`query_processor.py:139-164`](../src/retrieval/query_processor.py#L139-L164).
3. Nhánh lỗi đặt `self._disabled = True` rồi trả raw query. Không có retry,
   cooldown, failure counter hoặc reset state trong class này.
4. `_extract_json()` có thể raise do response không chứa JSON, JSON sai cú pháp
   hoặc JSON không phải object tại
   [`query_processor.py:226-237`](../src/retrieval/query_processor.py#L226-L237).
   Vì vậy processor có thể bị disable không chỉ bởi lỗi kết nối dài hạn mà còn
   bởi một response sai format nhất thời.
5. `query_processor` được giữ làm thuộc tính của `Retriever` tại
   [`search.py:45-56`](../src/retrieval/search.py#L45-L56) và được tái sử dụng
   khi `search()` xử lý các query tại
   [`search.py:88-90`](../src/retrieval/search.py#L88-L90). `_disabled` vì thế
   có thể ảnh hưởng các request sau, không chỉ request phát sinh exception.

```text
LLM timeout/parse error một lần
→ _disabled=True
→ request hiện tại fallback sang raw query
→ mọi query sau trong cùng session không thử LLM nữa
→ query preprocessing phụ thuộc lịch sử lỗi của process
```

#### Ảnh hưởng có điều kiện

Retrieval không dừng hoàn toàn vì raw query vẫn được đưa sang bước embedding.
Tuy nhiên các request sau có thể không còn translation/visual prompt expansion,
OCR/ASR keywords, metadata và trọng số do LLM sinh. Mức ảnh hưởng thực tế phụ
thuộc model, ngôn ngữ query và việc các chức năng này có được dùng trong track
đang chạy hay không.

#### Định hướng đề xuất — cần xác nhận policy

Do chưa rõ policy mong muốn của nhóm retrieval, báo cáo không mặc định rằng
processor luôn phải retry hoặc luôn phải disable cả session. Đề xuất cân nhắc:

1. Giữ fallback raw query để một lỗi LLM không làm hỏng retrieval.
2. Phân biệt lỗi parse nhất thời, timeout và lỗi kết nối kéo dài; không nhất
   thiết dùng cùng một policy cho mọi `Exception`.
3. Với lỗi đơn lẻ, chỉ fallback request hiện tại và cho request sau thử lại.
4. Nếu cần tránh timeout lặp lại, dùng circuit breaker có failure threshold,
   cooldown và trạng thái half-open thay cho disable vĩnh viễn.
5. Trả hoặc log mode `enhanced`/`fallback` và trạng thái processor để benchmark
   có thể truy vết query nào đã dùng LLM.
6. Quy định rõ `use_llm=False` là lựa chọn của request, không phải lỗi health và
   không được thay đổi circuit state.

Nếu policy được xác nhận là “sau lỗi đầu tiên phải disable đến khi restart”,
hành vi hiện tại vẫn cần observability rõ ràng để tránh việc các request sau âm
thầm chạy khác cấu hình kỳ vọng.

#### Test cần viết

1. Characterization test: fake client lỗi lần đầu rồi thành công; gọi
   `process()` hai lần và xác nhận behavior hiện tại không thử lại lần hai.
2. Regression test sau khi chốt policy: lỗi đơn lẻ chỉ fallback request hiện
   tại, hoặc circuit chỉ mở sau đúng failure threshold.
3. Test parse error, timeout và connection error riêng để xác nhận policy cho
   từng nhóm lỗi.
4. Test `use_llm=False` không làm thay đổi health/circuit state của processor.

#### Acceptance criteria cần chốt

- Lỗi LLM không làm retrieval request mất toàn bộ kết quả.
- Policy retry/disable được mô tả rõ và kiểm chứng bằng test.
- Không có thay đổi mode âm thầm: log/response cho biết query dùng enhanced hay
  fallback.
- Lỗi tạm thời không gây disable lâu hơn policy đã thống nhất.

### CN-09 — Tên model lạ fallback sang SigLIP

> **Trạng thái hiện tại: DONE trên branch `version3_10/8/2026` (2026-08-14).**
> Phần dưới giữ lại phân tích baseline. Implementation hiện dùng strict model
> registry chung cho preflight/factory, reject unknown trước khi sinh artifact,
> persist provenance và đối chiếu lại trước retrieval. Bằng chứng/test được tổng
> hợp tại
> [`codenova-offline-indexing-issue-scope.md`](./codenova-offline-indexing-issue-scope.md#32-cn-09--unknown-model-bị-ánh-xạ-sang-siglip).

#### Tóm tắt và mức độ xác nhận

`build_embedder()` nhận diện Jina, BEiT-3 và Vietnamese embedding bằng marker
trong tên. Mọi tên không khớp ba nhóm đó đều được chuyển sang
`SiglipEmbedder`, kể cả tên không chứa `siglip`, alias chưa được hỗ trợ hoặc
lỗi chính tả. Factory vì thế có thể che giấu lỗi cấu hình thay vì fail-fast.

Ở baseline, issue này ở mức **đã chứng minh bằng source**. Hậu quả thực tế phụ thuộc tên
model được truyền vào và việc `SiglipEmbedder` có tải được checkpoint tương ứng
hay không; chưa có bằng chứng deployment đã dùng một tên model sai.

#### Bằng chứng source

1. Factory kiểm tra lần lượt marker `jina`, `beit3` và Vietnamese tại
   [`embedding/__init__.py:17-45`](../src/modules/embedding/__init__.py#L17-L45).
2. Nhánh cuối không kiểm tra marker `siglip`; mọi tên còn lại đều được truyền
   vào `SiglipEmbedder` tại
   [`embedding/__init__.py:46-50`](../src/modules/embedding/__init__.py#L46-L50).
3. Offline indexing gọi trực tiếp factory theo từng tên trong experiment tại
   [`embeddings.py:194-199`](../src/indexing/embeddings.py#L194-L199).
4. Khi dựng retrieval, `build_retriever()` cũng gọi cùng factory cho toàn bộ
   `experiment.config.embedding_models` tại
   [`search.py:163-173`](../src/retrieval/search.py#L163-L173).

```text
model_name = "unknown-model" hoặc một typo
→ không khớp Jina/BEiT-3/Vietnamese
→ rơi vào nhánh mặc định SiglipEmbedder
→ lỗi cấu hình bị che giấu hoặc chỉ phát hiện muộn khi load model
→ encoder thực tế/model intent/artifact provenance có thể không nhất quán
```

#### Định hướng khắc phục

1. Dùng registry/alias rõ ràng cho các tên ngắn được hỗ trợ, ví dụ
   `siglip2-so400m`, thay vì xem mọi tên còn lại là SigLIP.
2. Sau khi kiểm tra alias, có thể nhận diện SigLIP khi tên hoặc Hugging Face
   model ID thực sự chứa marker `siglip`. Cách này vẫn hỗ trợ các tên như
   `google/siglip2-...` mà không biến `unknown-model` thành SigLIP.
3. Với tên không nhận diện được, ghi thông báo chứa tên nhận được và danh sách
   alias/kiến trúc hỗ trợ; không warning rồi âm thầm tạo SigLIP.
4. Trong offline indexing, unknown model nên raise typed configuration error và
   dừng trước khi tạo artifact. Tiếp tục index bằng encoder được đoán có thể tạo
   artifact sai provenance và khó sửa sau đó.
5. Khi activate experiment có nhiều model, có thể cân nhắc đánh dấu model không
   hợp lệ là `unavailable` và chỉ tiếp tục nếu policy cho phép degraded mode và
   còn ít nhất một model hợp lệ. Nếu không còn model hợp lệ thì activation phải
   thất bại. Đây là policy đề xuất, không phải hành vi hiện có.
6. Ghi resolved backend, model ID và revision vào metadata/readiness report để
   offline artifact và query encoder có thể được đối chiếu.

Warning vẫn hữu ích để chẩn đoán, nhưng warning không đủ nếu sau warning hệ
thống tiếp tục bằng một kiến trúc được đoán. Với unknown model, kết quả an toàn
là reject hoặc đánh dấu unavailable theo policy rõ ràng.

#### Test cần viết

1. Alias SigLIP hợp lệ như `siglip2-so400m` tạo `SiglipEmbedder`.
2. Hugging Face ID có marker `siglip` tạo `SiglipEmbedder` và giữ đúng model ID.
3. `unknown-model` và typo không chứa marker hợp lệ phải raise typed
   configuration error, không tạo `SiglipEmbedder`.
4. Offline indexing nhận unknown model phải dừng trước khi ghi embedding
   artifact cho model đó.
5. Nếu degraded activation được chọn, experiment nhiều model phải báo rõ model
   unavailable và không được tiếp tục khi không còn model hợp lệ.

#### Acceptance criteria

- Không còn nhánh mặc định ánh xạ mọi unknown model sang SigLIP.
- Alias và model ID SigLIP hợp lệ vẫn được nhận diện.
- Unknown model thất bại sớm với thông báo có thể hành động.
- Artifact lưu đủ backend/model ID/revision để truy vết encoder thực tế.
- Policy reject/degraded khi activation được mô tả và kiểm chứng bằng test.

### CN-10 — Frame path lỗi làm candidate biến mất

#### Tóm tắt và mức độ xác nhận

**Trạng thái triển khai: DONE (2026-08-14)** trong phạm vi tính nhất quán và
khả năng chẩn đoán frame artifact. Frame mới được lưu bằng đường dẫn POSIX tương
đối với `experiment.run_dir`; offline consumer, validator, retrieval hydrator và
endpoint ảnh dùng cùng resolver theo experiment, không còn phụ thuộc current
working directory. Activation từ chối run có frame thiếu/sai. Runtime vẫn loại
frame không thể hiển thị nhưng ghi reason code thay vì loại âm thầm.

Thay đổi này chủ ý không sửa WSF, trọng số, reranking, `top_k`, fusion pool hoặc
backfill. Backfill vẫn là đề xuất tối ưu riêng nếu team muốn bảo đảm luôn trả đủ
`top_k` khi file bị xóa sau activation.

#### Bằng chứng triển khai

1. Resolver và reason code `FRAME_PATH_MISSING`, `FRAME_FILE_MISSING`,
   `FRAME_PATH_OUTSIDE_EXPERIMENT` nằm tại
   [`src/core/paths.py`](../src/core/paths.py). Absolute legacy path trong run
   chỉ được đọc tương thích và bị đánh dấu non-canonical.
2. Frame extraction persist canonical run-relative path tại
   [`src/indexing/frames.py`](../src/indexing/frames.py); embedding, OCR và
   temporal loading resolve thành absolute path ngay tại consumer boundary ở
   [`src/indexing/embeddings.py`](../src/indexing/embeddings.py),
   [`src/indexing/extract_text.py`](../src/indexing/extract_text.py) và
   [`src/retrieval/temporal_search.py`](../src/retrieval/temporal_search.py).
3. Quality gate báo count/ratio và issue theo từng frame; activation recheck
   toàn bộ file tại [`src/indexing/validation.py`](../src/indexing/validation.py),
   được gọi trước khi dựng retriever/warmup tại
   [`src/retrieval/search.py`](../src/retrieval/search.py) và
   [`src/ui/server.py`](../src/ui/server.py).
4. Hydrator trả `HydrationBatch` gồm result hợp lệ và issue chi tiết; search log
   event `RETRIEVAL_CANDIDATES_DROPPED` kèm count, reason histogram và sample
   frame IDs tại [`src/retrieval/hydrator.py`](../src/retrieval/hydrator.py) và
   [`src/retrieval/search.py`](../src/retrieval/search.py).
5. Migration legacy mặc định dry-run, chỉ apply khi mọi record resolve về file
   bên trong run; nó cập nhật manifest tổng và partition, backup, ghi audit và
   xóa readiness cũ tại
   [`src/indexing/frame_paths.py`](../src/indexing/frame_paths.py) và command
   `migrate-frame-paths` trong [`src/cli/main.py`](../src/cli/main.py).

```text
Candidate có score trong embedding artifact
→ hydration không tìm được metadata hoặc path không resolve theo cwd
→ frame_path thiếu / Path.exists() = False
→ candidate bị bỏ không có diagnostic
→ output có thể ít hơn top_k hoặc mất candidate điểm cao
```

#### Cách vận hành legacy run

```bash
codenova migrate-frame-paths --experiment-name EXP --legacy-root /old/project/root
codenova migrate-frame-paths --experiment-name EXP --legacy-root /old/project/root --apply
codenova validate-index --experiment-name EXP
```

Lần đầu chỉ sinh audit dry-run. `--apply` từ chối toàn bộ plan nếu còn record
không resolve được; sau apply bắt buộc validate lại trước khi serve.

#### Test hồi quy

[`tests/unit/test_frame_paths.py`](../tests/unit/test_frame_paths.py) kiểm tra
resolve độc lập CWD, reject path ngoài run/missing file, migration cả aggregate
và partition, backup/audit/readiness invalidation, cùng activation recheck sau
khi file bị xóa. [`tests/unit/test_indexing.py`](../tests/unit/test_indexing.py)
kiểm tra hydration diagnostic cho frame metadata thiếu.

#### Acceptance criteria

- Mọi frame consumer đã sửa dùng quy tắc resolve theo experiment.
- Offline report có count/tỷ lệ và activation fail-fast khi còn lỗi.
- Runtime báo candidate bị loại và nguyên nhân.
- Legacy migration có dry-run, backup, audit và không tự chạy khi serve.

### CN-11 — Đề xuất rà soát reranker candidate limit 50

#### Tóm tắt và mức độ xác nhận

Khi reranker được bật, `Retriever.search()` chỉ rerank tối đa 50 candidate đầu
tiên. Candidate từ vị trí 51 trở đi được nối lại với score/ranking trước
rerank. Con số 50 không nằm trong experiment config, constructor, request hay
log pipeline.

Giới hạn candidate là cần thiết để kiểm soát latency và VRAM; bản thân giá trị
50 chưa được chứng minh là sai và có thể đã được nhóm retrieval chọn qua thực
nghiệm trước đó. CN-11 vì vậy **không được xác định là bug** trong báo cáo này.
Đây chỉ là đề xuất rà soát rationale, provenance và khả năng tái lập nếu các
thông tin benchmark chưa được lưu lại.

#### Bằng chứng source

1. `rerank_limit = min(50, len(valid_hydrated))` được hard-code tại
   [`search.py:121-122`](../src/retrieval/search.py#L121-L122).
2. Chỉ slice `valid_hydrated[:rerank_limit]` được đưa vào reranker; phần còn lại
   được nối nguyên thứ tự tại
   [`search.py:123-126`](../src/retrieval/search.py#L123-L126).
3. `Retriever.__init__()` có `fusion_pool_size` nhưng không có
   `reranker_candidate_k` tại
   [`search.py:41-58`](../src/retrieval/search.py#L41-L58).
4. `PipelineConfig` có final `top_k` nhưng không lưu candidate limit của
   reranker tại [`settings.py:41-64`](../src/config/settings.py#L41-L64).

```text
Fusion/hydration tạo hơn 50 candidate
→ chỉ top 50 được BLIP-2 chấm lại
→ candidate 51+ giữ score/ranking cũ
→ thay đổi 50 cần sửa source và không được ghi trong metadata thí nghiệm
```

Nếu final `top_k <= 50`, giới hạn có thể hoàn toàn phù hợp với mục tiêu chỉ
rerank phần được trả ra. Nếu final `top_k > 50` hoặc muốn một candidate ngoài
top 50 có cơ hội được reranker kéo lên, giới hạn này mới ảnh hưởng search
space. Không nên thay đổi con số 50 khi chưa kiểm tra tài liệu hoặc benchmark
đã có của nhóm retrieval.

#### Định hướng đề xuất — cần xác nhận trước khi thay đổi

1. Trước tiên xác nhận với nhóm retrieval liệu 50 đã được benchmark và chốt cho
   cấu hình thi đấu hay chưa. Nếu có, giữ nguyên giá trị và bổ sung rationale
   hoặc kết quả benchmark vào tài liệu.
2. Chỉ khi cần chạy nhiều profile/thí nghiệm mới cân nhắc đặt tên rõ
   `reranker_candidate_k` và đưa nó vào config; default nên giữ 50 để không thay
   đổi hành vi hiện tại.
3. Ghi model reranker, candidate limit, số candidate thực tế và latency vào log
   hoặc metadata benchmark để kết quả có thể tái lập.
4. Nếu chưa có bằng chứng cho lựa chọn 50, có thể benchmark các mức như
   20/50/100 theo recall/ranking quality, latency và VRAM. Kết quả benchmark,
   không phải việc con số đang hard-code, mới là căn cứ quyết định thay đổi.
5. Không xem việc chuyển 50 thành config là yêu cầu bắt buộc để đóng một bug;
   đây là cải tiến maintainability/experiment control nếu nhóm thấy cần.

#### Test cần viết

1. Characterization test: inject recording reranker với 100 candidate và xác
   nhận hiện trạng nhận 50; test này ghi lại behavior, không chứng minh behavior
   sai.
2. Nếu candidate limit được cấu hình hóa, test các mức 20/50/100, candidate ít
   hơn limit, giá trị không hợp lệ và reranker bị tắt.
3. Test final `top_k > candidate limit` để làm rõ cách nối phần chưa rerank nếu
   hệ thống thực sự hỗ trợ trường hợp này.

#### Tiêu chí để kết thúc việc rà soát

- Có căn cứ hoặc benchmark giải thích giá trị 50, hoặc có quyết định rõ rằng
  giá trị này chỉ là default tạm thời.
- Giá trị dùng khi thi và số candidate thực sự được rerank có thể truy vết.
- Nếu giữ hard-code 50 sau khi xác nhận, CN-11 vẫn có thể được đóng dưới dạng
  `reviewed / no change required`.
- Nếu cấu hình hóa, default 50 được giữ và có validation/test phù hợp.

### CN-12 — Caption không được render trên result card

#### Tóm tắt và mức độ xác nhận

Backend đã hỗ trợ caption được tạo trong offline indexing và gắn caption vào
`SearchResult`. Endpoint `/api/search` serialize trường này, nhưng result card
của UI không đọc `result.caption`. Vì vậy caption có thể có trong response mà
người dùng không nhìn thấy trên giao diện.

Issue này **đã chứng minh bằng source**. Nó là khoảng trống hiển thị/UX, không
làm thay đổi score retrieval; ảnh hưởng chỉ xuất hiện với experiment có caption
và endpoint trả trường caption.

#### Bằng chứng source

1. `SearchResult` khai báo `caption` và `to_dict()` serialize toàn bộ dataclass
   tại [`core/types.py:89-105`](../src/core/types.py#L89-L105).
2. `ResultHydrator` load `CaptionRepository` và gắn caption theo `frame_id` tại
   [`hydrator.py:21-46`](../src/retrieval/hydrator.py#L21-L46).
3. `/api/search` dùng `result_to_payload()` tại
   [`server.py:516-522`](../src/ui/server.py#L516-L522); hàm này gọi
   `result.to_dict()`, nên caption đi vào payload tại
   [`server.py:650-655`](../src/ui/server.py#L650-L655).
4. UI chỉ chọn snippet từ `text`, `matched_text`, `ocr_text` hoặc `asr_text` tại
   [`server.py:1348-1352`](../src/ui/server.py#L1348-L1352), không có
   `result.caption`.
5. Result card render `${textHtml}` nhưng không có vùng caption riêng tại
   [`server.py:1363-1377`](../src/ui/server.py#L1363-L1377).

```text
Offline caption tồn tại
→ hydrator gắn SearchResult.caption
→ /api/search trả caption trong JSON
→ renderResults không đọc caption
→ caption không xuất hiện trên card
```

#### Định hướng khắc phục

1. Hiển thị `result.caption` trong một vùng riêng có label `Caption`, không trộn
   với OCR/ASR snippet vì các trường có provenance và ý nghĩa khác nhau.
2. Tiếp tục dùng `escapeHtml()` và giới hạn chiều dài/expand-collapse để caption
   dài không phá layout hoặc tạo XSS.
3. Không hiển thị block caption khi giá trị `null`, rỗng hoặc experiment không
   chạy captioning.
4. Rà contract của từng endpoint: endpoint nào cam kết trả `SearchResult` có
   caption thì serialize nhất quán; endpoint không hydrate caption cần được ghi
   rõ thay vì để UI suy đoán.
5. Có thể thêm toggle hiển thị caption nếu card quá dày, nhưng caption không nên
   bị sử dụng như OCR/ASR match evidence nếu nó không tham gia score của query.

#### Test cần viết

1. Payload chỉ có `caption`: card phải hiển thị caption và label đúng.
2. Caption chứa HTML/script: output phải được escape.
3. Caption `null`/rỗng: không render block trống.
4. Payload có cả caption và OCR/ASR text: hai provenance phải được hiển thị tách
   biệt.
5. API serialization test xác nhận `/api/search` giữ trường caption từ
   `SearchResult`.

#### Acceptance criteria

- Caption có trong response được hiển thị có label và escape an toàn.
- Caption và OCR/ASR snippet không bị trộn provenance.
- Experiment không có caption không xuất hiện UI block rỗng.
- Contract caption của các endpoint được mô tả và test nhất quán.

## 4. Thứ tự xử lý đề xuất

1. ~~CN-01 — khóa persisted config của experiment.~~ **DONE 2026-08-14.**
2. ~~CN-03 — bảo đảm WSF align đúng frame ID.~~ **DONE 2026-08-14.**
3. CN-02 và CN-04 — đồng bộ model selection giữa UI và backend.
4. ~~CN-05 — ngăn metadata suy đoán làm sai temporal retrieval.~~ **DONE 2026-08-14.**
5. ~~CN-06 và CN-09 — startup/model health fail-fast.~~ **DONE 2026-08-14.**
6. CN-07, CN-08, ~~CN-10~~ và CN-12 — request safety, ổn định và UX
   (**CN-10 DONE 2026-08-14**).
7. CN-11 — đề xuất rà soát sau khi xác nhận benchmark/rationale hiện có.

Workflow cho từng issue:

```text
viết characterization test
→ chạy và lưu output
→ cập nhật trạng thái “đã tái hiện”
→ chốt expected behavior với team
→ viết regression test đang fail
→ sửa code
→ chạy unit + integration test
→ cập nhật acceptance criteria và trạng thái resolved
```

## 5. Mẫu thêm issue tiếp theo

Mọi issue CN-02 trở đi nên dùng cấu trúc:

```text
## CN-XX — Tiêu đề

### Tóm tắt
### Branch/commit đã kiểm tra
### Mức độ xác nhận
### Bằng chứng source
### Test tái hiện và output
### Chuỗi nguyên nhân
### Ảnh hưởng đã chứng minh
### Hậu quả có điều kiện/chưa xác minh
### Định hướng sửa
### Test hồi quy
### Acceptance criteria
### Quyết định cần team chốt
```

Bằng chứng filesystem/runtime phải kèm command và output. Bằng chứng source
phải kèm branch/commit, file và line hoặc content hash. Khi source thay đổi,
line anchor phải được rà lại trước khi gửi báo cáo.
