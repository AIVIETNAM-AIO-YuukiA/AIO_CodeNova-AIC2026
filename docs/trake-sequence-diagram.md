# TRAKE Pipeline — Sequential Windowed Search Diagram

```mermaid
flowchart TD
    subgraph Input["INPUT"]
        E["Events: [E₁, E₂, ..., Eₙ]"]
    end

    subgraph Step1["BƯỚC 1: GLOBAL SEARCH"]
        S1["Search E₁ → Qdrant top 200"]
        S2["Search E₂ → Qdrant top 200"]
        S3["Search Eₙ → Qdrant top 200"]
        G1["Gom theo video_id"]
    end

    subgraph Step2["BƯỚC 2: BUILD CHAINS"]
        V["Với mỗi video có trong E₁"]
        E1["Duyệt E₁ candidates (top 20 theo score)"]
        W["Cửa sổ 300s"]
        F2["Lọc E₂: cùng video + ts ∈ [t₁, t₁+300]"]
        F3["Lọc E₃: cùng video + ts ∈ [t₂, t₂+300]"]
        Fn["Lọc Eₙ ..."]
        CK{"Có đủ N events?"}
        SC["Tính Final Score"]
    end

    subgraph Step3["BƯỚC 3: SORT & OUTPUT"]
        SORT["Sort chains theo score giảm dần"]
        OUT["Return JSON cho UI"]
    end

    E --> S1
    E --> S2
    E --> S3
    S1 --> G1
    S2 --> G1
    S3 --> G1
    G1 --> V
    V --> E1
    E1 --> W
    W --> F2
    F2 --> F3
    F3 --> Fn
    Fn --> CK
    CK -- "Yes" --> SC
    CK -- "No → discard chain" --> E1
    SC --> SORT
    SORT --> OUT
```

## Key Parameters

| Parameter | Value | Ý nghĩa |
|-----------|-------|---------|
| `top_k` | 200 | Số frame lấy mỗi event |
| `WINDOW_SIZE` | 300s (5 phút) | Cửa sổ thời gian giữa 2 events |
| `λ` (lambda) | 0.5 | Hệ số phạt temporal |
| Max E₁ candidates | 20 | Số E₁ frame thử mỗi video |

## Scoring

```
Final_Score = (1/N × Σ Sim(Qᵢ, Fᵢ)) × exp(-0.5 × (tₙ - t₁) / 300)
```

- **Sim(Qᵢ, Fᵢ)** = vector similarity (Qdrant score) giữa event text và frame
- **exp(-0.5 × duration / 300)** = temporal penalty factor
- Events xảy ra càng sát nhau → factor càng gần 1 → score càng cao

## So sánh OLD vs NEW

| | OLD (Independent + Intersection) | NEW (Sequential Windowed) |
|---|---|---|
| Filter | Hard: video phải có trong ALL events | Soft: mỗi bước lọc theo window |
| Temporal | Boolean (OK / not OK) × 2.0 | Continuous penalty exp(-λ×d/W) |
| Scoring | Σ e^(-0.02×rank) | Mean_similarity × temporal_factor |
| Retriever | Build N lần (1/event) | Build 1 lần, reuse |

## Lưu ý

- **3 events → 3 lần search Qdrant × 200 = 600 frames** (không search lại trong vòng lặp)
- **Không có intersection cứng** — video chỉ cần có E₁ trong top 200 là được xét
- **Nếu E₂ không có frame nào trong window 300s → discard chain đó**
- **Không trùng frame_id giữa các events trong cùng chain** — mỗi candidate bị loại nếu `frame_id` đã có trong chain trước đó.
- **Timestamp strict** (`<` thay vì `<=`) — Eᵢ₊₁ phải có timestamp **lớn hơn hẳn** Eᵢ, không được bằng.
