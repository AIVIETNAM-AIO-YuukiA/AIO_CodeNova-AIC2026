# TRAKE Pipeline — Bidirectional Pair Join (BPJ) Diagram

```mermaid
flowchart TD
    subgraph Input["INPUT"]
        E["Events: [E₁, E₂, ..., Eₙ]"]
    end

    subgraph Step1["BƯỚC 1: GLOBAL SEARCH"]
        S1["Search E₁ → Qdrant top 300"]
        S2["Search E₂ → Qdrant top 300"]
        S3["Search Eₙ → Qdrant top 300"]
        VEC["Store vectors + metadata in-memory"]
    end

    subgraph Step2["BƯỚC 2: BIDIRECTIONAL PAIR JOIN"]
        direction LR
        subgraph Pair_AB["Pair (E₁, E₂)"]
            PAB_F["Forward: E₁→E₂<br/>cosine sim in-memory<br/>+5min window<br/>top 30 per candidate"]
            PAB_B["Backward: E₂←E₁<br/>cosine sim in-memory<br/>-5min window<br/>top 30 per candidate"]
            PAB_M["Merge + dedup<br/>top 300 pairs"]
        end
        subgraph Pair_BC["Pair (E₂, E₃)"]
            PBC_F["Forward: E₂→E₃<br/>cosine sim in-memory<br/>+5min window<br/>top 30 per candidate"]
            PBC_B["Backward: E₃←E₂<br/>cosine sim in-memory<br/>-5min window<br/>top 30 per candidate"]
            PBC_M["Merge + dedup<br/>top 300 pairs"]
        end
    end

    subgraph Step3["BƯỚC 3: JOIN & SCORE"]
        JOIN["Join pairs on common frame_id<br/>E₂ must match across both pairs"]
        SC["Chain Score = mean(sim)"]
    end

    subgraph Step4["BƯỚC 4: SORT & OUTPUT"]
        SORT["Sort chains theo score giảm dần"]
        OUT["Return JSON cho UI"]
    end

    E --> S1
    E --> S2
    E --> S3
    S1 --> VEC
    S2 --> VEC
    S3 --> VEC
    VEC --> PAB_F
    VEC --> PAB_B
    VEC --> PBC_F
    VEC --> PBC_B
    PAB_F --> PAB_M
    PAB_B --> PAB_M
    PBC_F --> PBC_B
    PBC_F --> PBC_M
    PBC_B --> PBC_M
    PAB_M --> JOIN
    PBC_M --> JOIN
    JOIN --> SC
    SC --> SORT
    SORT --> OUT
```

## Key Parameters

| Parameter | Value | Ý nghĩa |
|-----------|-------|---------|
| `top_k` | 300 | Số frame lấy mỗi event từ Qdrant |
| `WINDOW_SIZE` | 300s (5 phút) | Cửa sổ thời gian cho pair matching |
| Window search top-K | 30 | Số best match giữ lại mỗi candidate |
| Max pairs per adjacent pair | 300 | Số cặp giữ lại sau merge mỗi cặp |

## Scoring

### Pair Score (cho mỗi cặp frame trong forward/backward)

```
Pair_Score = Sim(fᵢ, eᵢ) + Sim(fᵢ₊₁, eᵢ₊₁)
```

- **Sim(fᵢ, eᵢ)** = vector similarity (Qdrant score) giữa event text eᵢ và frame fᵢ
- Không temporal penalty — score thuần semantic

### Chain Score (cho chain hoàn chỉnh)

```
Chain_Score = mean(Sim(f₁, e₁), Sim(f₂, e₂), ..., Sim(fₙ, eₙ))
```

- **mean(sim)** = trung bình similarity scores của tất cả frame
- Không temporal penalty — chỉ dựa trên độ khớp semantic

## So sánh OLD vs BPJ

| | OLD (Sequential Windowed) | NEW (Bidirectional Pair Join) |
|---|---|---|
| Strategy | Greedy-forward từ E₁ | Bidirectional, mỗi cặp độc lập |
| E₂ matching | Qdrant score (global) | In-memory cosine similarity với query E₂ |
| Search chiều | 1 chiều (forward) | 2 chiều (forward + backward) |
| Dedup | Tránh trùng frame_id trong chain | Dedup pairs theo (frame_idᵢ, frame_idᵢ₊₁) |
| Temporal penalty | Có (tuyến tính) | Không |
| Nhạy cảm với E₁ bad frame | Rất — nếu E₁ sai, chain hỏng | Ít — backward pass có thể cứu |

## Lưu ý

- **N events → N lần search Qdrant × 300 frames**
- **Forward + Backward cho mỗi adjacent pair → 2(N-1) passes in-memory**
- **In-memory cosine similarity** dùng preloaded frame embeddings và query vector — không search lại Qdrant
- **Window search top-30** mỗi candidate → ≤9000 pairs mỗi chiều → merge → top 300
- **Join pairs** bằng cách matching frame_id của event ở giữa
- **Không temporal penalty** — score chỉ dựa trên độ khớp semantic thuần
