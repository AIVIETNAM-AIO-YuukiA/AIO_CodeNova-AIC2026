# Chuẩn bị TransNetV2

> Bản tiếng Anh: [../transnetv2.md](../transnetv2.md)

Phát hiện shot dùng TransNetV2 (PyTorch). Code model và weights không kèm trong repo;
clone vào `external/` và convert weights sang PyTorch một lần.

## 1. Clone

```bash
mkdir -p external
git clone https://github.com/soCzech/TransNetV2 external/TransNetV2
```

## 2. Pull weights thật (Git LFS)

Weights dùng Git LFS. Nếu không pull, lúc convert sẽ lỗi `Wire format was corrupt` (file
chỉ là LFS pointer).

```bash
sudo apt install git-lfs
git lfs install
git -C external/TransNetV2 lfs pull
```

Kiểm tra weights là file thật (vài chục MB, không phải vài trăm byte):

```bash
ls -lh external/TransNetV2/inference/transnetv2-weights/saved_model.pb
ls -lh external/TransNetV2/inference/transnetv2-weights/variables/
```

## 3. Convert weights TensorFlow → PyTorch

Chạy script convert bằng Python 3.10 tạm thời qua uv:

```bash
cd external/TransNetV2/inference-pytorch
uv run --no-project --python 3.10 --with torch --with tensorflow --with numpy \
  python convert_weights.py --tf_weights ../inference/transnetv2-weights
cd -
```

Kết quả:

```
external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
```

Cảnh báo `Could not find cuda drivers` của TensorFlow khi convert là vô hại — bước convert
chạy CPU. Pipeline chính dùng PyTorch + CUDA sau đó.

## 4. Sử dụng

```bash
make detect-shots EXP=demo
```

Nếu gặp CUDA OOM, thêm tiền tố `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
