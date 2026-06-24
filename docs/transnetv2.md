# Preparing TransNetV2

Shot detection uses TransNetV2 (PyTorch). The model code and weights are not bundled;
clone them into `external/` and convert the weights to PyTorch once.

## 1. Clone

```bash
mkdir -p external
git clone https://github.com/soCzech/TransNetV2 external/TransNetV2
```

## 2. Pull the real weights (Git LFS)

The weights ship via Git LFS. Without pulling them you get a `Wire format was corrupt`
error during conversion (the file is only an LFS pointer).

```bash
sudo apt install git-lfs
git lfs install
git -C external/TransNetV2 lfs pull
```

Check the weights are real files (tens of MB, not a few hundred bytes):

```bash
ls -lh external/TransNetV2/inference/transnetv2-weights/saved_model.pb
ls -lh external/TransNetV2/inference/transnetv2-weights/variables/
```

## 3. Convert TensorFlow weights to PyTorch

Run the project's converter with a temporary Python 3.10 environment via uv:

```bash
cd external/TransNetV2/inference-pytorch
uv run --no-project --python 3.10 --with torch --with tensorflow --with numpy \
  python convert_weights.py --tf_weights ../inference/transnetv2-weights
cd -
```

Result:

```
external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
```

TensorFlow `Could not find cuda drivers` warnings during conversion are harmless — the
conversion runs on CPU. The main pipeline uses PyTorch + CUDA afterwards.

## 4. Use it

```bash
make detect-shots EXP=demo
# which runs, with defaults TN2_DIR / TN2_WEIGHTS:
#   codenova detect-shots --experiment-name demo \
#     --transnetv2-module-dir external/TransNetV2/inference-pytorch \
#     --transnetv2-weights    external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
```

If you hit CUDA OOM, prefix with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
