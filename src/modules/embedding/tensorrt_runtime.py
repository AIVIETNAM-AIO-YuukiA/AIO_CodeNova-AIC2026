"""Export/build/chay inference TensorRT dung chung cho cac vision encoder.

Export vision tower cua model ra ONNX, build engine TensorRT FP16, roi
cache vao ``weights/<model>/``. Lan goi dau tien se build; cac lan sau chi
nap engine da cache. Text encoding van chay PyTorch - query duoc embed
tung cai mot nen khong co loi ich gi tu batching.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from core.errors import EmbeddingError

LOGGER = logging.getLogger(__name__)

_DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights"
_DEFAULT_MIN_BATCH = 1
_DEFAULT_OPT_BATCH = 32
_DEFAULT_MAX_BATCH = 64


def weights_dir(model_key: str) -> Path:
    root = Path(os.environ.get("TENSORRT_WEIGHTS_DIR", _DEFAULT_WEIGHTS_DIR))
    return root / model_key


class TensorRTVisionEncoder:
    """Build (1 lan) va chay engine TensorRT FP16 cho 1 vision encoder.

    ``export_onnx_fn(onnx_path)`` trace model bang ``torch.onnx.export`` roi
    ghi ra ``onnx_path`` - do noi goi truyen vao vi cach trace dummy input
    va chon ten output tensor khac nhau tuy model.
    """

    def __init__(
        self,
        model_key: str,
        export_onnx_fn,
        input_name: str,
        output_name: str,
        output_dim: int,
        min_batch: int = _DEFAULT_MIN_BATCH,
        opt_batch: int = _DEFAULT_OPT_BATCH,
        max_batch: int = _DEFAULT_MAX_BATCH,
    ) -> None:
        self.model_key = model_key
        self.export_onnx_fn = export_onnx_fn
        self.input_name = input_name
        self.output_name = output_name
        self.output_dim = output_dim
        self.min_batch = min_batch
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        self._engine = None
        self._context = None
        self._stream = None
        self._buffers: dict | None = None
        self._buffers_batch: int | None = None

    def _paths(self) -> tuple[Path, Path]:
        directory = weights_dir(self.model_key)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "vision.onnx", directory / "vision_fp16.engine"

    def ensure_built(self) -> Path:
        """Export + build neu chua cache; da co san thi khong lam gi."""
        onnx_path, engine_path = self._paths()
        if engine_path.exists():
            LOGGER.info("[%s] Using cached TensorRT engine: %s", self.model_key, engine_path)
            return engine_path

        if not onnx_path.exists():
            LOGGER.info("[%s] Exporting to ONNX: %s", self.model_key, onnx_path)
            self.export_onnx_fn(onnx_path)

        LOGGER.info("[%s] Building TensorRT FP16 engine (one-time)...", self.model_key)
        _build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            input_name=self.input_name,
            min_batch=self.min_batch,
            opt_batch=self.opt_batch,
            max_batch=self.max_batch,
        )
        LOGGER.info("[%s] TensorRT engine built: %s", self.model_key, engine_path)
        return engine_path

    def _load(self):
        if self._context is not None:
            return self._context

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise EmbeddingError(
                "Install tensorrt before using the TensorRT vision encoder."
            ) from exc

        engine_path = self.ensure_built()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as handle:
            self._engine = runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise EmbeddingError(
                f"Failed to load TensorRT engine at {engine_path} — likely built by an "
                "incompatible TensorRT version. Delete the file to force a rebuild."
            )
        self._context = self._engine.create_execution_context()
        return self._context

    def infer(self, pixel_values):
        """Chay vision encoder tren 1 batch pixel tensor CUDA da preprocess.

        ``pixel_values``: tensor float32 lien tuc dang ``(batch, 3, H, W)``.
        Tra ve tensor CUDA dang ``(batch, output_dim)``, dtype theo dung
        dtype ma engine khai bao (fp16 hoac fp32 tuy model).
        """
        import torch

        context = self._load()
        pixel_values = pixel_values.contiguous()
        batch = pixel_values.shape[0]
        if batch > self.max_batch:
            raise EmbeddingError(
                f"[{self.model_key}] Batch size {batch} exceeds the engine's max batch "
                f"{self.max_batch}. Lower --batch-size or rebuild with a larger max_batch."
            )

        if self._stream is None:
            self._stream = torch.cuda.Stream()

        # Gan lai dia chi tensor moi lan goi ton overhead do duoc o quy mo
        # nay; tai su dung buffer khi batch size lap lai (truong hop pho
        # bien - moi lan goi deu full-size, tru lan cuoi con du).
        if self._buffers is None or self._buffers_batch != batch:
            context.set_input_shape(self.input_name, tuple(pixel_values.shape))
            # Buffer phai dung dung dtype engine khai bao: engine ghi fp16 vao
            # buffer fp32 (hoac nguoc lai) se doc ra so rac hoan toan.
            output = torch.empty(
                (batch, self.output_dim),
                dtype=self._torch_dtype(torch, self.output_name),
                device=pixel_values.device,
            )
            scratch_tensors = {}
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                if name in (self.input_name, self.output_name):
                    continue
                shape = tuple(context.get_tensor_shape(name))
                scratch_tensors[name] = torch.empty(
                    shape, dtype=self._torch_dtype(torch, name), device=pixel_values.device
                )
                context.set_tensor_address(name, scratch_tensors[name].data_ptr())
            context.set_tensor_address(self.output_name, output.data_ptr())
            self._buffers = {"output": output, "scratch": scratch_tensors}
            self._buffers_batch = batch
        else:
            output = self._buffers["output"]

        context.set_tensor_address(self.input_name, pixel_values.data_ptr())
        context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()
        return output

    def _torch_dtype(self, torch, tensor_name: str):
        """Doi dtype TensorRT cua 1 tensor sang dtype torch tuong ung."""
        import tensorrt as trt

        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        trt_dtype = self._engine.get_tensor_dtype(tensor_name)
        if trt_dtype not in mapping:
            raise EmbeddingError(
                f"[{self.model_key}] Chua ho tro dtype TensorRT {trt_dtype} "
                f"cho tensor '{tensor_name}'."
            )
        return mapping[trt_dtype]


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_name: str,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
) -> None:
    """Build engine TensorRT FP16 ho tro dynamic-batch tu 1 file ONNX."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise EmbeddingError("Install tensorrt before building a TensorRT engine.") from exc

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 10+ chi con che do explicit batch, bo EXPLICIT_BATCH;
    # TensorRT 11 thay tiep BuilderFlag.FP16 bang strongly-typed network,
    # do chinh xac lay tu ONNX graph. Ho tro ca 2 phien ban.
    flags = 0
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit_batch is not None:
        flags |= 1 << int(explicit_batch)
    strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if not hasattr(trt.BuilderFlag, "FP16") and strongly_typed is not None:
        flags |= 1 << int(strongly_typed)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    # Dung parse_from_file, khong dung parse(bytes): tim file external-data
    # (vd "vision.onnx.data") tuong doi theo vi tri file ONNX, khong phai cwd.
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise EmbeddingError(f"Failed to parse ONNX model {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    chw = tuple(network.get_input(0).shape[1:])
    profile.set_shape(input_name, (min_batch, *chw), (opt_batch, *chw), (max_batch, *chw))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise EmbeddingError(f"TensorRT engine build failed for {onnx_path}")

    # Ghi ra file tam roi doi ten, de neu build bi ngat giua chung se khong
    # de lai engine hong ma lan sau tuong da co san.
    tmp_path = engine_path.with_suffix(engine_path.suffix + ".tmp")
    tmp_path.write_bytes(serialized)
    tmp_path.replace(engine_path)
