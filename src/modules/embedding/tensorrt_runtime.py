"""Shared TensorRT export/build/inference for vision encoders.

Exports a model's vision tower to ONNX, builds a TensorRT FP16 engine, and
caches it under ``weights/<model>/``. First call builds; later calls just
load the cached engine. Text encoding stays on PyTorch — queries are embedded
one at a time, so there's no batching gain to capture.
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
    """Builds (once) and runs a TensorRT FP16 engine for a vision encoder.

    ``export_onnx_fn(onnx_path)`` traces the model with ``torch.onnx.export``
    and writes it to ``onnx_path`` — supplied by the caller since tracing a
    dummy input and picking the output tensor name differs per model.
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
        """Export + build if not already cached; a no-op otherwise."""
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
        """Run the vision encoder on a batch of preprocessed CUDA pixel tensors.

        ``pixel_values``: contiguous ``(batch, 3, H, W)`` float32. Returns a
        ``(batch, output_dim)`` float32 CUDA tensor.
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

        # Rebinding tensor addresses each call is measurable overhead at this
        # scale; reuse the buffers when the batch size repeats (the common
        # case — every call is full-size except the last, partial one).
        if self._buffers is None or self._buffers_batch != batch:
            context.set_input_shape(self.input_name, tuple(pixel_values.shape))
            output = torch.empty(
                (batch, self.output_dim), dtype=torch.float32, device=pixel_values.device
            )
            scratch_tensors = {}
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                if name in (self.input_name, self.output_name):
                    continue
                shape = tuple(context.get_tensor_shape(name))
                scratch_tensors[name] = torch.empty(
                    shape, dtype=torch.float32, device=pixel_values.device
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


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_name: str,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
) -> None:
    """Build a dynamic-batch FP16 TensorRT engine from an ONNX file."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise EmbeddingError("Install tensorrt before building a TensorRT engine.") from exc

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # parse_from_file, not parse(bytes): resolves external-data files (e.g.
    # "vision.onnx.data") relative to the ONNX file, not the cwd.
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise EmbeddingError(f"Failed to parse ONNX model {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    chw = tuple(network.get_input(0).shape[1:])
    profile.set_shape(input_name, (min_batch, *chw), (opt_batch, *chw), (max_batch, *chw))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise EmbeddingError(f"TensorRT engine build failed for {onnx_path}")

    # Temp file + rename so a killed build never leaves a corrupt engine that
    # looks present on retry.
    tmp_path = engine_path.with_suffix(engine_path.suffix + ".tmp")
    tmp_path.write_bytes(serialized)
    tmp_path.replace(engine_path)
