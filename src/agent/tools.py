"""Tool interface and implementations for VLM Agent."""

from __future__ import annotations

from pathlib import Path
from abc import ABC, abstractmethod
import logging
import os

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_DOTENV_LOADED = False


def _ensure_dotenv():
    global _DOTENV_LOADED
    if _DOTENV_LOADED or os.environ.get("GEMINI_API_KEY"):
        _DOTENV_LOADED = True
        return
    env_path = _ROOT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                if key.strip() == "GEMINI_API_KEY":
                    os.environ["GEMINI_API_KEY"] = val.strip()
                    break
    _DOTENV_LOADED = True


LOGGER = logging.getLogger(__name__)


class Tool(ABC):
    """Base class for all Agent tools."""

    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return result as string."""


class CaptionTool(Tool):
    """Describe an image using Gemini Vision API."""

    name = "caption"
    description = (
        "Describe an image in detail. Input: image_path. "
        "Output: a detailed English description of what is visible."
    )

    def __init__(self, model_name: str = "gemini-2.5-flash-lite") -> None:
        self.model_name = model_name
        self._client = None

    def _init_client(self):
        if self._client is not None:
            return True
        _ensure_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            LOGGER.warning("GEMINI_API_KEY not set; caption tool will return placeholder.")
            return False
        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            return True
        except Exception as exc:
            LOGGER.exception("Failed to init Gemini client: %s", exc)
            return False

    def run(self, image_path: str = "", prompt: str = "") -> str:
        if not image_path:
            return "Error: image_path is required."
        if not self._init_client():
            return f"[Caption unavailable] Image: {image_path}"

        # If no prompt provided, use a detailed default
        if not prompt:
            prompt = (
                "Describe this image in detail in Vietnamese. "
                "Focus on: objects, colors, text, people, actions, positions. "
                "Be specific about colors (xanh dương, đỏ, trắng, vàng, v.v.) "
                "and any text you see."
            )

        try:
            from PIL import Image

            img = Image.open(image_path)
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[prompt, img],
            )
            return response.text.strip()
        except Exception as exc:
            LOGGER.exception("Caption tool failed: %s", exc)
            return f"[Caption error: {exc}]"


class OCRTool(Tool):
    """Read text from an image using Gemini Vision API."""

    name = "ocr"
    description = (
        "Read and extract text visible in an image. Input: image_path. "
        "Output: all text found in the image."
    )

    def __init__(self, model_name: str = "gemini-2.5-flash-lite") -> None:
        self.model_name = model_name
        self._client = None

    def _init_client(self):
        if self._client is not None:
            return True
        _ensure_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            LOGGER.warning("GEMINI_API_KEY not set; ocr tool will return placeholder.")
            return False
        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            return True
        except Exception as exc:
            LOGGER.exception("Failed to init Gemini client: %s", exc)
            return False

    def run(self, image_path: str = "") -> str:
        if not image_path:
            return "Error: image_path is required."
        if not self._init_client():
            return f"[OCR unavailable] Image: {image_path}"

        try:
            from PIL import Image

            img = Image.open(image_path)
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[
                    "Extract all visible text from this image. Return only the text found.",
                    img,
                ],
            )
            return response.text.strip()
        except Exception as exc:
            LOGGER.exception("OCR tool failed: %s", exc)
            return f"[OCR error: {exc}]"


class DetectTool(Tool):
    """Detect objects in an image using YOLOv8."""

    name = "detect"
    description = (
        "Detect objects in an image and return bounding boxes. "
        "Input: image_path. Output: list of detected objects with labels and coordinates."
    )

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return True
        try:
            from ultralytics import YOLO

            self._model = YOLO("yolov8n.pt")
            return True
        except ImportError:
            LOGGER.warning("ultralytics not installed. Install with: uv add ultralytics")
            return False
        except Exception as exc:
            LOGGER.exception("Failed to load YOLO model: %s", exc)
            return False

    def run(self, image_path: str = "") -> str:
        if not image_path:
            return "Error: image_path is required."
        if not self._load_model():
            return "[Detection unavailable: install ultralytics]"

        try:
            results = self._model(image_path, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = r.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    detections.append(f"{label} ({conf:.2f}) [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            return "\n".join(detections) if detections else "No objects detected."
        except Exception as exc:
            LOGGER.exception("Detection tool failed: %s", exc)
            return f"[Detection error: {exc}]"


class ASRTool(Tool):
    """Transcribe speech from audio using Whisper."""

    name = "asr"
    description = (
        "Transcribe speech from an audio file. " "Input: audio_path. Output: transcribed text."
    )

    def __init__(self, model_size: str = "small") -> None:
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return True
        try:
            import whisper

            self._model = whisper.load_model(self.model_size)
            return True
        except ImportError:
            LOGGER.warning("openai-whisper not installed.")
            return False
        except Exception as exc:
            LOGGER.exception("Failed to load Whisper model: %s", exc)
            return False

    def run(self, audio_path: str = "") -> str:
        if not audio_path:
            return "Error: audio_path is required."
        if not self._load_model():
            return "[ASR unavailable: install openai-whisper]"

        try:
            result = self._model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as exc:
            LOGGER.exception("ASR tool failed: %s", exc)
            return f"[ASR error: {exc}]"


def default_tools() -> dict[str, Tool]:
    """Return a dict of all available tools."""
    return {
        "caption": CaptionTool(),
        "ocr": OCRTool(),
        "detect": DetectTool(),
        "asr": ASRTool(),
    }
