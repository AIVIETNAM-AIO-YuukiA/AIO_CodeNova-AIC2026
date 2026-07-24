"""Frame/video captioning modules."""

from modules.captioning.base import Caption, CaptioningModel
from modules.captioning.vllm import VllmCaptioningModel

__all__ = ["Caption", "CaptioningModel", "VllmCaptioningModel"]
