"""Project-specific exceptions."""


class CodeNovaError(Exception):
    """Base class for application errors."""


class ExperimentNameError(CodeNovaError):
    """Raised when an experiment name is invalid or conflicts with an existing run."""


class VideoReadError(CodeNovaError):
    """Raised when video metadata or frames cannot be read."""


class ShotDetectionError(CodeNovaError):
    """Raised when shot detection fails."""


class FrameExtractionError(CodeNovaError):
    """Raised when frame extraction fails."""


class EmbeddingError(CodeNovaError):
    """Raised when embedding generation fails."""


class CaptioningError(CodeNovaError):
    """Raised when VLM caption generation fails."""


class IndexBuildError(CodeNovaError):
    """Raised when index creation or loading fails."""


class RetrievalError(CodeNovaError):
    """Raised when query retrieval fails."""
