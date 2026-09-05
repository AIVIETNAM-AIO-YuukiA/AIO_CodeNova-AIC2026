"""Project-specific exceptions."""


class CodeNovaError(Exception):
    """Base class for application errors."""


class ExperimentNameError(CodeNovaError):
    """Raised when an experiment name is invalid or conflicts with an existing run."""


class ExperimentConfigError(CodeNovaError):
    """Raised when persisted experiment metadata is invalid or mismatched."""


class VideoReadError(CodeNovaError):
    """Raised when video metadata or frames cannot be read."""


class ShotDetectionError(CodeNovaError):
    """Raised when shot detection fails."""


class FrameExtractionError(CodeNovaError):
    """Raised when frame extraction fails."""


class FramePathError(CodeNovaError):
    """Raised when a persisted frame path cannot be resolved safely."""


class EmbeddingError(CodeNovaError):
    """Raised when embedding generation fails."""


class CaptioningError(CodeNovaError):
    """Raised when VLM caption generation fails."""


class IndexBuildError(CodeNovaError):
    """Raised when index creation or loading fails."""


class RetrievalError(CodeNovaError):
    """Raised when query retrieval fails."""


class FusionError(RetrievalError):
    """Raised when model results cannot be aligned safely for fusion."""
