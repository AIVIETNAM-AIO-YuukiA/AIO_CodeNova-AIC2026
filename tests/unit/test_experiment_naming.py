from datetime import UTC, datetime

import pytest

from codenova.config.settings import PipelineConfig, validate_experiment_name
from codenova.core.errors import ExperimentNameError


def test_default_experiment_name_is_valid_and_stable() -> None:
    config = PipelineConfig(
        clip_model="ViT-B/32",
        frame_sampling="Shot Midpoint",
        index_backend="Qdrant",
    )

    name = config.default_experiment_name(datetime(2026, 6, 12, tzinfo=UTC))

    assert name.startswith("20260612_retrieval_vit-b-32_shot-midpoint_qdrant_")
    assert validate_experiment_name(name) == name


@pytest.mark.parametrize("name", ["Bad Name", "../bad", "ab", "bad/name", "bad.name"])
def test_invalid_experiment_names_are_rejected(name: str) -> None:
    with pytest.raises(ExperimentNameError):
        validate_experiment_name(name)
