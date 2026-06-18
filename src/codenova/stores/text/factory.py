"""Build a text index from experiment configuration and environment."""

from __future__ import annotations

import os

from codenova.config.settings import Experiment
from codenova.stores.text.base import TextIndex
from codenova.stores.text.elasticsearch import ElasticTextIndex


def build_text_index(experiment: Experiment) -> TextIndex:
    """Create the Elasticsearch text index for an experiment.

    Connection details come from the environment (loaded from ``.env`` at
    startup). The per-experiment index name is namespaced with the experiment
    name so separate runs never collide in a shared instance.
    """
    base_index = os.environ.get("ELASTIC_INDEX", "codenova_text")
    return ElasticTextIndex(
        url=os.environ.get("ELASTIC_URL", "http://localhost:9200"),
        index_name=f"{base_index}__{experiment.name}",
        api_key=os.environ.get("ELASTIC_API_KEY") or None,
    )
