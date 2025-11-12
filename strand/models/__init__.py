"""Model integrations (experimental).

These classes are placeholders for future adapters and are not wired into
the reward blocks by default. Current rewards use lightweight heuristics.
"""

from .biobert import BioBERTModel
from .embedding_cache import EmbeddingCache
from .esmfold import ESMFoldModel
from .protbert import ProtBERTModel

__all__ = ["ESMFoldModel", "ProtBERTModel", "BioBERTModel", "EmbeddingCache"]
