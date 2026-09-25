"""Candidate generation and blocking strategies for entity resolution."""
from src.blocking.normalizers import (
    normalize_name_conservative,
    tokenize_name,
    extract_address_keys,
)

__all__ = [
    "normalize_name_conservative",
    "tokenize_name",
    "extract_address_keys",
]
