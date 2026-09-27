"""Dynamic high-frequency blocking key filtering and fan-out protection.

Detects pathological blocking keys with extremely high S1 fan-out to prevent
generic country/city/street/business tokens from triggering massive RapidFuzz
scoring loops.

Key Features:
- Fully dynamic and statistical: No hard-coded countries, cities, or business names.
- Configurable frequency thresholds (absolute posting count or entity ratio).
- Preserves rare and informative keys (low fan-out tokens).
- Selectively preserves high-value discriminator keys (e.g. exact names, PIN codes).
- Preserves 100% baseline behavior when unconfigured (threshold=None).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class BlockingKeyFilterConfig:
    """Configuration for dynamic high-frequency blocking key pruning."""

    max_key_fanout: Optional[int] = None
    """Maximum allowed S1 postings per key (e.g. 500, 200, 20). None = no limit."""

    max_key_fanout_ratio: Optional[float] = None
    """Maximum allowed posting ratio relative to total indexed S1 entities (e.g. 0.005 for 0.5%). None = no limit."""

    preserve_exact_name: bool = True
    """Whether to exempt exact normalized name keys (en_...) from pruning."""

    preserve_pin_keys: bool = True
    """Whether to exempt specific postal PIN keys (pin_...) from pruning."""

    min_prune_threshold: int = 5
    """Safety floor: never prune keys with fewer than this many entities, even if ratio is tiny."""


def compute_effective_fanout_threshold(
    total_s1: int,
    config: Optional[BlockingKeyFilterConfig] = None,
) -> Optional[int]:
    """Compute the effective integer fan-out threshold for a given S1 entity pool size."""
    if config is None:
        return None

    if config.max_key_fanout is not None and config.max_key_fanout > 0:
        if config.max_key_fanout_ratio is not None and config.max_key_fanout_ratio > 0.0:
            ratio_th = max(config.min_prune_threshold, int(total_s1 * config.max_key_fanout_ratio))
            return min(config.max_key_fanout, ratio_th)
        return config.max_key_fanout

    if config.max_key_fanout_ratio is not None and config.max_key_fanout_ratio > 0.0:
        return max(config.min_prune_threshold, int(total_s1 * config.max_key_fanout_ratio))

    return None



def is_exempt_key(key: str, config: BlockingKeyFilterConfig) -> bool:
    """Determine whether a key is protected from pruning regardless of frequency."""
    if config.preserve_exact_name and key.startswith("en_"):
        return True
    if config.preserve_pin_keys and key.startswith("pin_"):
        return True
    return False


def prune_high_frequency_keys(
    c_idx: Dict[str, List[str]],
    total_s1: int,
    config: Optional[BlockingKeyFilterConfig] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Prune pathological high-fanout blocking keys from an inverted index.

    Parameters
    ----------
    c_idx : Dict[str, List[str]]
        Raw inverted index mapping blocking_key -> list of s1_ids.
    total_s1 : int
        Total number of S1 entities indexed in this country/partition.
    config : Optional[BlockingKeyFilterConfig]
        Filter configuration. If None or thresholds are None, returns c_idx untouched.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, Any]]
        - Pruned inverted index dictionary.
        - Pruning statistics and audit metadata.
    """
    if config is None:
        return c_idx, {
            "enabled": False,
            "effective_threshold": None,
            "total_keys_before": len(c_idx),
            "total_keys_after": len(c_idx),
            "pruned_key_count": 0,
            "pruned_postings": 0,
            "top_pruned_keys": [],
        }

    effective_threshold = compute_effective_fanout_threshold(total_s1, config)

    if effective_threshold is None:
        return c_idx, {
            "enabled": False,
            "effective_threshold": None,
            "total_keys_before": len(c_idx),
            "total_keys_after": len(c_idx),
            "pruned_key_count": 0,
            "pruned_postings": 0,
            "top_pruned_keys": [],
        }

    filtered_idx: Dict[str, List[str]] = {}
    pruned_keys: List[Tuple[str, int]] = []
    total_postings_before = 0
    total_postings_after = 0

    for key, postings in c_idx.items():
        sz = len(postings)
        total_postings_before += sz

        if sz > effective_threshold and not is_exempt_key(key, config):
            pruned_keys.append((key, sz))
        else:
            filtered_idx[key] = postings
            total_postings_after += sz

    pruned_keys.sort(key=lambda x: x[1], reverse=True)
    pruned_postings = total_postings_before - total_postings_after

    stats = {
        "enabled": True,
        "effective_threshold": effective_threshold,
        "total_keys_before": len(c_idx),
        "total_keys_after": len(filtered_idx),
        "pruned_key_count": len(pruned_keys),
        "pruned_postings": pruned_postings,
        "postings_reduction_pct": (
            (pruned_postings / total_postings_before * 100.0)
            if total_postings_before > 0
            else 0.0
        ),
        "top_pruned_keys": [
            {"key": k, "fanout": sz, "ratio": sz / total_s1 if total_s1 > 0 else 0.0}
            for k, sz in pruned_keys[:25]
        ],
    }

    return filtered_idx, stats
