"""Ground truth parser for Amazon ML Challenge 2026.

Robustly parses tab-separated ground truth files (e.g. dataset/train/train_ground_truth.tsv)
into mappings or streams of Source 1 entity IDs to sets of true matched entity IDs.
Handles singleton entities (empty matched_entity_ids) cleanly as empty sets.
"""

from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Set, TextIO, Tuple, Union


EXPECTED_HEADER = ["source1_entity_id", "matched_entity_ids"]


def stream_ground_truth(
    source: Union[str, Path, TextIO],
) -> Iterator[Tuple[str, Set[str]]]:
    """Stream ground-truth entries as (s1_id, set_of_matched_ids).

    Args:
        source: File path (str or Path) or open text file object.

    Yields:
        Tuple of (source1_entity_id, set_of_matched_ids).
        If the entity is a singleton (no matches), matched IDs is an empty set().
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8-sig", newline="") as f:
            yield from _stream_from_file_object(f, source_name=str(source))
    else:
        yield from _stream_from_file_object(source, source_name="<stream>")


def _stream_from_file_object(
    f: TextIO, source_name: str
) -> Iterator[Tuple[str, Set[str]]]:
    header_line = f.readline()
    if not header_line:
        return

    header = [c.strip().lower() for c in header_line.rstrip("\r\n").split("\t")]
    if header != EXPECTED_HEADER:
        raise ValueError(
            f"Invalid ground-truth header in {source_name}: {header}. "
            f"Expected exactly {EXPECTED_HEADER}."
        )

    for line_num, line in enumerate(f, start=2):
        if not line.strip():
            continue  # skip empty lines or trailing newline

        parts = line.rstrip("\r\n").split("\t")
        if len(parts) == 1:
            # Trailing tab was omitted when matches are empty (singleton)
            s1_id = parts[0].strip()
            yield s1_id, set()
            continue

        if len(parts) != 2:
            raise ValueError(
                f"Malformed ground-truth row at line {line_num} in {source_name}: "
                f"found {len(parts)} tab-separated fields, expected 2."
            )

        s1_id = parts[0].strip()
        raw_matches = parts[1].strip()

        if not raw_matches:
            yield s1_id, set()
        else:
            matched_ids = {m.strip() for m in raw_matches.split(",") if m.strip()}
            yield s1_id, matched_ids


def parse_ground_truth(
    source: Union[str, Path, TextIO],
    allowed_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Parse ground truth into an in-memory dictionary.

    Args:
        source: Path to ground-truth TSV or an open file object.
        allowed_s1_ids: Optional set of S1 IDs to filter for. If provided,
            only matching S1 IDs are retained in memory.

    Returns:
        Mapping of source1_entity_id -> set of matched entity IDs.
    """
    ground_truth: Dict[str, Set[str]] = {}
    for s1_id, matched_ids in stream_ground_truth(source):
        if allowed_s1_ids is None or s1_id in allowed_s1_ids:
            ground_truth[s1_id] = matched_ids
    return ground_truth
