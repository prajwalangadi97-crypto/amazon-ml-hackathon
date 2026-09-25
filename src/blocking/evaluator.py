"""Comprehensive evaluation metrics and error analysis for candidate blocking."""

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple


@dataclass
class BlockingResult:
    """Detailed evaluation result for a blocking strategy."""

    strategy_id: str
    strategy_name: str
    total_val_entities: int
    total_true_matches: int
    retrieved_true_matches: int
    pair_recall: float
    macro_candidate_recall: float
    s2_recall: float
    s3_recall: float
    s2_true_count: int
    s2_retrieved_count: int
    s3_true_count: int
    s3_retrieved_count: int
    total_candidate_pairs: int
    full_search_space_size: int
    reduction_ratio: float
    avg_candidates_per_s1: float
    median_candidates_per_s1: float
    p95_candidates_per_s1: float
    max_candidates_per_s1: int
    entities_with_zero_candidates: int
    entities_with_all_true_retrieved: int
    entities_with_missed_true_matches: int
    runtime_seconds: float
    peak_memory_mb: float
    country_restricted: bool
    config: Dict[str, Any] = field(default_factory=dict)
    sample_retrieved_matches: List[Dict[str, Any]] = field(default_factory=list)
    sample_missed_matches: List[Dict[str, Any]] = field(default_factory=list)
    sample_large_candidate_entities: List[Dict[str, Any]] = field(default_factory=list)
    sample_zero_candidate_entities: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"Blocking Strategy: {self.strategy_id} — {self.strategy_name}",
            "=" * 70,
            f"Country Restricted        : {self.country_restricted}",
            f"Validation S1 Entities    : {self.total_val_entities:,}",
            f"Pair Candidate Recall     : {self.pair_recall * 100:.2f}% ({self.retrieved_true_matches:,} / {self.total_true_matches:,})",
            f"  S2 Candidate Recall     : {self.s2_recall * 100:.2f}% ({self.s2_retrieved_count:,} / {self.s2_true_count:,})",
            f"  S3 Candidate Recall     : {self.s3_recall * 100:.2f}% ({self.s3_retrieved_count:,} / {self.s3_true_count:,})",
            f"Macro Candidate Recall    : {self.macro_candidate_recall * 100:.2f}%",
            f"Entities Perfect Recall   : {self.entities_with_all_true_retrieved:,} ({self.entities_with_all_true_retrieved / self.total_val_entities * 100:.2f}%)",
            f"Entities Missed Matches   : {self.entities_with_missed_true_matches:,} ({self.entities_with_missed_true_matches / self.total_val_entities * 100:.2f}%)",
            f"Entities Zero Candidates  : {self.entities_with_zero_candidates:,} ({self.entities_with_zero_candidates / self.total_val_entities * 100:.2f}%)",
            "-" * 70,
            f"Total Candidate Pairs     : {self.total_candidate_pairs:,}",
            f"Reduction Ratio           : {self.reduction_ratio * 100:.6f}%",
            f"Avg Candidates per S1     : {self.avg_candidates_per_s1:.2f}",
            f"Median Candidates per S1  : {self.median_candidates_per_s1:.1f}",
            f"P95 Candidates per S1     : {self.p95_candidates_per_s1:.1f}",
            f"Max Candidates per S1     : {self.max_candidates_per_s1:,}",
            "-" * 70,
            f"Runtime                   : {self.runtime_seconds:.2f}s",
            f"Peak Memory (Delta)       : {self.peak_memory_mb:.2f} MB",
            "=" * 70,
        ]
        return "\n".join(lines)


def evaluate_blocking(
    candidates_map: Mapping[str, Set[str]],
    ground_truth: Mapping[str, Set[str]],
    s1_records: Mapping[str, Tuple[str, str, str]],  # s1_id -> (name, addr, country)
    target_records: Mapping[str, Tuple[str, str, str]],  # target_id -> (name, addr, country)
    total_pool_size: int,
    strategy_id: str,
    strategy_name: str,
    country_restricted: bool,
    runtime_seconds: float,
    peak_memory_mb: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
) -> BlockingResult:
    """Compute primary blocking metrics, percentiles, reduction ratio, and error cases."""
    total_val_entities = len(ground_truth)
    candidate_counts: List[int] = []
    entity_recalls: List[float] = []

    total_true_matches = 0
    retrieved_true_matches = 0

    s2_true_count = 0
    s2_retrieved_count = 0
    s3_true_count = 0
    s3_retrieved_count = 0

    entities_with_zero_cands = 0
    entities_with_all_true = 0
    entities_with_missed = 0

    retrieved_examples: List[Dict[str, Any]] = []
    missed_examples: List[Dict[str, Any]] = []
    zero_cand_examples: List[Dict[str, Any]] = []
    large_cand_candidates: List[Tuple[int, str]] = []

    for s1_id, true_set in ground_truth.items():
        cands = candidates_map.get(s1_id, set())
        n_cands = len(cands)
        candidate_counts.append(n_cands)
        large_cand_candidates.append((n_cands, s1_id))

        if n_cands == 0:
            entities_with_zero_cands += 1
            if len(zero_cand_examples) < 3:
                s1_data = s1_records.get(s1_id, ("", "", ""))
                zero_cand_examples.append({
                    "s1_id": s1_id,
                    "s1_name": s1_data[0],
                    "s1_address": s1_data[1],
                    "s1_country": s1_data[2],
                    "true_match_count": len(true_set),
                })

        n_true = len(true_set)
        total_true_matches += n_true

        # Partition true targets by source
        s2_true = {m for m in true_set if m.startswith("S2-")}
        s3_true = {m for m in true_set if m.startswith("S3-")}

        s2_true_count += len(s2_true)
        s3_true_count += len(s3_true)

        retrieved_set = cands & true_set
        n_retrieved = len(retrieved_set)
        retrieved_true_matches += n_retrieved

        retrieved_s2 = retrieved_set & s2_true
        retrieved_s3 = retrieved_set & s3_true

        s2_retrieved_count += len(retrieved_s2)
        s3_retrieved_count += len(s3_retrieved_count_set := retrieved_s3)

        if n_true == 0:
            # Singleton: 0 true matches lost -> recall is 1.0
            entity_recalls.append(1.0)
            entities_with_all_true += 1
        else:
            rec = n_retrieved / n_true
            entity_recalls.append(rec)
            if n_retrieved == n_true:
                entities_with_all_true += 1
            else:
                entities_with_missed += 1

            # Collect retrieved examples
            if len(retrieved_examples) < 3 and retrieved_set:
                target_id = next(iter(retrieved_set))
                s1_data = s1_records.get(s1_id, ("", "", ""))
                target_data = target_records.get(target_id, ("", "", ""))
                retrieved_examples.append({
                    "s1_id": s1_id,
                    "s1_name": s1_data[0],
                    "s1_address": s1_data[1],
                    "s1_country": s1_data[2],
                    "target_id": target_id,
                    "target_name": target_data[0],
                    "target_address": target_data[1],
                    "target_country": target_data[2],
                })

            # Collect missed examples
            missed_set = true_set - cands
            if len(missed_examples) < 3 and missed_set:
                target_id = next(iter(missed_set))
                s1_data = s1_records.get(s1_id, ("", "", ""))
                target_data = target_records.get(target_id, ("", "", ""))
                missed_examples.append({
                    "s1_id": s1_id,
                    "s1_name": s1_data[0],
                    "s1_address": s1_data[1],
                    "s1_country": s1_data[2],
                    "target_id": target_id,
                    "target_name": target_data[0],
                    "target_address": target_data[1],
                    "target_country": target_data[2],
                })

    # Collect large candidate set examples
    large_cand_candidates.sort(reverse=True)
    sample_large_cands: List[Dict[str, Any]] = []
    for count, s1_id in large_cand_candidates[:3]:
        s1_data = s1_records.get(s1_id, ("", "", ""))
        sample_large_cands.append({
            "s1_id": s1_id,
            "candidate_count": count,
            "s1_name": s1_data[0],
            "s1_address": s1_data[1],
            "s1_country": s1_data[2],
        })

    total_candidates = sum(candidate_counts)
    full_search_space = total_val_entities * total_pool_size
    reduction_ratio = 1.0 - (total_candidates / full_search_space) if full_search_space > 0 else 1.0

    pair_recall = (retrieved_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    macro_recall = statistics.mean(entity_recalls) if entity_recalls else 0.0
    s2_recall = (s2_retrieved_count / s2_true_count) if s2_true_count > 0 else 0.0
    s3_recall = (s3_retrieved_count / s3_true_count) if s3_true_count > 0 else 0.0

    sorted_counts = sorted(candidate_counts)
    avg_candidates = statistics.mean(candidate_counts) if candidate_counts else 0.0
    med_candidates = statistics.median(candidate_counts) if candidate_counts else 0.0
    p95_idx = int(0.95 * len(sorted_counts))
    p95_candidates = sorted_counts[min(p95_idx, len(sorted_counts) - 1)] if sorted_counts else 0.0
    max_candidates = sorted_counts[-1] if sorted_counts else 0

    return BlockingResult(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        total_val_entities=total_val_entities,
        total_true_matches=total_true_matches,
        retrieved_true_matches=retrieved_true_matches,
        pair_recall=pair_recall,
        macro_candidate_recall=macro_recall,
        s2_recall=s2_recall,
        s3_recall=s3_recall,
        s2_true_count=s2_true_count,
        s2_retrieved_count=s2_retrieved_count,
        s3_true_count=s3_true_count,
        s3_retrieved_count=s3_retrieved_count,
        total_candidate_pairs=total_candidates,
        full_search_space_size=full_search_space,
        reduction_ratio=reduction_ratio,
        avg_candidates_per_s1=avg_candidates,
        median_candidates_per_s1=med_candidates,
        p95_candidates_per_s1=p95_candidates,
        max_candidates_per_s1=max_candidates,
        entities_with_zero_candidates=entities_with_zero_cands,
        entities_with_all_true_retrieved=entities_with_all_true,
        entities_with_missed_true_matches=entities_with_missed,
        runtime_seconds=runtime_seconds,
        peak_memory_mb=peak_memory_mb,
        country_restricted=country_restricted,
        config=config or {},
        sample_retrieved_matches=retrieved_examples,
        sample_missed_matches=missed_examples,
        sample_large_candidate_entities=sample_large_cands,
        sample_zero_candidate_entities=zero_cand_examples,
    )
