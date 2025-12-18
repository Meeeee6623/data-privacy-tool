import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from privacy_pipeline.utils import filter_slug

logger = logging.getLogger(__name__)


def _load_jsonl(paths: Sequence[Path]) -> Iterable[Dict]:
    for path in paths:
        if not path.exists():
            logger.warning("JSONL path %s does not exist; skipping", path)
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.exception("Failed to decode JSON line in %s", path)
                    continue


def _matches_filters(attributes: Dict, filters: Optional[Dict[str, str]]) -> bool:
    if not filters:
        return True
    return all(attributes.get(key) == value for key, value in filters.items())


def list_attribute_values(inputs: Sequence[Path], attributes: Sequence[str]) -> Dict[str, List[str]]:
    """Return sorted unique values for the provided attributes."""

    values: Dict[str, set[str]] = {attr: set() for attr in attributes}
    for record in _load_jsonl(inputs):
        attrs = record.get("attributes", {}) if isinstance(record, dict) else {}
        for attr in attributes:
            value = attrs.get(attr)
            if value is not None:
                values[attr].add(str(value))
    return {attr: sorted(vals) for attr, vals in values.items()}


def summarize_attributes(
    inputs: Sequence[Path], attributes: Sequence[str], filter_sets: Optional[List[Dict[str, str]]] = None
) -> Dict:
    """Provide counts for attribute values and optional filter coverage."""

    attr_counters: Dict[str, Counter] = {attr: Counter() for attr in attributes}
    filter_counts: Dict[str, int] = {}
    total = 0

    records = list(_load_jsonl(inputs))
    for record in records:
        total += 1
        attrs = record.get("attributes", {}) if isinstance(record, dict) else {}
        for attr in attributes:
            value = attrs.get(attr, "<missing>")
            attr_counters[attr][str(value)] += 1

    if filter_sets:
        for filt in filter_sets:
            slug = filter_slug(filt)
            filter_counts[slug] = sum(1 for record in records if _matches_filters(record.get("attributes", {}), filt))

    return {
        "total_records": total,
        "attributes": {attr: dict(counter) for attr, counter in attr_counters.items()},
        "filter_counts": filter_counts,
    }


def merge_filtered_outputs(
    stage: str, filename: Optional[str] = None, base_dir: Path = Path("filtered"), output_path: Optional[Path] = None
) -> Path:
    """Merge filter-specific JSONL files for a pipeline stage into a single output."""

    stage_dir = base_dir / stage
    resolved_filename = filename or {
        "yoloe": "yoloe_output.jsonl",
        "gemini": "gemini_output.jsonl",
        "index": "image_index.jsonl",
    }.get(stage, f"{stage}.jsonl")

    direct_candidates = list(stage_dir.glob("*.jsonl")) if stage_dir.exists() else []
    nested_candidates = sorted(stage_dir.glob(f"*/{resolved_filename}")) if stage_dir.exists() else []

    candidates = sorted(direct_candidates) if direct_candidates else nested_candidates
    if not candidates:
        raise FileNotFoundError(
            f"No filtered outputs found under {stage_dir} for {resolved_filename} or direct JSONL files"
        )

    destination = output_path or Path(resolved_filename)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w") as dest:
        for path in candidates:
            logger.info("Merging %s", path)
            with path.open() as src:
                for line in src:
                    if line.strip():
                        dest.write(line if line.endswith("\n") else line + "\n")
    logger.info("Merged %d files into %s", len(candidates), destination)
    return destination
