import re
from pathlib import Path
from typing import Dict, Optional


def filter_slug(filters: Optional[Dict[str, str]]) -> str:
    """Return a filesystem-friendly slug for a filter mapping.

    Filters are sorted to keep slugs stable regardless of input order.
    """

    if not filters:
        return "all"

    parts: list[str] = []
    for key, value in sorted(filters.items()):
        safe_key = re.sub(r"[^A-Za-z0-9_-]", "-", str(key))
        safe_value = re.sub(r"[^A-Za-z0-9_-]", "-", str(value))
        parts.append(f"{safe_key}-{safe_value}")
    return "_".join(parts)


def filtered_stage_dir(stage: str, filters: Optional[Dict[str, str]]) -> Path:
    """Return the directory to store outputs for a filter-specific run."""

    return Path("filtered") / stage / filter_slug(filters)
