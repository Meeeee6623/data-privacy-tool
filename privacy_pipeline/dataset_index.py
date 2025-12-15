import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from privacy_pipeline.config import DatasetConfig


def _gather_images(config: DatasetConfig) -> List[Path]:
    pattern = "**/*" if config.recursive else "*"
    return [
        p
        for p in config.image_root.glob(pattern)
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ]


def _attributes_from_path(image_path: Path, config: DatasetConfig) -> Dict[str, str]:
    if not config.path_attributes and not config.path_attribute_map:
        return {}

    if config.path_attribute_map:
        attributes: Dict[str, str] = {}
        current = image_path.parent
        for name, levels_up in config.path_attribute_map.items():
            if levels_up < 1:
                continue
            target = current
            for _ in range(levels_up - 1):
                if target.parent == target:
                    target = None
                    break
                target = target.parent
            if target and target != target.parent:
                attributes[name] = target.name
        return attributes

    rel_parts = image_path.relative_to(config.image_root).parts
    attributes: Dict[str, str] = {}
    for idx, name in enumerate(config.path_attributes or []):
        if idx < len(rel_parts) - 1:  # skip filename
            attributes[name] = rel_parts[idx]
    return attributes


def _load_user_jsonl(user_jsonl: Path) -> Iterable[Dict]:
    with user_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_image_index(config: DatasetConfig) -> Path:
    images = _gather_images(config)
    records: List[Dict] = []

    for image_path in images:
        base_record = {
            "image_path": str(image_path),
            "attributes": _attributes_from_path(image_path, config),
        }
        records.append(base_record)

    if config.user_jsonl:
        for user_record in _load_user_jsonl(config.user_jsonl):
            if "image_path" not in user_record:
                continue
            merged = {
                "image_path": user_record["image_path"],
                "attributes": user_record.get("attributes", {}),
            }
            records.append(merged)

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return config.output_jsonl
