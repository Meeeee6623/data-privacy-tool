import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from privacy_pipeline.config import DatasetConfig
from privacy_pipeline.progress import progress

logger = logging.getLogger(__name__)


def _gather_images(config: DatasetConfig) -> List[Path]:
    pattern = "**/*" if config.recursive else "*"
    images = [
        p
        for p in config.image_root.glob(pattern)
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ]
    logger.debug("Found %d images under %s", len(images), config.image_root)
    return images


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
        logger.debug("Attributes for %s from path map: %s", image_path, attributes)
        return attributes

    rel_parts = image_path.relative_to(config.image_root).parts
    attributes: Dict[str, str] = {}
    for idx, name in enumerate(config.path_attributes or []):
        if idx < len(rel_parts) - 1:  # skip filename
            attributes[name] = rel_parts[idx]
    logger.debug("Attributes for %s from path segments: %s", image_path, attributes)
    return attributes


def _load_user_jsonl(user_jsonl: Path) -> Iterable[Dict]:
    with user_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_image_index(config: DatasetConfig, verbose: bool = False) -> Path:
    logger.info("Building image index from %s", config.image_root)
    images = _gather_images(config)
    records: List[Dict] = []

    for image_path in progress(images, verbose, "Indexing images", total=len(images), unit="image"):
        base_record = {
            "image_path": str(image_path),
            "attributes": _attributes_from_path(image_path, config),
        }
        records.append(base_record)

    if config.user_jsonl:
        logger.debug("Loading user-provided JSONL from %s", config.user_jsonl)
        for user_record in _load_user_jsonl(config.user_jsonl):
            if "image_path" not in user_record:
                continue
            merged = {
                "image_path": user_record["image_path"],
                "attributes": user_record.get("attributes", {}),
            }
            records.append(merged)
    logger.info("Prepared %d index records", len(records))

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote image index to %s", config.output_jsonl)

    return config.output_jsonl
