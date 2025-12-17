import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image
from ultralytics import YOLOE

from privacy_pipeline.config import YoloEConfig

logger = logging.getLogger(__name__)


def _load_classes(classes_path: Path) -> List[str]:
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {classes_path}")
    logger.debug("Loaded %d classes from %s", len(classes), classes_path)
    return classes


def _save_class_mapping(output_dir: Path, classes: List[str]) -> Path:
    mapping_path = output_dir / "yoloe_custom_mapping.txt"
    with mapping_path.open("w") as f:
        for idx, name in enumerate(classes):
            f.write(f"{name}: {idx}\n")
    logger.debug("Saved class mapping to %s", mapping_path)
    return mapping_path


def _load_custom_model(model_path: Path, classes_path: Path, output_dir: Path):
    classes = _load_classes(classes_path)
    model = YOLOE(str(model_path))
    logger.debug("Loaded YOLOE model from %s", model_path)

    model.set_classes(classes, model.get_text_pe(classes))
    logger.info("Configured YOLOE model with %d custom classes", len(classes))

    custom_model_path = model_path.with_name(f"{model_path.stem}-custom{model_path.suffix}")
    custom_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(custom_model_path)
    logger.info("Saved customized YOLOE model to %s", custom_model_path)

    mapping_path = _save_class_mapping(output_dir, classes)
    return model, custom_model_path, mapping_path


def _load_index(jsonl_path: Path) -> Iterable[Dict]:
    logger.debug("Loading index from %s", jsonl_path)
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _matches_filters(attributes: Dict, filters: Optional[Dict[str, str]]) -> bool:
    if not filters:
        return True
    return all(attributes.get(key) == value for key, value in filters.items())


def run_yoloe(index_jsonl: Path, config: YoloEConfig) -> Path:
    logger.info("Running YOLOE on index %s", index_jsonl)
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    model, custom_model_path, mapping_path = _load_custom_model(
        config.model_path, config.classes_path, config.output_jsonl.parent
    )
    logger.info(
        "Model ready. Custom weights: %s. Class mapping: %s",
        custom_model_path,
        mapping_path,
    )

    index_records = [
        record
        for record in _load_index(index_jsonl)
        if _matches_filters(record.get("attributes", {}), config.attribute_filters)
    ]

    if not index_records:
        logger.warning("No index records matched the provided attribute filters; nothing to process")

    image_paths = [Path(record["image_path"]) for record in index_records]
    common_root = config.dataset_image_root
    if common_root is None and image_paths:
        common_root = Path(os.path.commonpath([str(p.parent) for p in image_paths]))

    viz_dir: Optional[Path] = None
    if config.visualize:
        viz_dir = config.visualization_dir or config.output_jsonl.parent / "yoloe_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving YOLOE visualizations to %s", viz_dir)

    processed = 0
    with config.output_jsonl.open("w") as f:
        for record in index_records:
            image_path = Path(record["image_path"])
            result = model.predict(str(image_path), conf=config.threshold, verbose=False)[0]
            detections = []
            for box, cls_idx, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                name = result.names[int(cls_idx)]
                detections.append(
                    {
                        "class": name,
                        "confidence": float(conf),
                        "bbox": [float(v) for v in box],
                    }
                )
            logger.debug(
                "Processed %s with %d detections (threshold=%.2f)",
                image_path,
                len(detections),
                config.threshold,
            )

            visualization_path = None
            if viz_dir and detections:
                plotted = result.plot()
                relative_image = None
                if common_root:
                    try:
                        relative_image = image_path.relative_to(common_root)
                    except ValueError:
                        logger.debug("Could not derive relative path for %s from %s", image_path, common_root)
                relative_image = relative_image or Path(image_path.name)

                viz_path = viz_dir / relative_image.with_name(f"{relative_image.stem}_yoloe.png")
                viz_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(plotted[..., ::-1]).save(viz_path)
                visualization_path = str(viz_path)
            
            if detections:
                output_record = {
                    "image_path": str(image_path),
                    "attributes": record.get("attributes", {}),
                    "detections": detections,
                    "visualization_path": visualization_path,
                }
                f.write(json.dumps(output_record) + "\n")
                f.flush()
            processed += 1

    logger.info("Wrote YOLOE output for %d images to %s", processed, config.output_jsonl)

    return config.output_jsonl
