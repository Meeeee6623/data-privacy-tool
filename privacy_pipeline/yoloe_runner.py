import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image
from ultralytics import YOLO

from privacy_pipeline.config import YoloEConfig


def _load_classes(classes_file: Optional[Path]) -> Optional[List[str]]:
    if not classes_file:
        return None
    with classes_file.open() as f:
        return [line.strip() for line in f if line.strip()]


def _load_index(jsonl_path: Path) -> Iterable[Dict]:
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _should_keep_detection(name: str, allowed_classes: Optional[List[str]]) -> bool:
    if not allowed_classes:
        return True
    return name in allowed_classes


def run_yoloe(index_jsonl: Path, config: YoloEConfig) -> Path:
    classes = _load_classes(config.classes_file)
    model = YOLO(str(config.model_path))

    records: List[Dict] = []
    viz_dir: Optional[Path] = None
    if config.visualize:
        viz_dir = config.visualization_dir or config.output_jsonl.parent / "yoloe_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    for record in _load_index(index_jsonl):
        image_path = Path(record["image_path"])
        result = model.predict(str(image_path), conf=config.threshold, verbose=False)[0]
        detections = []
        for box, cls_idx, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            name = result.names[int(cls_idx)]
            if not _should_keep_detection(name, classes):
                continue
            detections.append(
                {
                    "class": name,
                    "confidence": float(conf),
                    "bbox": [float(v) for v in box],
                }
            )

        visualization_path = None
        if viz_dir and detections:
            plotted = result.plot()
            viz_path = viz_dir / f"{image_path.stem}_yoloe.png"
            Image.fromarray(plotted[..., ::-1]).save(viz_path)
            visualization_path = str(viz_path)

        output_record = {
            "image_path": str(image_path),
            "attributes": record.get("attributes", {}),
            "detections": detections,
            "visualization_path": visualization_path,
        }
        records.append(output_record)

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return config.output_jsonl
