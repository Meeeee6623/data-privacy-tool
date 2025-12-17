import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from privacy_pipeline.config import DatasetConfig, GeminiConfig, YoloEConfig
from privacy_pipeline.dataset_index import build_image_index
from privacy_pipeline.yoloe_runner import run_yoloe
from privacy_pipeline.gemini_pipeline import (
    collect_flagged_scenes,
    create_gemini_batches,
    parse_gemini_output,
    submit_gemini_batches,
)


DEFAULT_CONFIG_PATH = Path("config.yaml")


def _add_common_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("image_root", type=Path, nargs="?", help="Root directory containing images")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Search recursively for images",
    )
    parser.add_argument(
        "--path-attributes",
        nargs="*",
        help="Optional attribute names mapped to directory segments (e.g. lab building scene)",
    )
    parser.add_argument(
        "--path-attribute-map",
        nargs="*",
        metavar="NAME:LEVELS_UP",
        help="Map attribute name to levels up from the file (1=parent, e.g. lab:3 building:2)",
    )
    parser.add_argument("--user-jsonl", type=Path, help="Additional JSONL with custom attributes")


def _parse_attribute_map(raw: list[str] | None) -> dict[str, int] | None:
    if not raw:
        return None

    mapping = {}
    for item in raw:
        if ":" not in item:
            continue
        name, value = item.split(":", 1)
        try:
            mapping[name] = int(value)
        except ValueError:
            continue
    return mapping or None


def _parse_attribute_filters(raw: list[str] | None) -> dict[str, str] | None:
    if not raw:
        return None

    filters: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        if name:
            filters[name] = value
    return filters or None


def _lookup(config: Dict[str, Any] | None, *keys: str) -> Any:
    current: Any = config or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _load_pipeline_config(config_path: Optional[Path]) -> Dict[str, Any]:
    path_to_load = config_path
    if path_to_load is None and DEFAULT_CONFIG_PATH.exists():
        path_to_load = DEFAULT_CONFIG_PATH

    if path_to_load is None:
        return {}

    data = yaml.safe_load(path_to_load.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping")
    logging.getLogger(__name__).info("Loaded configuration from %s", path_to_load)
    return data


def _resolve_path(cli_value: Optional[Path], config_value: Any, default: Optional[Path]) -> Optional[Path]:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return Path(config_value)
    return default


def _coerce_bool(cli_value: Optional[bool], config_value: Any, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if config_value is not None:
        return bool(config_value)
    return default


def _normalize_attribute_map(raw: Any) -> dict[str, int] | None:
    if raw is None:
        return None

    if isinstance(raw, dict):
        normalized: dict[str, int] = {}
        for key, value in raw.items():
            try:
                normalized[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return normalized or None

    if isinstance(raw, list):
        return _parse_attribute_map([str(item) for item in raw])

    return None


def _arg_value(args: argparse.Namespace, name: str) -> Any:
    return getattr(args, name, None)


def _build_dataset_config(args: argparse.Namespace, config: Dict[str, Any]) -> DatasetConfig:
    dataset_cfg = config.get("dataset", {}) if isinstance(config, dict) else {}
    image_root = _resolve_path(args.image_root, dataset_cfg.get("image_root"), None)
    if image_root is None:
        raise ValueError("image_root must be provided via CLI or config file")

    recursive = _coerce_bool(args.recursive, dataset_cfg.get("recursive"), True)
    path_attributes = args.path_attributes if args.path_attributes is not None else dataset_cfg.get("path_attributes")
    path_attribute_map = (
        _parse_attribute_map(args.path_attribute_map)
        if args.path_attribute_map is not None
        else _normalize_attribute_map(dataset_cfg.get("path_attribute_map"))
    )
    user_jsonl = _resolve_path(args.user_jsonl, dataset_cfg.get("user_jsonl"), None)
    output_jsonl = _resolve_path(args.output, dataset_cfg.get("output_jsonl"), Path("image_index.jsonl"))

    return DatasetConfig(
        image_root=image_root,
        recursive=recursive,
        path_attributes=path_attributes,
        path_attribute_map=path_attribute_map,
        user_jsonl=user_jsonl,
        output_jsonl=output_jsonl,
    )


def _build_yoloe_config(args: argparse.Namespace, config: Dict[str, Any]) -> YoloEConfig:
    yolo_cfg = config.get("yoloe", {}) if isinstance(config, dict) else {}
    model_path = _resolve_path(args.model, yolo_cfg.get("model_path"), Path("yoloe-11l-seg.pt"))
    threshold = args.threshold if args.threshold is not None else yolo_cfg.get("threshold", 0.5)
    visualize = _coerce_bool(args.visualize, yolo_cfg.get("visualize"), False)
    visualization_dir = _resolve_path(
        args.viz_dir,
        yolo_cfg.get("visualization_dir"),
        Path("yoloe_visualizations"),
    )
    output_jsonl = _resolve_path(args.output, yolo_cfg.get("output_jsonl"), Path("yoloe_output.jsonl"))
    attribute_filters = (
        _parse_attribute_filters(args.attribute_filter)
        if args.attribute_filter is not None
        else yolo_cfg.get("attribute_filters")
    )
    dataset_image_root = _resolve_path(
        None, _lookup(config, "dataset", "image_root"), None
    )

    return YoloEConfig(
        model_path=model_path,
        threshold=threshold,
        visualize=visualize,
        visualization_dir=visualization_dir,
        output_jsonl=output_jsonl,
        attribute_filters=attribute_filters,
        dataset_image_root=dataset_image_root,
    )


def _build_gemini_config(
    args: argparse.Namespace, config: Dict[str, Any], require_prompt: bool = True
) -> GeminiConfig:
    gemini_cfg = config.get("gemini", {}) if isinstance(config, dict) else {}

    prompt_arg = _arg_value(args, "prompt")
    prompt = prompt_arg if prompt_arg is not None else gemini_cfg.get("prompt")
    if require_prompt and not prompt:
        raise ValueError("prompt must be provided via CLI or config file")

    classes_arg = _arg_value(args, "classes")
    classes_to_forward = classes_arg if classes_arg is not None else gemini_cfg.get("classes_to_forward")
    threshold_arg = _arg_value(args, "threshold")
    min_confidence = threshold_arg if threshold_arg is not None else gemini_cfg.get("min_confidence", 0.5)
    scene_level_arg = _arg_value(args, "scene_level")
    scene_level = scene_level_arg if scene_level_arg is not None else gemini_cfg.get("scene_directory_level", 1)
    max_bytes_arg = _arg_value(args, "max_bytes")
    max_batch_size = max_bytes_arg if max_bytes_arg is not None else gemini_cfg.get("max_batch_size_bytes", 1.85 * 1024 ** 3)
    output_batch_dir = _resolve_path(
        _arg_value(args, "batch_dir"),
        gemini_cfg.get("output_batch_dir"),
        Path("gemini_batches"),
    )
    gcs_arg = _arg_value(args, "gcs_bucket")
    gcs_bucket = gcs_arg if gcs_arg is not None else gemini_cfg.get("gcs_bucket")
    submitted_jobs_file = _resolve_path(
        _arg_value(args, "jobs_file"),
        gemini_cfg.get("submitted_jobs_file"),
        Path("gemini_jobs.json"),
    )
    project_arg = _arg_value(args, "project")
    project = project_arg if project_arg is not None else gemini_cfg.get("project")
    location_arg = _arg_value(args, "location")
    location = location_arg if location_arg is not None else gemini_cfg.get("location", "us-central1")
    model_arg = _arg_value(args, "model")
    model = model_arg if model_arg is not None else gemini_cfg.get("model", "gemini-2.0-flash")
    final_output = _resolve_path(
        _arg_value(args, "final_output"),
        gemini_cfg.get("final_output_jsonl"),
        Path("gemini_output.jsonl"),
    )
    attribute_filters = (
        _parse_attribute_filters(_arg_value(args, "attribute_filter"))
        if _arg_value(args, "attribute_filter") is not None
        else gemini_cfg.get("attribute_filters")
    )

    return GeminiConfig(
        prompt=prompt or "",
        classes_to_forward=classes_to_forward,
        min_confidence=min_confidence,
        scene_directory_level=scene_level,
        max_batch_size_bytes=max_batch_size,
        output_batch_dir=output_batch_dir,
        gcs_bucket=gcs_bucket,
        submitted_jobs_file=submitted_jobs_file,
        project=project,
        location=location,
        model=model,
        final_output_jsonl=final_output,
        attribute_filters=attribute_filters,
    )


def _add_common_yolo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        type=Path,
        nargs="?",
        help="Path to YOLOE model (defaults to yoloe-11l-seg.pt)",
    )
    parser.add_argument("index", type=Path, nargs="?", help="Input image index JSONL")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--visualize",
        "--no-visualize",
        dest="visualize",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--viz-dir", type=Path, help="Directory for YOLOE visualizations")
    parser.add_argument(
        "--attribute-filter",
        nargs="*",
        metavar="NAME=VALUE",
        help="Only process records whose attributes match all provided filters",
    )


def _add_common_gemini_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yolo_output", type=Path, help="YOLOE output JSONL")
    parser.add_argument("--prompt")
    parser.add_argument("--classes", nargs="*", help="YOLOE classes that trigger Gemini")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--scene-level", type=int)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--max-bytes", type=float)
    parser.add_argument("--gcs-bucket", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--location", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--jobs-file", type=Path)
    parser.add_argument("--final-output", type=Path)
    parser.add_argument(
        "--attribute-filter",
        nargs="*",
        metavar="NAME=VALUE",
        help="Only process records whose attributes match all provided filters",
    )


def _configure_logging(verbose: bool) -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "privacy_pipeline.log"

    console_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger().handlers[0].setLevel(logging.DEBUG)
    logging.getLogger().handlers[1].setLevel(console_level)
    logging.getLogger(__name__).debug(
        "Logging configured. Verbose=%s. Log file: %s", verbose, log_file
    )
    return log_file


def main():
    parser = argparse.ArgumentParser(description="Privacy screening pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML configuration file to load defaults from",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and debug print statements",
    )

    index_parser = subparsers.add_parser("index", help="Build image index JSONL")
    _add_common_index_args(index_parser)

    yolo_parser = subparsers.add_parser("yoloe", help="Run YOLOE inference")
    _add_common_yolo_args(yolo_parser)

    gemini_parser = subparsers.add_parser("prepare-gemini", help="Create Gemini batch JSONL files")
    _add_common_gemini_args(gemini_parser)

    submit_parser = subparsers.add_parser("submit-gemini", help="Submit Gemini batches and store job names")
    submit_parser.add_argument("batch_dir", type=Path, nargs="?")
    submit_parser.add_argument("--jobs-file", type=Path)
    submit_parser.add_argument("--gcs-bucket")
    submit_parser.add_argument("--project")
    submit_parser.add_argument("--location")
    submit_parser.add_argument("--model")

    parse_parser = subparsers.add_parser("parse-gemini", help="Parse Gemini batch outputs into JSONL")
    parse_parser.add_argument("outputs", nargs="+", type=Path)
    parse_parser.add_argument("--original-index", type=Path)
    parse_parser.add_argument("--scene-level", type=int)
    parse_parser.add_argument("--final-output", type=Path)
    parse_parser.add_argument(
        "--attribute-filter",
        nargs="*",
        metavar="NAME=VALUE",
        help="Only include Gemini outputs whose attributes match all provided filters",
    )

    args = parser.parse_args()

    config_data = _load_pipeline_config(args.config)

    log_file = _configure_logging(args.verbose)
    logging.getLogger(__name__).info("Logs will be written to %s", log_file)

    if args.command == "index":
        config = _build_dataset_config(args, config_data)
        output = build_image_index(config)
        print(f"Wrote image index to {output}")

    elif args.command == "yoloe":
        dataset_cfg = config_data.get("dataset", {}) if isinstance(config_data, dict) else {}
        index_path = args.index or dataset_cfg.get("output_jsonl")
        if not index_path:
            raise ValueError("Image index JSONL must be provided via CLI or config file")
        config = _build_yoloe_config(args, config_data)
        output = run_yoloe(Path(index_path), config)
        print(f"Wrote YOLOE output to {output}")

    elif args.command == "prepare-gemini":
        config = _build_gemini_config(args, config_data)
        yolo_output = _resolve_path(
            args.yolo_output,
            _lookup(config_data, "yoloe", "output_jsonl"),
            default=Path("yoloe_output.jsonl"),
        )
        flagged = collect_flagged_scenes(yolo_output, config)
        batch_files = create_gemini_batches(flagged, config)
        print(json.dumps({"flagged_scenes": len(flagged), "batch_files": [str(p) for p in batch_files]}, indent=2))

    elif args.command == "submit-gemini":
        output_batch_dir = _resolve_path(
            args.batch_dir,
            _lookup(config_data, "gemini", "output_batch_dir"),
            default=Path("gemini_batches"),
        )
        batch_files = sorted(output_batch_dir.glob("*.jsonl"))
        config = _build_gemini_config(args, config_data, require_prompt=False)
        job_names = submit_gemini_batches(batch_files, config)
        print(json.dumps({"submitted_jobs": job_names}, indent=2))

    elif args.command == "parse-gemini":
        config = _build_gemini_config(args, config_data, require_prompt=False)
        original_index = _resolve_path(
            args.original_index,
            _lookup(config_data, "dataset", "output_jsonl"),
            default=None,
        )
        if original_index is None:
            raise ValueError("--original-index must be provided via CLI or config file")
        output = parse_gemini_output(args.outputs, original_index, config)
        print(f"Wrote parsed Gemini output to {output}")


if __name__ == "__main__":
    main()
