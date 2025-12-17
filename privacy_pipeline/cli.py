import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from privacy_pipeline.config import DatasetConfig, GeminiConfig, YoloEConfig
from privacy_pipeline.dataset_index import build_image_index
from privacy_pipeline.json_utils import list_attribute_values, merge_filtered_outputs, summarize_attributes
from privacy_pipeline.utils import filtered_stage_dir
from privacy_pipeline.yoloe_runner import run_yoloe
from privacy_pipeline.gemini_pipeline import (
    collect_flagged_scenes,
    create_gemini_batches,
    download_completed_jobs,
    get_gemini_job_status,
    clean_gemini_logs,
    parse_gemini_output,
    submit_gemini_batches,
)


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_JSON_UTILS_CONFIG_PATH = Path("json_utils.config.yaml")


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


def _parse_filter_sets(raw: list[list[str]] | None) -> list[dict[str, str]] | None:
    if not raw:
        return None

    parsed: list[dict[str, str]] = []
    for item in raw:
        parsed_filter = _parse_attribute_filters(item)
        if parsed_filter:
            parsed.append(parsed_filter)
    return parsed or None


def _normalize_path_list(raw: Any) -> list[Path] | None:
    if raw is None:
        return None

    if isinstance(raw, (str, Path)):
        return [Path(raw)]

    if isinstance(raw, list):
        paths: list[Path] = []
        for item in raw:
            if item is None:
                continue
            paths.append(Path(item))
        return paths or None
    return None


def _normalize_str_list(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        values = [str(item) for item in raw if item is not None]
        return values or None
    if isinstance(raw, (str, Path)):
        return [str(raw)]
    return None


def _parse_category_descriptions(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None

    if isinstance(raw, dict):
        parsed = {str(k): str(v) for k, v in raw.items() if k}
        return parsed or None

    if isinstance(raw, list):
        parsed: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, str) or ":" not in item:
                continue
            name, description = item.split(":", 1)
            if name:
                parsed[name] = description
        return parsed or None

    if isinstance(raw, (str, Path)):
        text = str(raw)
        if ":" in text:
            name, description = text.split(":", 1)
            if name:
                return {name: description}
    return None


def _normalize_filter_sets(raw: Any) -> list[dict[str, str]] | None:
    if raw is None:
        return None

    candidates = raw if isinstance(raw, list) else [raw]
    parsed: list[dict[str, str]] = []
    for item in candidates:
        if isinstance(item, dict):
            filt = {str(k): str(v) for k, v in item.items() if v is not None}
            if filt:
                parsed.append(filt)
        elif isinstance(item, list):
            filter_dict = _parse_attribute_filters([str(x) for x in item])
            if filter_dict:
                parsed.append(filter_dict)
        elif isinstance(item, (str, Path)):
            filter_dict = _parse_attribute_filters([str(item)])
            if filter_dict:
                parsed.append(filter_dict)
    return parsed or None


def _lookup(config: Dict[str, Any] | None, *keys: str) -> Any:
    current: Any = config or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _load_config(config_path: Optional[Path], default_path: Path) -> Dict[str, Any]:
    path_to_load = config_path
    if path_to_load is None and default_path.exists():
        path_to_load = default_path

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


def _load_job_names(job_names: list[str], jobs_file: Path) -> list[str]:
    if job_names:
        return job_names
    if not jobs_file.exists():
        raise ValueError("Job names must be provided via CLI or config file")
    loaded_jobs = json.loads(jobs_file.read_text() or "[]")
    if not isinstance(loaded_jobs, list):
        raise ValueError("Jobs file must contain a JSON array of job names")
    return [str(job) for job in loaded_jobs if job]


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
    attribute_filters = (
        _parse_attribute_filters(args.attribute_filter)
        if args.attribute_filter is not None
        else yolo_cfg.get("attribute_filters")
    )
    default_output = Path("yoloe_output.jsonl")
    default_visualizations = Path("yoloe_visualizations")
    if attribute_filters:
        stage_dir = filtered_stage_dir("yoloe", attribute_filters)
        default_output = stage_dir / default_output
        default_visualizations = stage_dir / default_visualizations

    visualization_dir = _resolve_path(
        args.viz_dir,
        yolo_cfg.get("visualization_dir"),
        default_visualizations,
    )

    output_jsonl = _resolve_path(args.output, yolo_cfg.get("output_jsonl"), default_output)
    dataset_image_root = _resolve_path(
        None, _lookup(config, "dataset", "image_root"), None
    )

    return YoloEConfig(
        model_path=model_path,
        threshold=threshold,
        visualize=visualize,
        visualization_dir=visualization_dir or default_visualizations,
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
    base_batch_dir = Path("gemini_batches")
    base_jobs_file = Path("gemini_jobs.json")
    base_final_output = Path("gemini_output.jsonl")
    attribute_filters = (
        _parse_attribute_filters(_arg_value(args, "attribute_filter"))
        if _arg_value(args, "attribute_filter") is not None
        else gemini_cfg.get("attribute_filters")
    )

    if attribute_filters:
        stage_dir = filtered_stage_dir("gemini", attribute_filters)
        base_batch_dir = stage_dir / base_batch_dir
        base_jobs_file = stage_dir / base_jobs_file
        base_final_output = stage_dir / base_final_output

    output_batch_dir = _resolve_path(
        _arg_value(args, "batch_dir"),
        gemini_cfg.get("output_batch_dir"),
        base_batch_dir,
    )
    gcs_arg = _arg_value(args, "gcs_bucket")
    gcs_bucket = gcs_arg if gcs_arg is not None else gemini_cfg.get("gcs_bucket")
    submitted_jobs_file = _resolve_path(
        _arg_value(args, "jobs_file"),
        gemini_cfg.get("submitted_jobs_file"),
        base_jobs_file,
    )
    project_arg = _arg_value(args, "project")
    project = project_arg if project_arg is not None else gemini_cfg.get("project")
    location_arg = _arg_value(args, "location")
    location = location_arg if location_arg is not None else gemini_cfg.get("location", "us-central1")
    model_arg = _arg_value(args, "model")
    model = model_arg if model_arg is not None else gemini_cfg.get("model", "gemini-2.0-flash")
    flag_categories_arg = _arg_value(args, "flag_categories")
    flag_categories = (
        _normalize_str_list(flag_categories_arg)
        if flag_categories_arg is not None
        else _normalize_str_list(gemini_cfg.get("flag_categories"))
    )
    flag_category_descriptions_arg = _arg_value(args, "flag_category_descriptions")
    flag_category_descriptions = (
        _parse_category_descriptions(flag_category_descriptions_arg)
        if flag_category_descriptions_arg is not None
        else _parse_category_descriptions(gemini_cfg.get("flag_category_descriptions"))
    )
    final_output = _resolve_path(
        _arg_value(args, "final_output"),
        gemini_cfg.get("final_output_jsonl"),
        base_final_output,
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
        flag_categories=flag_categories,
        flag_category_descriptions=flag_category_descriptions,
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
        "--flag-categories",
        nargs="*",
        help="Optional list of allowed #FLAG categories (defaults to built-in set)",
    )
    parser.add_argument(
        "--flag-category-descriptions",
        action="append",
        metavar="NAME:DESCRIPTION",
        help="Optional descriptions for #FLAG categories (can be provided multiple times)",
    )
    parser.add_argument(
        "--attribute-filter",
        nargs="*",
        metavar="NAME=VALUE",
        help="Only process records whose attributes match all provided filters",
    )


def _configure_logging(verbose: bool, command: str) -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_suffix = command.replace("-", "_") if command else "privacy_pipeline"
    log_file = log_dir / f"{log_suffix}.log"

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
        "Logging configured. Verbose=%s. Command=%s. Log file: %s",
        verbose,
        command,
        log_file,
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

    check_parser = subparsers.add_parser("check-gemini", help="Check status of submitted Gemini batches")
    check_parser.add_argument("job_names", nargs="*", help="Optional Gemini batch job names")
    check_parser.add_argument("--jobs-file", type=Path, help="JSON file containing submitted job names")
    check_parser.add_argument("--project")
    check_parser.add_argument("--location")

    download_parser = subparsers.add_parser(
        "download-gemini", help="Download completed Gemini batch outputs and clean logs"
    )
    download_parser.add_argument("job_names", nargs="*", help="Optional Gemini batch job names")
    download_parser.add_argument("--jobs-file", type=Path, help="JSON file containing submitted job names")
    download_parser.add_argument("--output-dir", type=Path, help="Directory to store downloaded outputs")
    download_parser.add_argument("--cleaned-dir", type=Path, help="Directory to store cleaned logs")
    download_parser.add_argument("--project")
    download_parser.add_argument("--location")
    download_parser.add_argument(
        "--flag-categories",
        nargs="*",
        help="Optional list of allowed #FLAG categories (defaults to built-in set)",
    )
    download_parser.add_argument(
        "--flag-category-descriptions",
        action="append",
        metavar="NAME:DESCRIPTION",
        help="Optional descriptions for #FLAG categories (can be provided multiple times)",
    )

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

    json_parser = subparsers.add_parser("json-utils", help="Inspect and merge pipeline JSONL files")
    json_subparsers = json_parser.add_subparsers(dest="json_command", required=True)

    list_parser = json_subparsers.add_parser("list-values", help="List unique values for attributes")
    list_parser.add_argument("inputs", nargs="*", type=Path, help="Input JSONL files (or defaults from config)")
    list_parser.add_argument("--attributes", nargs="*", help="Attributes to inspect (or defaults from config)")

    summary_parser = json_subparsers.add_parser("summarize", help="Summarize attribute distributions")
    summary_parser.add_argument("inputs", nargs="*", type=Path, help="Input JSONL files (or defaults from config)")
    summary_parser.add_argument("--attributes", nargs="*", help="Attributes to summarize (or defaults from config)")
    summary_parser.add_argument(
        "--filter",
        action="append",
        nargs="*",
        metavar="NAME=VALUE",
        help="Optional filters to count coverage for (provide multiple groups for multiple filters)",
    )

    merge_parser = json_subparsers.add_parser("merge-filtered", help="Merge filter-specific outputs")
    merge_parser.add_argument("stage", nargs="?", help="Pipeline stage name (e.g. yoloe or gemini)")
    merge_parser.add_argument(
        "--filename",
        help="Name of the file to merge from each filter directory (defaults to the stage's primary output)",
    )
    merge_parser.add_argument("--base-dir", type=Path, help="Base filters directory (or default from config)")
    merge_parser.add_argument("--output", type=Path, help="Destination merged JSONL path")

    args = parser.parse_args()

    default_config_path = DEFAULT_JSON_UTILS_CONFIG_PATH if args.command == "json-utils" else DEFAULT_CONFIG_PATH
    config_data = _load_config(args.config, default_config_path)

    log_file = _configure_logging(args.verbose, args.command)
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

    elif args.command == "check-gemini":
        config = _build_gemini_config(args, config_data, require_prompt=False)
        job_names = _load_job_names(args.job_names or [], config.submitted_jobs_file)
        statuses = get_gemini_job_status(job_names, config)
        print(json.dumps({"jobs": statuses}, indent=2))

    elif args.command == "download-gemini":
        config = _build_gemini_config(args, config_data, require_prompt=False)
        job_names = _load_job_names(args.job_names or [], config.submitted_jobs_file)
        stage_root = filtered_stage_dir("gemini", config.attribute_filters) if config.attribute_filters else Path(".")
        output_dir = _resolve_path(args.output_dir, None, stage_root / "gemini_outputs")
        cleaned_dir = _resolve_path(args.cleaned_dir, None, stage_root / "gemini_outputs_clean")

        downloaded = download_completed_jobs(job_names, output_dir, config)
        cleaned = (
            clean_gemini_logs(
                downloaded,
                cleaned_dir,
                allowed_categories=config.flag_categories,
                category_descriptions=config.flag_category_descriptions,
            )
            if downloaded
            else []
        )
        print(json.dumps({"downloaded": [str(p) for p in downloaded], "cleaned": [str(p) for p in cleaned]}, indent=2))

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

    elif args.command == "json-utils":
        json_cfg = config_data.get("json_utils", {}) if isinstance(config_data, dict) else {}

        if args.json_command == "list-values":
            defaults = json_cfg.get("list_values", {}) if isinstance(json_cfg, dict) else {}
            inputs = args.inputs or _normalize_path_list(defaults.get("inputs"))
            attributes = args.attributes or _normalize_str_list(defaults.get("attributes"))
            if not inputs:
                raise ValueError("At least one input JSONL must be provided via CLI or config file")
            if not attributes:
                raise ValueError("At least one attribute must be provided via CLI or config file")
            result = list_attribute_values(inputs, attributes)
            print(json.dumps(result, indent=2))
        elif args.json_command == "summarize":
            defaults = json_cfg.get("summarize", {}) if isinstance(json_cfg, dict) else {}
            inputs = args.inputs or _normalize_path_list(defaults.get("inputs"))
            attributes = args.attributes or _normalize_str_list(defaults.get("attributes"))
            filters_cfg = _normalize_filter_sets(defaults.get("filters")) if isinstance(defaults, dict) else None
            filter_sets = _parse_filter_sets(args.filter) or filters_cfg
            if not inputs:
                raise ValueError("At least one input JSONL must be provided via CLI or config file")
            if not attributes:
                raise ValueError("At least one attribute must be provided via CLI or config file")
            result = summarize_attributes(inputs, attributes, filter_sets)
            print(json.dumps(result, indent=2))
        elif args.json_command == "merge-filtered":
            defaults = json_cfg.get("merge_filtered", {}) if isinstance(json_cfg, dict) else {}
            stage = args.stage or (defaults.get("stage") if isinstance(defaults, dict) else None)
            if not stage:
                raise ValueError("Stage must be provided via CLI or config file")
            filename = args.filename if args.filename is not None else (defaults.get("filename") if isinstance(defaults, dict) else None)
            base_dir_raw = args.base_dir if args.base_dir is not None else (defaults.get("base_dir") if isinstance(defaults, dict) else None)
            base_dir = Path(base_dir_raw) if base_dir_raw is not None else Path("filters")
            output_raw = args.output if args.output is not None else (defaults.get("output") if isinstance(defaults, dict) else None)
            output_path = Path(output_raw) if output_raw is not None else None
            output = merge_filtered_outputs(stage, filename, base_dir, output_path)
            print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
