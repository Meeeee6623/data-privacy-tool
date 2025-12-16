import argparse
import json
import logging
from pathlib import Path

from privacy_pipeline.config import DatasetConfig, GeminiConfig, YoloEConfig
from privacy_pipeline.dataset_index import build_image_index
from privacy_pipeline.yoloe_runner import run_yoloe
from privacy_pipeline.gemini_pipeline import (
    collect_flagged_scenes,
    create_gemini_batches,
    download_completed_jobs,
    parse_gemini_output,
    submit_gemini_batches,
)


def _add_common_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("image_root", type=Path, help="Root directory containing images")
    parser.add_argument("--output", type=Path, default=Path("image_index.jsonl"))
    parser.add_argument("--recursive", default=True, action="store_true", help="Search recursively for images")
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


def _add_common_yolo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        type=Path,
        nargs="?",
        default=Path("yoloe-11l-seg.pt"),
        help="Path to YOLOE model (defaults to yoloe-11l-seg.pt)",
    )
    parser.add_argument("index", type=Path, help="Input image index JSONL")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("yoloe_output.jsonl"))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-dir", type=Path, help="Directory for YOLOE visualizations")


def _add_common_gemini_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("yolo_output", type=Path, help="YOLOE output JSONL")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--classes", nargs="*", help="YOLOE classes that trigger Gemini")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scene-level", type=int, default=1)
    parser.add_argument("--batch-dir", type=Path, default=Path("gemini_batches"))
    parser.add_argument("--max-bytes", type=float, default=1.85 * 1024 ** 3)
    parser.add_argument("--gcs-bucket", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--location", type=str, default="us-central1")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--jobs-file", type=Path, default=Path("gemini_jobs.json"))
    parser.add_argument("--final-output", type=Path, default=Path("gemini_output.jsonl"))


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
    submit_parser.add_argument("batch_dir", type=Path)
    submit_parser.add_argument("--jobs-file", type=Path, default=Path("gemini_jobs.json"))
    submit_parser.add_argument("--gcs-bucket", required=True)
    submit_parser.add_argument("--project", required=True)
    submit_parser.add_argument("--location", default="us-central1")
    submit_parser.add_argument("--model", default="gemini-2.0-flash")

    parse_parser = subparsers.add_parser("parse-gemini", help="Parse Gemini batch outputs into JSONL")
    parse_parser.add_argument("outputs", nargs="+", type=Path)
    parse_parser.add_argument("--original-index", type=Path, required=True)
    parse_parser.add_argument("--scene-level", type=int, default=1)
    parse_parser.add_argument("--final-output", type=Path, default=Path("gemini_output.jsonl"))

    args = parser.parse_args()

    log_file = _configure_logging(args.verbose)
    logging.getLogger(__name__).info("Logs will be written to %s", log_file)

    if args.command == "index":
        config = DatasetConfig(
            image_root=args.image_root,
            recursive=args.recursive,
            path_attributes=args.path_attributes,
            path_attribute_map=_parse_attribute_map(args.path_attribute_map),
            user_jsonl=args.user_jsonl,
            output_jsonl=args.output,
        )
        output = build_image_index(config)
        print(f"Wrote image index to {output}")

    elif args.command == "yoloe":
        config = YoloEConfig(
            model_path=args.model,
            threshold=args.threshold,
            visualize=args.visualize,
            visualization_dir=args.viz_dir,
            output_jsonl=args.output,
        )
        output = run_yoloe(args.index, config)
        print(f"Wrote YOLOE output to {output}")

    elif args.command == "prepare-gemini":
        config = GeminiConfig(
            prompt=args.prompt,
            classes_to_forward=args.classes,
            min_confidence=args.threshold,
            scene_directory_level=args.scene_level,
            max_batch_size_bytes=args.max_bytes,
            output_batch_dir=args.batch_dir,
            gcs_bucket=args.gcs_bucket,
            submitted_jobs_file=args.jobs_file,
            project=args.project,
            location=args.location,
            model=args.model,
            final_output_jsonl=args.final_output,
        )
        flagged = collect_flagged_scenes(args.yolo_output, config)
        batch_files = create_gemini_batches(flagged, config)
        print(json.dumps({"flagged_scenes": len(flagged), "batch_files": [str(p) for p in batch_files]}, indent=2))

    elif args.command == "submit-gemini":
        batch_files = sorted(args.batch_dir.glob("*.jsonl"))
        config = GeminiConfig(
            prompt="",
            classes_to_forward=None,
            gcs_bucket=args.gcs_bucket,
            submitted_jobs_file=args.jobs_file,
            project=args.project,
            location=args.location,
            model=args.model,
            scene_directory_level=1,
            min_confidence=0.0,
            max_batch_size_bytes=1.0,
            output_batch_dir=args.batch_dir,
        )
        job_names = submit_gemini_batches(batch_files, config)
        print(json.dumps({"submitted_jobs": job_names}, indent=2))

    elif args.command == "parse-gemini":
        config = GeminiConfig(
            prompt="",
            classes_to_forward=None,
            scene_directory_level=args.scene_level,
            final_output_jsonl=args.final_output,
            min_confidence=0.0,
            max_batch_size_bytes=1.0,
            output_batch_dir=Path(),
        )
        output = parse_gemini_output(args.outputs, args.original_index, config)
        print(f"Wrote parsed Gemini output to {output}")


if __name__ == "__main__":
    main()
