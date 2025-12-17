import base64
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from google import genai
from google.cloud import storage

from privacy_pipeline.config import GeminiConfig

logger = logging.getLogger(__name__)


def _load_yolo_output(path: Path) -> Iterable[Dict]:
    logger.debug("Loading YOLOE output from %s", path)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _matches_filters(attributes: Dict, filters: Dict[str, str] | None) -> bool:
    if not filters:
        return True
    return all(attributes.get(key) == value for key, value in filters.items())


def _scene_root(image_path: Path, levels_up: int) -> Path:
    levels_up = max(levels_up, 1)
    current = image_path
    for _ in range(levels_up):
        current = current.parent
    return current


def collect_flagged_scenes(yolo_output: Path, config: GeminiConfig) -> Dict[Path, List[str]]:
    all_images: Dict[Path, List[str]] = defaultdict(list)
    flagged_scenes: set[Path] = set()

    for record in _load_yolo_output(yolo_output):
        if not _matches_filters(record.get("attributes", {}), config.attribute_filters):
            continue
        image_path = Path(record["image_path"])
        scene_path = _scene_root(image_path, config.scene_directory_level)
        all_images[scene_path].append(str(image_path))

        for detection in record.get("detections", []):
            meets_class = not config.classes_to_forward or detection["class"] in config.classes_to_forward
            meets_conf = detection.get("confidence", 0.0) >= config.min_confidence
            if meets_class and meets_conf:
                flagged_scenes.add(scene_path)
                break

    logger.info("Flagged %d scenes for Gemini processing", len(flagged_scenes))
    return {scene: sorted(all_images[scene]) for scene in flagged_scenes}


def _encode_image(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_record(scene: Path, images: List[str], prompt: str) -> str:
    parts = [{"text": prompt}]
    for image in images:
        parts.append({"inlineData": {"mime_type": "image/png", "data": _encode_image(Path(image))}})

    record = {
        "key": str(scene),
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        },
    }
    return json.dumps(record)


def create_gemini_batches(flagged_scenes: Dict[Path, List[str]], config: GeminiConfig) -> List[Path]:
    config.output_batch_dir.mkdir(parents=True, exist_ok=True)
    batch_files: List[Path] = []
    current_records: List[str] = []
    current_size = 0

    for idx, (scene, images) in enumerate(flagged_scenes.items()):
        record = _build_record(scene, images, config.prompt)
        record_size = len(record) + 1
        would_exceed = current_size + record_size > config.max_batch_size_bytes
        if would_exceed and current_records:
            batch_path = config.output_batch_dir / f"batch_{len(batch_files)}.jsonl"
            with batch_path.open("w") as f:
                f.write("\n".join(current_records) + "\n")
            batch_files.append(batch_path)
            current_records = []
            current_size = 0

        current_records.append(record)
        current_size += record_size

    if current_records:
        batch_path = config.output_batch_dir / f"batch_{len(batch_files)}.jsonl"
        with batch_path.open("w") as f:
            f.write("\n".join(current_records) + "\n")
        batch_files.append(batch_path)

    logger.info("Created %d Gemini batch files in %s", len(batch_files), config.output_batch_dir)
    return batch_files


def submit_gemini_batches(batch_files: List[Path], config: GeminiConfig) -> List[str]:
    if not config.gcs_bucket:
        raise ValueError("gcs_bucket must be set on GeminiConfig to submit jobs")

    storage_client = storage.Client()
    bucket = storage_client.bucket(config.gcs_bucket)
    if not bucket.exists():
        bucket = storage_client.create_bucket(bucket, location=config.location)
        logger.info("Created GCS bucket %s in %s", bucket.name, config.location)

    client = genai.Client(vertexai=True, project=config.project, location=config.location)

    job_names: List[str] = []
    for batch_file in batch_files:
        blob = bucket.blob(batch_file.name)
        blob.upload_from_filename(str(batch_file))
        blob_uri = f"gs://{bucket.name}/{blob.name}"

        job = client.batches.create(
            model=config.model,
            src=blob_uri,
            config={"display_name": batch_file.stem},
        )
        job_names.append(job.name)
        logger.info("Submitted Gemini batch %s as job %s", batch_file.name, job.name)

    config.submitted_jobs_file.write_text(json.dumps(job_names, indent=2))
    logger.info("Recorded %d submitted jobs to %s", len(job_names), config.submitted_jobs_file)
    return job_names


def get_gemini_job_status(job_names: List[str], config: GeminiConfig) -> List[Dict]:
    if not job_names:
        return []

    client = genai.Client(vertexai=True, project=config.project, location=config.location)
    statuses: List[Dict] = []
    for job_name in job_names:
        job = client.batches.get(name=job_name)
        statuses.append(
            {
                "name": job.name,
                "state": getattr(job, "state", None),
                "display_name": getattr(job, "display_name", None),
                "output_uri": getattr(job, "output_output_gcs_uri", None),
                "error": getattr(job, "error", None),
            }
        )
    logger.info("Retrieved status for %d Gemini jobs", len(statuses))
    return statuses


def download_completed_jobs(job_names: List[str], destination_dir: Path, config: GeminiConfig) -> List[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(vertexai=True, project=config.project, location=config.location)

    downloaded: List[Path] = []
    for job_name in job_names:
        job = client.batches.get(name=job_name)
        if not job.output_output_gcs_uri:
            continue
        output_uri = job.output_output_gcs_uri
        storage_client = storage.Client()
        bucket_name, path = output_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=path):
            local_path = destination_dir / Path(blob.name).name
            blob.download_to_filename(local_path)
            downloaded.append(local_path)
            logger.debug("Downloaded output blob %s to %s", blob.name, local_path)
    logger.info("Downloaded %d completed job outputs to %s", len(downloaded), destination_dir)
    return downloaded


def parse_gemini_output(batch_outputs: List[Path], original_index: Path, config: GeminiConfig) -> Path:
    scene_to_attributes: Dict[str, Dict] = {}
    for record in _load_yolo_output(original_index):
        if not _matches_filters(record.get("attributes", {}), config.attribute_filters):
            continue
        scene_path = str(_scene_root(Path(record["image_path"]), config.scene_directory_level))
        scene_to_attributes.setdefault(scene_path, record.get("attributes", {}))

    parsed: List[Dict] = []
    for batch_output in batch_outputs:
        for record in _load_yolo_output(batch_output):
            key = record.get("key")
            result = record.get("result", {})
            contents = result.get("contents", []) if isinstance(result, dict) else []
            if key in scene_to_attributes:
                parsed.append(
                    {
                        "scene_path": key,
                        "attributes": scene_to_attributes.get(key, {}),
                        "gemini_response": contents,
                    }
                )

    logger.info("Parsed %d Gemini batch outputs", len(parsed))

    config.final_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.final_output_jsonl.open("w") as f:
        for record in parsed:
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote parsed Gemini output to %s", config.final_output_jsonl)
    return config.final_output_jsonl
