import base64
import importlib.util
import json
import logging
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TypedDict

from google import genai
from google.genai import types
from google.genai.types import CreateBatchJobConfig
from google.cloud import storage

from privacy_pipeline.config import GeminiConfig

ORJSON_AVAILABLE = importlib.util.find_spec("orjson") is not None
if ORJSON_AVAILABLE:
    import orjson  # type: ignore


logger = logging.getLogger(__name__)


FLAG_PATTERN = re.compile(r"#FLAG\s*\[?([^\]\n]+)\]?")
DEFAULT_FLAG_CATEGORIES = {
    "PII",
    "CONFIDENTIAL_INFO",
    "SECURITY_INFO",
    "MACHINE_READABLE_CODE",
    "BRANDING_LOGOS",
    "OTHER",
}


class GeminiJobRecord(TypedDict):
    name: str
    output_uri: Optional[str]


def _load_yolo_output(path: Path) -> Iterable[Dict]:
    logger.debug("Loading YOLOE output from %s", path)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _load_json_line(line: str) -> Dict:
    if ORJSON_AVAILABLE:
        return orjson.loads(line)
    return json.loads(line)


def _dump_json_line(record: Dict) -> bytes:
    if ORJSON_AVAILABLE:
        return orjson.dumps(record)
    return json.dumps(record).encode("utf-8")


def _matches_filters(attributes: Dict, filters: Dict[str, str] | None) -> bool:
    if not filters:
        return True
    return all(attributes.get(key) == value for key, value in filters.items())


def _strip_inline_images(request: Dict | None) -> Dict | None:
    if not isinstance(request, dict):
        return request

    cleaned = dict(request)
    cleaned_contents = []
    for content in request.get("contents", []):
        if not isinstance(content, dict):
            cleaned_contents.append(content)
            continue

        cleaned_parts = []
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                cleaned_parts.append(part)
                continue

            if "inlineData" in part and isinstance(part["inlineData"], dict):
                inline_data = dict(part["inlineData"])
                inline_data.pop("data", None)
                cleaned_part = dict(part)
                cleaned_part["inlineData"] = inline_data
                cleaned_parts.append(cleaned_part)
            else:
                cleaned_parts.append(part)

        cleaned_content = dict(content)
        if cleaned_parts:
            cleaned_content["parts"] = cleaned_parts
        cleaned_contents.append(cleaned_content)

    if cleaned_contents:
        cleaned["contents"] = cleaned_contents
    return cleaned


def _extract_response_text(response: Dict | None) -> str | None:
    if not isinstance(response, dict):
        return None
    candidates = response.get("candidates")
    if not candidates:
        return None
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return None
    content = candidate.get("content", {})
    if not isinstance(content, dict):
        return None
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        return None
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            return part.get("text")
    return None


def _normalize_flag_categories(categories: Iterable[str] | None) -> set[str]:
    if not categories:
        return set()
    normalized = set()
    for category in categories:
        normalized_name = str(category).strip().upper().replace(" ", "_")
        if normalized_name:
            normalized.add(normalized_name)
    return normalized


def _extract_flag_categories(text: str | None, allowed_categories: Optional[set[str]] = None) -> list[str]:
    if not text:
        return []
    match = FLAG_PATTERN.search(text)
    if not match:
        return []
    raw_categories = re.split(r"[\s,]+", match.group(1))
    categories: list[str] = []
    normalized_allowed = allowed_categories or _normalize_flag_categories(DEFAULT_FLAG_CATEGORIES)
    for category in raw_categories:
        normalized = category.strip().upper().replace(" ", "_")
        if not normalized:
            continue
        if normalized_allowed and normalized not in normalized_allowed:
            continue
        categories.append(normalized)
    return categories


def _scene_root(image_path: Path, levels_up: int) -> Path:
    levels_up = max(levels_up, 1)
    current = image_path
    for _ in range(levels_up):
        current = current.parent
    return current


def _load_scene_attributes(original_index: Optional[Path], config: GeminiConfig) -> Dict[str, Dict]:
    if not original_index or not original_index.exists():
        return {}

    scene_to_attributes: Dict[str, Dict] = {}
    for record in _load_yolo_output(original_index):
        if not _matches_filters(record.get("attributes", {}), config.attribute_filters):
            continue
        scene_path = str(_scene_root(Path(record["image_path"]), config.scene_directory_level))
        scene_to_attributes.setdefault(scene_path, record.get("attributes", {}))
    return scene_to_attributes


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


def _ensure_bucket(storage_client: storage.Client, bucket_name: str, location: str) -> storage.Bucket:
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        bucket = storage_client.create_bucket(bucket, location=location)
        logger.info("Created GCS bucket %s in %s", bucket.name, location)
    return bucket


def _build_output_uri(config: GeminiConfig, batch_file: Path) -> str:
    prefix = (config.gcs_output_prefix or "").strip("/")
    unique_suffix = uuid.uuid4().hex
    relative_path = f"{batch_file.stem}-{unique_suffix}"
    if prefix:
        relative_path = f"{prefix}/{relative_path}"
    if not config.gcs_output_bucket:
        raise ValueError("gcs_output_bucket must be set on GeminiConfig to submit jobs")
    return f"gs://{config.gcs_output_bucket}/{relative_path}"


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    bucket_and_path = uri[len("gs://") :]
    parts = bucket_and_path.split("/", 1)
    bucket_name = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    return bucket_name, path


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


def submit_gemini_batches(batch_files: List[Path], config: GeminiConfig) -> List[GeminiJobRecord]:
    if not config.gcs_bucket:
        raise ValueError("gcs_bucket must be set on GeminiConfig to submit jobs")
    if not config.gcs_output_bucket:
        raise ValueError("gcs_output_bucket must be set on GeminiConfig to submit jobs")

    storage_client = storage.Client()
    input_bucket = _ensure_bucket(storage_client, config.gcs_bucket, config.location)
    _ensure_bucket(storage_client, config.gcs_output_bucket, config.location)

    client = genai.Client(vertexai=True, project=config.project, location=config.location)

    job_records: List[GeminiJobRecord] = []
    for batch_file in batch_files:
        blob = input_bucket.blob(batch_file.name)
        blob.upload_from_filename(str(batch_file))
        blob_uri = f"gs://{input_bucket.name}/{blob.name}"
        output_uri = _build_output_uri(config, batch_file)

        job = client.batches.create(
            model=config.model,
            src=blob_uri,
            config=CreateBatchJobConfig(display_name=batch_file.stem, dest=output_uri),
        )
        job_records.append({"name": job.name, "output_uri": output_uri})
        logger.info(
            "Submitted Gemini batch %s as job %s (output -> %s)",
            batch_file.name,
            job.name,
            output_uri,
        )

    config.submitted_jobs_file.write_text(json.dumps(job_records, indent=2))
    logger.info("Recorded %d submitted jobs to %s", len(job_records), config.submitted_jobs_file)
    return job_records


def get_gemini_job_status(job_records: List[GeminiJobRecord], config: GeminiConfig) -> List[Dict]:
    if not job_records:
        return []

    client = genai.Client(vertexai=True, project=config.project, location=config.location)
    statuses: List[Dict] = []
    for job_record in job_records:
        job = client.batches.get(name=job_record["name"])
        statuses.append(
            {
                "name": job.name,
                "state": getattr(job, "state", None),
                "display_name": getattr(job, "display_name", None),
                "output_uri": job_record.get("output_uri"),
                "error": getattr(job, "error", None),
            }
        )
    logger.info("Retrieved status for %d Gemini jobs", len(statuses))
    return statuses


def download_completed_jobs(job_records: List[GeminiJobRecord], destination_dir: Path, config: GeminiConfig) -> List[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[Path] = []
    if not job_records:
        return downloaded

    storage_client = storage.Client()
    for job_record in job_records:
        output_uri = job_record.get("output_uri")
        if not output_uri:
            raise ValueError(f"No output_uri recorded for job {job_record['name']}")
        bucket_name, path = _split_gcs_uri(output_uri)
        bucket = storage_client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=path):
            local_path = destination_dir / Path(blob.name).name
            blob.download_to_filename(local_path)
            downloaded.append(local_path)
            logger.debug("Downloaded output blob %s to %s", blob.name, local_path)
    logger.info("Downloaded %d completed job outputs to %s", len(downloaded), destination_dir)
    return downloaded


def clean_and_merge_gemini_logs(
    batch_outputs: List[Path],
    merged_output: Path,
    config: GeminiConfig,
    original_index: Optional[Path] = None,
    allowed_categories: Optional[Iterable[str]] = None,
) -> Path:
    merged_output.parent.mkdir(parents=True, exist_ok=True)

    normalized_allowed = _normalize_flag_categories(allowed_categories)
    if not normalized_allowed:
        normalized_allowed = _normalize_flag_categories(DEFAULT_FLAG_CATEGORIES)

    scene_attributes = _load_scene_attributes(original_index, config)

    def _clean_record(record: Dict | None) -> Dict:
        request = record.get("request") if isinstance(record, dict) else None
        response = record.get("response") if isinstance(record, dict) else None

        response_text = _extract_response_text(response)
        categories = _extract_flag_categories(response_text, normalized_allowed)
        return {
            "scene_path": record.get("key") if isinstance(record, dict) else None,
            "response_text": response_text,
            "flag_categories": categories,
            "is_flagged": bool(FLAG_PATTERN.search(response_text or "")),
            "request": _strip_inline_images(request),
        }

    with merged_output.open("wb") as outfile:
        cleaned_count = 0
        for batch_output in batch_outputs:
            if not batch_output.exists():
                logger.warning("Batch output %s does not exist; skipping", batch_output)
                continue
            with batch_output.open() as infile:
                for line in infile:
                    if not line.strip():
                        continue

                    record = _load_json_line(line)
                    cleaned_record = _clean_record(record)
                    scene_path = cleaned_record.get("scene_path")
                    attributes = scene_attributes.get(scene_path, {}) if scene_path else {}

                    merged_record = {
                        "scene_path": scene_path,
                        "attributes": attributes,
                        "response_text": cleaned_record.get("response_text"),
                        "flag_categories": cleaned_record.get("flag_categories", []),
                        "is_flagged": cleaned_record.get("is_flagged", False),
                    }

                    request = cleaned_record.get("request")
                    if request:
                        merged_record["request"] = request

                    outfile.write(_dump_json_line(merged_record) + b"\n")
                    cleaned_count += 1

    logger.info("Merged and cleaned %d Gemini records into %s", cleaned_count, merged_output)
    return merged_output


def parse_gemini_output(batch_outputs: List[Path], original_index: Path, config: GeminiConfig) -> Path:
    scene_to_attributes = _load_scene_attributes(original_index, config)

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
