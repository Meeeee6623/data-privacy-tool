from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """Configuration describing how to load image metadata."""

    image_root: Path
    recursive: bool = True
    path_attributes: Optional[List[str]] = None
    # Optional mapping of attribute name -> levels up from the image file (1 = parent directory)
    path_attribute_map: Optional[Dict[str, int]] = None
    user_jsonl: Optional[Path] = None
    output_jsonl: Path = Path("image_index.jsonl")


@dataclass
class YoloEConfig:
    """Configuration for running YOLOE inference."""

    model_path: Path = Path("yoloe-11l-seg.pt")
    classes_path: Path = Path(__file__).resolve().parent / "yoloe_classes.txt"
    threshold: float = 0.5
    visualize: bool = False
    visualization_dir: Optional[Path] = None
    output_jsonl: Path = Path("yoloe_output.jsonl")
    attribute_filters: Optional[Dict[str, str]] = None
    dataset_image_root: Optional[Path] = None


@dataclass
class GeminiConfig:
    """Configuration for Gemini / VLLM OCR processing."""

    prompt: str
    classes_to_forward: Optional[List[str]] = None
    min_confidence: float = 0.5
    scene_directory_level: int = 1
    max_batch_size_bytes: float = 1.85 * 1024 ** 3
    output_batch_dir: Path = Path("gemini_batches")
    gcs_bucket: Optional[str] = None
    gcs_output_bucket: Optional[str] = None
    gcs_output_prefix: str = "gemini_outputs"
    submitted_jobs_file: Path = Path("gemini_jobs.json")
    final_output_jsonl: Path = Path("gemini_output.jsonl")
    project: Optional[str] = None
    location: str = "us-central1"
    model: str = "gemini-2.0-flash"
    attribute_filters: Optional[Dict[str, str]] = None


@dataclass
class PipelineConfig:
    dataset: DatasetConfig
    yoloe: YoloEConfig
    gemini: GeminiConfig
    extra_metadata: Dict[str, str] = field(default_factory=dict)
