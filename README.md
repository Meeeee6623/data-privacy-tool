# data-privacy-tool
Tool to screen datasets for explicit private data leakage.

## Overview
The new `privacy_pipeline` package turns the legacy YOLOE + Gemini scripts into a
single, configurable tool. It supports arbitrary image datasets (e.g., DROID
lab/building/scene hierarchies or any custom structure) and keeps metadata in a
consistent JSONL format between steps.

Key steps:

1. **Index images** – Glob files from a root directory (recursive optional),
   derive attributes from the directory structure, and/or merge custom user
   metadata JSONL files. Output is a JSONL file containing `image_path` and an
   `attributes` dictionary for each image.
2. **Run YOLOE** – Read the index JSONL, run the YOLOE detector with
   configurable thresholds and class lists, optionally save visualizations, and
   write detections back to JSONL.
3. **Prepare & submit Gemini batches** – Choose which YOLOE classes/confidences
   trigger Gemini. When any image in a scene is flagged, all images from that
   scene (configurable directory level) are bundled into a Gemini batch request.
   Batches are automatically split below 1.85 GB and can be uploaded/submitted
   via the CLI. Results are parsed back into a JSONL that maps each scene to the
   Gemini OCR output.

## CLI usage
Commands are provided via `python -m privacy_pipeline.cli`:

```bash
# 1) Build an image index
python -m privacy_pipeline.cli index /path/to/images \
  --output image_index.jsonl \
  --recursive \
  --path-attributes lab building scene \
  --path-attribute-map lab:3 building:2 scene:1

# 2) Run YOLOE
python -m privacy_pipeline.cli yoloe path/to/model.pt image_index.jsonl \
  --classes yoloe_classes.txt --threshold 0.5 --visualize --viz-dir yoloe_viz \
  --output yoloe_output.jsonl

# 3) Prepare Gemini batches
python -m privacy_pipeline.cli prepare-gemini yoloe_output.jsonl \
  --prompt "<your prompt>" \
  --classes person screen --threshold 0.5 --scene-level 2 \
  --batch-dir gemini_batches --gcs-bucket your-bucket --project your-gcp-project

# 4) Submit batches (requires GCP access)
python -m privacy_pipeline.cli submit-gemini gemini_batches \
  --gcs-bucket your-bucket --project your-gcp-project

# 5) Parse Gemini outputs
python -m privacy_pipeline.cli parse-gemini /path/to/output/*.jsonl \
  --original-index image_index.jsonl --scene-level 2 \
  --final-output gemini_output.jsonl

# 6) Build HTML viewers
# YOLOE viewer (optionally expand scenes using the original index)
python -m privacy_pipeline.cli visualize-yoloe yoloe_output.jsonl \
  --scene-level 2 --index-jsonl image_index.jsonl --output-dir viewers/yoloe

# Gemini OCR viewer (scenes only, can source scene images from the index or YOLOE output)
python -m privacy_pipeline.cli visualize-gemini gemini_output.jsonl \
  --scene-level 2 --index-jsonl image_index.jsonl --output-dir viewers/gemini
```

Notes:
- `--scene-level` controls how far up the directory tree to group images into a
  “scene.”
- Use `--path-attribute-map` to name directories relative to the file (e.g.
  `lab:3` means three levels up from the image path is labeled `lab`). This can
  be combined with `--path-attributes` for root-relative naming.
- When Gemini is triggered by any image in a scene, **all images in that scene**
  are sent for OCR/classification.
- `--max-bytes` can be adjusted if GCP batch limits change (default 1.85 GB).
- YOLOE viewer: supports grouping by scene, filtering by class/confidence/any
  custom attributes, and toggling visualization vs. raw images. Click through a
  thumbnail to view every frame from that scene. Grid and page sizes are
  adjustable in the UI.
- Gemini OCR viewer: grids are scene-based with filters for categories and
  custom attributes. You can toggle OCR previews under thumbnails and click a
  scene to see every frame alongside the OCR text and detected categories.
