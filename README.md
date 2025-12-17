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

# 2) Run YOLOE (uses yoloe-11l-seg.pt and packaged classes by default)
python -m privacy_pipeline.cli yoloe image_index.jsonl \
  --threshold 0.5 --visualize --viz-dir yoloe_viz \
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
```

### YAML configuration

All CLI options can be provided via a YAML file and overridden by explicit CLI
flags. The CLI will automatically load `config.yaml` from the current working
directory when present. You can also point to any file explicitly with
`--config path/to/config.yaml`.

An exhaustive example is provided at `config.example.yaml`; copy it to
`config.yaml` and edit it to match your environment.

Example:

```yaml
dataset:
  image_root: /path/to/images
  recursive: true
  path_attributes: [lab, building, scene]
  path_attribute_map: {lab: 3, building: 2, scene: 1}
  user_jsonl: /path/to/custom_attributes.jsonl
  output_jsonl: image_index.jsonl

yoloe:
  model_path: yoloe-11l-seg.pt
  threshold: 0.5
  visualize: false
  visualization_dir: yoloe_visualizations
  output_jsonl: yoloe_output.jsonl
  attribute_filters: {lab: example-lab}

gemini:
  prompt: "<your prompt>"
  classes_to_forward: [person, screen]
  min_confidence: 0.5
  scene_directory_level: 2
  max_batch_size_bytes: 1981808640
  output_batch_dir: gemini_batches
  gcs_bucket: your-bucket
  project: your-gcp-project
  submitted_jobs_file: gemini_jobs.json
  final_output_jsonl: gemini_output.jsonl
```

With this file in place, running `python -m privacy_pipeline.cli prepare-gemini`
loads defaults from `config.yaml` automatically and only needs CLI overrides
for values that should differ from the file.

The YOLOE runner automatically downloads the requested checkpoint (defaulting to
`yoloe-11l-seg.pt`), remaps the classes from `privacy_pipeline/yoloe_classes.txt`
to match `yoloe_test.ipynb`, saves the customized weights alongside the original
model file, and writes a `yoloe_custom_mapping.txt` next to the YOLOE output.

Notes:
- `--scene-level` controls how far up the directory tree to group images into a
  “scene.”
- Use `--path-attribute-map` to name directories relative to the file (e.g.
  `lab:3` means three levels up from the image path is labeled `lab`). This can
  be combined with `--path-attributes` for root-relative naming.
- When Gemini is triggered by any image in a scene, **all images in that scene**
  are sent for OCR/classification.
- `--max-bytes` can be adjusted if GCP batch limits change (default 1.85 GB).
