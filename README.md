# Data Privacy Tool
Pipeline for screening large image corpora for sensitive data before publication. The tool chains YOLOE detections with Gemini OCR to surface PII, internal documents, barcodes, and other risky content, and ships ready-made notebooks for visual validation.

## Goals & Highlights
- Normalize any folder hierarchy into a JSONL index so downstream stages share the same metadata.
- Run YOLOE detections with customizable confidence thresholds, class remapping, attribute filters, and optional visualization exports.
- Escalate risky “scenes” (all images in a directory level) to Gemini with a single prompt, batching automatically under GCP limits and resuming from checkpoints.
- Track, download, clean, and enrich Gemini outputs so analysts can focus on flagged categories across labs/buildings/scenes.
- Explore detections and OCR results interactively in the included Plotly + ipywidgets notebooks.
- Keep all artifacts in `output/` (or `output/filtered/<stage>/<slug>` when filters are used) so every run is reproducible.

---

## Requirements & Installation
1. **Python 3.13+** with GPU-enabled PyTorch if you intend to run YOLOE on GPU.
2. **Google Cloud credentials** with access to Vertex Gemini and the buckets you plan to use (`GOOGLE_APPLICATION_CREDENTIALS` works well locally).
3. Install dependencies inside your preferred virtual environment:
   ```bash
   uv sync  # or: pip install -e .
   ```
   The `pyproject.toml` declares `ultralytics`, `torch`, `google-genai`, `google-cloud-storage`, `pandas`, `plotly`, `ipywidgets`, and Jupyter.
4. Optional: download the default YOLOE checkpoint into `models/` ahead of time to avoid on-demand fetches.

All commands are executed via `python -m privacy_pipeline.cli ...`. The CLI automatically loads `config/config.yaml` when present (override with `--config path/to/config.yaml`).

---

## Pipeline at a Glance
```
images -> index -> yoloe -> prepare gemini -> submit -> check -> download -> clean -> notebooks/json-utils
```
1. **Index**: glob images, derive attributes from directory structure, merge user metadata.
2. **YOLOE**: detect configured classes, optionally emit overlay PNGs.
3. **Prepare Gemini**: determine scenes needing review (class + confidence filters) and create <=1.85 GB batch JSONLs.
4. **Submit & Check**: upload batches to GCS, create Vertex Gemini jobs, and poll their status.
5. **Download**: fetch job outputs and track them alongside submissions.
6. **Clean**: strip inline payloads, parse `#FLAG[...]` markers, attach scene attributes, and write clean JSONL.
7. **Explore**: use notebooks or `json-utils` to analyze detections/flags, merge filtered runs, or summarize coverage.

Quick start run (assuming `config/config.yaml` mirrors your environment):
```bash
python -m privacy_pipeline.cli index /data/images
python -m privacy_pipeline.cli yoloe
python -m privacy_pipeline.cli prepare-gemini --prompt "$(cat prompts/gemini_prompt.txt)"
python -m privacy_pipeline.cli submit-gemini
python -m privacy_pipeline.cli check-gemini
python -m privacy_pipeline.cli download-gemini
python -m privacy_pipeline.cli clean-gemini --original-index output/image_index.jsonl
```

---

## Pipeline Walkthrough

### 1. Build the dataset index
**Purpose**: turn an arbitrary folder hierarchy into a structured JSONL (`image_path`, `attributes`). Supports optional recursion, attribute derivation, and user-provided metadata merges.

```bash
python -m privacy_pipeline.cli index /data/images \
  --output output/image_index.jsonl \
  --recursive \
  --path-attributes lab building scene \
  --path-attribute-map lab:3 building:2 scene:1 \
  --user-jsonl data/manual_annotations.jsonl
```

Key configuration (`dataset` in `config.yaml`):
- `image_root` (required): root folder to crawl.
- `recursive`: include subdirectories (default `true`).
- `path_attributes`: attribute names mapped to directory segments relative to the root.
- `path_attribute_map`: attribute → “levels up from file” (e.g., `lab:3`).
- `user_jsonl`: optional JSONL with additional `image_path` + `attributes`.
- `output_jsonl`: where the index is written (`output/image_index.jsonl` by default).

Each record only stores metadata; pixel data stays on disk. The notebooks rely on this file for lookups, so keep it around even after cleaning.

### 2. Run YOLOE detections
**Purpose**: detect potentially sensitive classes before escalating to Gemini. The runner loads `models/yoloe-11l-seg.pt`, remaps classes listed in `privacy_pipeline/yoloe_classes.txt`, writes the mapping to `config/yoloe_custom_mapping.txt`, and only records images with detections to keep JSONL sizes manageable.

```bash
python -m privacy_pipeline.cli yoloe output/image_index.jsonl \
  --threshold 0.45 \
  --visualize \
  --viz-dir output/visualizations/yoloe \
  --attribute-filter lab=alpha building=west
```

Key configuration (`yoloe`):
- `model_path`: checkpoint to load (auto-downloaded if missing).
- `threshold`: minimum confidence (default `0.5`).
- `visualize`, `visualization_dir`: save overlay PNGs for detections only.
- `output_jsonl`: defaults to `output/yoloe_output.jsonl` or `output/filtered/yoloe/<slug>/...` when attribute filters are applied.
- `attribute_filters`: restrict which index records are processed. Filtered runs keep their own JSONL and visualization directory so you can run multiple slices in parallel.

**Notebook tie-in**: use `notebooks/yoloe_stage_explorer.ipynb` to slice detections by attribute/class, review Plotly charts, and preview top matches before adjusting thresholds. The notebook auto-discovers base outputs plus anything under `output/filtered/yoloe/*`.

### 3. Prepare Gemini batches
**Purpose**: determine which “scenes” (folders defined by `scene_directory_level`) should be escalated to OCR, bundle their images, and emit JSONL batches sized for Vertex uploads.

```bash
python -m privacy_pipeline.cli prepare-gemini \
  --yolo_output output/yoloe_output.jsonl \
  --prompt "$(cat prompts/gemini_prompt.txt)" \
  --classes person screen \
  --threshold 0.5 \
  --scene-level 2 \
  --batch-dir output/gemini/batches \
  --attribute-filter lab=alpha
```

Key configuration (`gemini`):
- `prompt`: full Gemini instruction block (multi-line YAML string supported).
- `classes_to_forward` + `min_confidence`: detections that trigger escalation.
- `scene_directory_level`: how many directory levels to climb to form a scene (1 = parent folder).
- `max_batch_size_bytes`: default 1.85 GB cap so uploads stay under Vertex thresholds.
- `output_batch_dir`: typically `output/gemini/batches`, or a filter-specific directory when `attribute_filters` are provided.

For every scene that contains at least one qualifying detection, **all images in that scene are sent** so Gemini sees full context.

### 4. Submit Gemini jobs
**Purpose**: upload batch files to GCS, create Vertex Gemini batch jobs, and persist job metadata for later commands.

```bash
python -m privacy_pipeline.cli submit-gemini \
  --batch-dir output/gemini/batches \
  --gcs-bucket privacy-pipeline-inputs \
  --gcs-output-bucket privacy-pipeline-outputs \
  --gcs-output-prefix gemini_outputs \
  --project my-gcp-project \
  --location us-central1 \
  --model gemini-2.5-flash
```

Key configuration fields:
- `gcs_bucket`: uploads batches here (created automatically if missing).
- `gcs_output_bucket` + `gcs_output_prefix`: where Vertex writes job results.
- `project`, `location`, `model`: Vertex deployment you want to use.
- `submitted_jobs_file`: JSON tracking job name + output URI (`output/gemini/gemini_jobs.json` or `output/filtered/gemini/<slug>/gemini_jobs.json`).

### 5. Monitor batch status
Check job health before downloading outputs. You can pass explicit job names or rely on the jobs file.

```bash
python -m privacy_pipeline.cli check-gemini --jobs-file output/gemini/gemini_jobs.json
python -m privacy_pipeline.cli check-gemini job-123 job-456 --project my-gcp-project
```

The command prints JSON summaries (`state`, `display_name`, `error`, `output_uri`). Use this to wait until jobs reach `SUCCEEDED` before downloading.

### 6. Download Gemini outputs
Pull the JSON lines emitted by Vertex from the output bucket. Filenames are disambiguated by job name so reruns do not overwrite each other.

```bash
python -m privacy_pipeline.cli download-gemini \
  --jobs-file output/gemini/gemini_jobs.json \
  --output-dir output/gemini/gemini_outputs
```

Downloaded files are typically named `prediction_<job>_<chunk>.jsonl`.

### 7. Clean and merge Gemini logs
**Purpose**: remove inline image payloads, extract `#FLAG[...]` markers, attach attributes from the original index, and emit a lean JSONL for analysis.

```bash
python -m privacy_pipeline.cli clean-gemini \
  --outputs-dir output/gemini/gemini_outputs \
  --output output/gemini_output_cleaned.jsonl \
  --original-index output/image_index.jsonl
```

What happens here:
- `response_text` captures Gemini’s combined text / explanation.
- `flag_categories` is derived from the categories listed inside `#FLAG[...]`.
- `scene_path` and `attributes` come from the original index (respecting attribute filters if provided).
- `is_flagged` is set when a `#FLAG` marker exists.

The cleaned JSONL is the primary input for the Gemini explorer notebook and any downstream metrics.

### 8. Inspect results
Options once the cleaned files exist:
- **Visualization notebooks**:
  - `notebooks/yoloe_stage_explorer.ipynb`: Filter detections by attribute/class/confidence, visualize Plotly bar charts (class distribution, stacked attribute breakdowns, confidence histograms), and preview sample detections or per-scene galleries. All inputs respect the same widgets so what you see in charts mirrors the preview rows.
  - `notebooks/gemini_stage_explorer.ipynb`: Work with cleaned OCR outputs. Widgets cover attributes, flag categories, scene status, and keyword searches. The notebook provides totals, category charts, stacked bar views by any attribute, a scene table, and a detail pane that shows Gemini’s response text plus inline thumbnails for the first few images in that scene (if the index is available). Previous/Next buttons help analysts page through scenes without re-running filters.
- **JSON utilities** (`python -m privacy_pipeline.cli json-utils ...`):
  - `list-values`: list unique attribute values across one or more JSONL files.
  - `summarize`: count how many records fall into each attribute bucket and how filters would partition them.
  - `merge-filtered`: combine outputs from `output/filtered/<stage>/<slug>` back into a single JSONL when you have run multiple attribute slices independently.

You can keep defaults for these utilities inside `config/json_utils.config.yaml` so analysts run short commands such as `python -m privacy_pipeline.cli json-utils summarize`.

---

## Configuration Files
Copy the exhaustive example to get started:
```bash
cp config/config.example.yaml config/config.yaml
cp config/json_utils.config.example.yaml config/json_utils.config.yaml
```

Example `config/config.yaml` (trimmed):
```yaml
dataset:
  image_root: /data/images
  recursive: true
  path_attributes: [lab, building, scene]
  path_attribute_map: {lab: 3, building: 2, scene: 1}
  user_jsonl: data/manual_annotations.jsonl
  output_jsonl: output/image_index.jsonl

yoloe:
  model_path: models/yoloe-11l-seg.pt
  threshold: 0.5
  visualize: true
  visualization_dir: output/visualizations/yoloe
  output_jsonl: output/yoloe_output.jsonl
  attribute_filters: {lab: alpha}

gemini:
  prompt: |
    Analyze the following images from the same scene...
  classes_to_forward: [person, screen]
  min_confidence: 0.5
  scene_directory_level: 2
  max_batch_size_bytes: 1981808640
  output_batch_dir: output/gemini/batches
  gcs_bucket: privacy-pipeline-inputs
  gcs_output_bucket: privacy-pipeline-outputs
  gcs_output_prefix: gemini_outputs
  project: my-gcp-project
  location: us-central1
  model: gemini-2.5-flash
  submitted_jobs_file: output/gemini/gemini_jobs.json
  final_output_jsonl: output/gemini_output.jsonl
```

Any CLI flag overrides the YAML value for that run. Attribute filters automatically relocate outputs to `output/filtered/<stage>/<filter-slug>` and log files to `logs/<command>_<slug>.log`.

For JSON utilities, store defaults separately:
```yaml
list_values:
  inputs: [output/image_index.jsonl]
  attributes: [lab, building]

summarize:
  inputs: [output/yoloe_output.jsonl]
  attributes: [lab, building]
  filters:
    - lab=alpha
    - {lab: alpha, building: west}

merge_filtered:
  stage: yoloe
  base_dir: output/filtered
  output: output/merged_yoloe_output.jsonl
```

---

## Outputs, Logs, and Artifacts
- `output/`: canonical location for stage JSONLs (`image_index.jsonl`, `yoloe_output.jsonl`, `gemini_output.jsonl`, `gemini_output_cleaned.jsonl`).
- `output/visualizations/yoloe/`: optional overlay PNGs mirroring the original directory structure.
- `output/gemini/`: contains `batches/`, `gemini_jobs.json`, `gemini_outputs/`, and cleaned exports. Filter-specific runs mirror this structure under `output/filtered/gemini/<slug>/`.
- `models/`: YOLOE checkpoints plus auto-generated custom weights (`*-custom.pt`) and the latest `config/yoloe_custom_mapping.txt`.
- `logs/`: every CLI run logs to `logs/<command>.log`, or `logs/<command>_<filter-slug>.log` when filters are in play.

Always keep the index and cleaned outputs under version control or archival storage if you need to reproduce findings later.

---

## Troubleshooting & Tips
- **Missing config**: pass `--config path/to/config.yaml` explicitly if your file lives elsewhere or if you are running JSON utilities with a different config.
- **Scene previews in notebooks**: the Gemini explorer shows thumbnails only when the original index path is reachable from the notebook environment.
- **Filtered runs**: use `--attribute-filter name=value` for YOLOE, `prepare-gemini`, or `clean-gemini` to analyze subsets independently. Merge them later via `json-utils merge-filtered`.
- **Batch sizing**: reduce `max_batch_size_bytes` if you hit Vertex upload quota or increase it (within platform limits) to reduce job count.
- **Credentials**: ensure `GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth application-default login` is configured before submitting/checking/downloading Gemini jobs.

With these steps, the data-privacy-tool can screen entire datasets end-to-end, surface sensitive artifacts, and provide human analysts with the charts and notebooks they need to make informed release decisions.
