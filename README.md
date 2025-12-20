# Data Privacy Tool

- [Overview](#overview)
- [Requirements](#requirements)
- [Notes](#notes)
- [Pipeline Steps](#pipeline-steps)
  - [1. Indexing](#1-indexing)
  - [2. YOLOE Detection](#2-yoloe-detection)
  - [3. Gemini OCR + Classification](#3-gemini-ocr--classification)
- [Configuration Reference](#configuration-reference)
- [JSON Utilities](#json-utilities)
- [Visualization Notebooks](#visualization-notebooks)

## Overview
This tool is broken down into a pipeline with 3 main stages: Indexing, YOLOE detection, and Gemini OCR + classification. The indexing stage ingests all images in some root folder and categorizes them, inferring attributes from the image paths. Having these attribute fields is useful for sorting through/filtering images later, as well as separating sets of images for parallel processing. If you have other attributes you would like to tie to your images, you can pass those in from a file as well. For later stages and for [visualization notebooks](#visualization-notebooks), it is assumed that images from some folder level come from the same collection time (for example, for `room/timestamp/cameraX/images` you could group all images for each timestamped collection together). The level at which this split happens is configurable.

Next, the YOLOE stage helps to narrow down the search space of images by searching for a set of custom classes. It also optionally generates visualizations of the detections. There is a Jupyter notebook to make looking through detections easy, with dropdowns to filter by attributes.

Last is the Gemini OCR stage. This stage uses Gemini Flash to attempt to extract any readable text from images, as well as classify images with privacy leaks into broad categories for filtering. For every image with a detection from YOLOE, it groups all images from that "scene" and adds them to a single request, encoding all requests in JSONL files to upload to GCP Vertex batch processing. Then it can submit the jobs for you, check job status, download the jobs, and extract the relevant parts of the responses. There is another Google/Colab-style notebook for visualizing these responses.

Note: If there is a need for handlers for other LLM apis/cloud providers, feel free to reach out to me and I can help out (or submit a pull request!)
## Requirements
- Python 3.13+, GPU recommended for running YOLOE
- Google Cloud account with Vertex AI Gemini access (batch jobs are used for the 50% discount mentioned in the [Gemini section](#3-gemini-ocr--classification))
- `uv` for dependency management. Install everything with:
  ```bash
  uv sync
  # or: uv venv && uv pip install -e .
  ```
- All Python dependencies are declared in `pyproject.toml`, so the CLI can also be run via `python -m privacy_pipeline.cli ...` once synced

## Notes
This tool requires all input files to be images; use `ffmpeg` to convert videos to images before indexing and keep the folder hierarchy consistent with how you intend to group scenes.

Example commands (overwrite `frames/%05d.png` as needed):
```bash
# 1 fps extraction at high quality
ffmpeg -i input.mp4 -vf fps=1 -qscale:v 1 frames/frame_%05d.png

# Full FPS extraction at source rate and quality
ffmpeg -i input.mp4 -qscale:v 1 frames/full_%05d.png

# First, middle, last frames only (stored as frame_01.png, frame_02.png, frame_03.png)
TOTAL=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 input.mp4); \
MID=$((TOTAL/2)); LAST=$((TOTAL-1)); \
ffmpeg -i input.mp4 -vf "select='eq(n,0)+eq(n,$MID)+eq(n,$LAST)'" -vsync 0 -qscale:v 1 frames/frame_%02d.png
```

## Pipeline Steps
The CLI is exposed as `python -m privacy_pipeline.cli <command>` and it automatically loads `config/config.yaml` (copy `config/config.example.yaml` to get started). Every stage writes JSONL outputs to `output/` by default, and when you add `--attribute-filter` the results are redirected into `output/filtered/<stage>/<filter-slug>/...` so you can parallelize slices and merge them later (see [JSON Utilities](#json-utilities)).

### 1. Indexing
**Goal**: flatten a folder tree of images into a JSONL with friendly metadata so subsequent stages and notebooks can reason about scenes.

Ways to attach attributes during indexing:
- **Path-derived fields**: `--path-attributes lab building scene` labels each level under the root.
- **Levels-up mapping**: `--path-attribute-map lab:3 building:2 scene:1` climbs up relative to the image itself (useful when the folder structure is `camera/images`).
- **External metadata**: `--user-jsonl data/extra_attributes.jsonl` merges arbitrary attributes (e.g., shift ID, operator, experiment ID) by aligning on `image_path`.

Example full run and a variant focused on a strict folder depth:
```bash
# Standard end-to-end index, recursive crawl, writes to output/image_index.jsonl
python -m privacy_pipeline.cli index /data/images \
  --path-attributes lab building scene \
  --path-attribute-map lab:3 building:2 scene:1 \
  --user-jsonl data/manual_annotations.jsonl

# Index a single collection level (scene_level=2 set later in config) and override output
python -m privacy_pipeline.cli index /datasets/run42 --output output/run42_index.jsonl --no-recursive
```

Key configuration links for indexing (see [Configuration Reference](#configuration-reference)):
- `dataset.image_root`, `dataset.recursive`, and `dataset.scene_grouping_level` define which files become rows and how scenes are grouped.
- `dataset.path_attributes` and `dataset.path_attribute_map` decide how attributes appear in notebooks and how filters can be applied later.
- `dataset.user_jsonl` lets you merge spreadsheets or annotations; the CLI keeps user attributes even if the same image is re-indexed later.

Scene grouping is especially important downstream: for the assumptions outlined in the [Overview](#overview), choose the folder level that matches your capture fidelity (e.g., `scene_grouping_level: 2` to group by `session` in `lab/session/camera/image.png`).

### 2. YOLOE Detection
**Goal**: run YOLOE on the indexed inventory to locate sensitive classes, optionally emit visualization overlays, and (optionally) apply attribute filters so expensive GPU runs can be sharded.

How filtering works: `--attribute-filter lab=alpha building=west` limits the run to rows with those attributes. The CLI automatically writes detections and visualizations to `output/filtered/yoloe/lab-alpha_building-west/` so you can spawn multiple workers, one per slice. After the slices complete, merge them with `python -m privacy_pipeline.cli json-utils merge-filtered yoloe --base-dir output/filtered --output output/yoloe_output_merged.jsonl` before continuing, or run downstream stages with the same filters if you want to keep the slices independent.

Commands you will use most often:
```bash
# Run YOLOE on the full index, render visualizations, keep the default classes
python -m privacy_pipeline.cli yoloe
# Filtered run that only processes one lab/building pair and writes its own outputs
python -m privacy_pipeline.cli yoloe output/image_index.jsonl \
  --attribute-filter lab=alpha building=west \
  --model models/yoloe-11l-seg.pt \
  --output output/filtered/yoloe/lab-alpha_building-west/yoloe_output.jsonl
```

Important configuration options:
- `config/yoloe_classes.txt`: List of default classes that YOLOE is configured to search for, change to look for other objects in images. 
- `yoloe.model_path`: YOLOE checkpoint, downloaded into `models/` on demand.
- `yoloe.threshold`: score threshold used both for JSONL detections and to decide which scenes get sent to Gemini.
- `yoloe.visualize` and `yoloe.visualization_dir`: toggles overlay PNGs per detection.
- `yoloe.output_jsonl`: base output path when no filter is provided.
- `yoloe.attribute_filters`: Allows setting a filters to only run on images with certain attributes


### 3. Gemini OCR + Classification
**Goal**: send YOLOE-positive scenes to Vertex AI Gemini (2.5 flash by default), batch requests for the 50% batch-discount pricing, keep all requests under the 2 GB Vertex limit, and return cleaned structured logs for analysis.

This stage only targets **Google Cloud Vertex Batch**. Each record is encoded by embedding the prompt plus base64 versions of every PNG/JPEG in the scene directly into a JSON line. To stay within Vertex's 2 GB cap per batch request, the [`max_batch_size_bytes`](#configuration-reference) limit (1.85 GB by default) splits requests across files, and the [cleaning step](#35-clean--merge) merges the responses back.

It is suggested to look through YOLOE visualizations to determine what classes/thresholds you want to use for this step. 

#### 3.1 Prepare batches
```bash
# Run over the merged YOLOE output and forward any person/screen detections >=0.5
python -m privacy_pipeline.cli prepare-gemini \
  --yolo_output output/yoloe_output.jsonl \
  --prompt "$(cat prompts/gemini_prompt.txt)" \
  --classes person screen \
  --threshold 0.5 \

# Parallel slice: only escalate scenes from lab alpha
python -m privacy_pipeline.cli prepare-gemini \
  --attribute-filter lab=alpha
```

For every detection that matches `classes_to_forward` and `min_confidence`, the CLI collects **all** images from that scene (leveraging `dataset.scene_grouping_level`) and places them in one JSONL record. Key configuration fields:
- `gemini.prompt`: full text instructions sent alongside each scene (multiline YAML supported).
- `gemini.classes_to_forward` + `gemini.min_confidence`: detection criteria. Leave blank for all classes at 0.5 threshold. 
- `gemini.scene_grouping_level`: inherits from the dataset section unless overridden via `--scene-level`.
- `gemini.max_batch_size_bytes`: ensures batch files are <2 GB before upload and is what drives automatic splitting.
- `gemini.output_batch_dir`: where `batch_*.jsonl` lives; filters rewrite this under `output/filtered/gemini/<slug>/batches`.
- `gemini.attribute_filters`: default per-stage filter; combine with YOLOE filters to process slices consistently.

If you generated multiple filtered YOLOE runs you can either run `prepare-gemini` per slice or merge them first via [`json-utils merge-filtered yoloe`](#json-utilities).

#### 3.2 Submit batches
```bash
python -m privacy_pipeline.cli submit-gemini \
  output/gemini/batches \
  --gcs-bucket my-input-bucket \
  --gcs-output-bucket my-output-bucket \
  --project my-gcp-project \
  --location us-central1 \
  --model gemini-2.5-flash
```

`submit-gemini` uploads each batch JSONL to Cloud Storage (creating buckets if needed) and starts  Vertex Batch jobs that link the uploaded file to your chosen Gemini Flash model. A `gemini_jobs.json` file is produced so the next steps can reference the job IDs and google cloud URIs (querying for the URIs to download files was not working, so the tool generates an output URI on job submission and saves it here as well).

Key configuration fields:
- `gemini.gcs_bucket`: input bucket for batch files.
- `gemini.project`, `gemini.location`, : Vertex project information.
- `gemini.model`: The VLLM model to use. 
- `gemini.submitted_jobs_file`: JSON file to keep track of submitted jobs. 

#### 3.3 Check jobs
Monitor job state before downloading results:
```bash
python -m privacy_pipeline.cli check-gemini
# or pass a file
python -m privacy_pipeline.cli check-gemini --jobs-file output/gemini/gemini_jobs.json
# or target specific job IDs
python -m privacy_pipeline.cli check-gemini job-123 job-456 --project my-gcp-project
```
The command uses the Vertex API to print JSON including `state`, `display_name`, and any `error` payload for jobs in the gemini_jobs.json (or provided) file. 

#### 3.4 Download
```bash
python -m privacy_pipeline.cli download-gemini \
  --jobs-file output/gemini/gemini_jobs.json \
  --output-dir output/gemini/gemini_outputs
```
One Vertex job might emit multiple files (one per chunk). Downloading after everything succeeds gives you `prediction_<job>_<chunk>.jsonl` under `output/gemini/gemini_outputs/`.

#### 3.5 Clean & merge
```bash
python -m privacy_pipeline.cli clean-gemini
```

Cleaning reads the downloaded JSONL chunks, removes inline image payloads, extracts the `#FLAG[...]` markers added by Gemini, and merges everything into a compact JSONL that includes:
- `scene_path` plus the original scene attributes (matched using the index and respecting any filters)
- `response_text`: the full gemini output.
- `flag_categories`: comma-separated categories normalized into a list for filtering
- `is_flagged`: a boolean for dashboards/statistics

## Configuration Reference
Copy `config/config.example.yaml` to `config/config.yaml` and adjust the following keys:

**Dataset**
- `image_root`: root directory to scan. Overridable via `index` positional argument.
- `recursive`: crawl subfolders (default `true`).
- `scene_grouping_level`: how many directories up count as a scene (1 for direct parent, etc.)
- `path_attributes`: ordered names extracted from directories under `image_root`.
- `path_attribute_map`: Gives you more control over which attributes come from which path sections (can skip certain folder levels)
- `user_jsonl`: JSONL of `{ "image_path": ..., "attributes": { ... } }` to merge (if you have some other custom metadata).
- `output_jsonl`: destination for the index (default `output/image_index.jsonl`).

**YOLOE**
- `model_path`: checkpoint file (downloaded into `models/` if missing). Should be a variant of yoloe-11-seg.
- `threshold`: detection confidence cutoff.
- `visualize`: whether or not to render visualizations for detections.
- `visualization_dir`: where to save renderings.
- `output_jsonl`: base detections file.
- `attribute_filters`: default filter dictionary for repeated filtered runs.

**Gemini**
- `prompt`: Prompt to describe how Gemini should combine OCR text and flag categories.
- `classes_to_forward`: YOLOE class names to forward.
- `min_confidence`: detection threshold to forward a scene.
- `max_batch_size_bytes`: caps `batch_*.jsonl` before upload (1.85 GB has been working for me).
- `output_batch_dir`: local staging folder for JSONL requests.
- `gcs_bucket`: where requests are uploaded.
- `gcs_output_bucket`/`gcs_output_prefix`: location for Vertex responses.
- `project`, `location`, `model`: Vertex deployment parameters (Gemini Flash only at the moment).
- `submitted_jobs_file`: JSON record of submitted jobs; reused by check/download steps.
- `final_output_jsonl`: destination for the cleaned/merged output (used by `clean-gemini`).
- `attribute_filters`: filter dictionary shared by `prepare`, `clean`, and the other Gemini sub-commands.

`config/json_utils.config.example.yaml` holds defaults for the helper commands documented below.

## JSON Utilities
`python -m privacy_pipeline.cli json-utils ...` exposes lightweight helpers for working with JSONL outputs. They respect `config/json_utils.config.yaml` so they can be run without extra flags (or you can pass in arguments through the cli)

- `list-values`: report all unique values for select attributes. Helpful before choosing filters.
  ```bash
  python -m privacy_pipeline.cli json-utils list-values output/image_index.jsonl --attributes lab building
  ```
- `summarize`: count how many rows fall into each attribute bucket and how proposed filters would split the work.
  ```bash
  python -m privacy_pipeline.cli json-utils summarize output/yoloe_output.jsonl --attributes lab building --filter lab=alpha --filter lab=beta
  ```
- `merge-filtered`: stitch together outputs from runs that used `--attribute-filter`, primarily for the heavy YOLOE and Gemini stages.
  ```bash
  python -m privacy_pipeline.cli json-utils merge-filtered yoloe --base-dir output/filtered --output output/yoloe_output_merged.jsonl
  ```

Because filters are meant to enable parallel processing (especially of YOLOE), the merge command is the bridge that lets you re-connect a complete dataset when you have multiple filtered slices sitting under `output/filtered/<stage>/`.

## Visualization Notebooks
Two notebooks in the [`notebooks/`](notebooks) folder make it easy to validate detections and OCR outputs once the CLI produces the JSONLs described above:

- `notebooks/yoloe_stage_explorer.ipynb`: loads `output/yoloe_output.jsonl` plus any filtered variants, provides widget-based filters for attributes/classes/confidence, renders Plotly summaries, and previews images with bounding boxes (if `visualize` was enabled). Use it whenever you need to tune thresholds or confirm a filter before running Gemini.
- `notebooks/gemini_stage_explorer.ipynb`: ingests `output/gemini_output_cleaned.jsonl` (or merged filtered outputs) and faceted charts by attribute, scene, and `flag_categories`. It also displays Gemini's combined text along with thumbnails drawn from the original index, so it's easy to see what the model decided for each scene. The notebook works after the [cleaning step](#35-clean--merge) because that is where the inline payloads are stripped and scenes inherit their attributes.

Start both notebooks with `uv run jupyter lab` (or your preferred Jupyter launcher) after completing the stages in [Pipeline Steps](#pipeline-steps).
