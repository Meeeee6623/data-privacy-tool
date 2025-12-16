import gc
import re
import json
import base64
import uuid
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

prompt = """Analyze the following images, which are all from the same scene.

1. Combine Text:
Under the heading `## Combined Text ##`, combine all readable text from all images into a single, coherent block. Only respond with text that actually visible in the image, DO NOT make up or text that you are unsure of from blurry parts of the image.  

2. Analyze and Flag:
Review the synthesized text and the images for any sensitive data. If sensitive data is found, you must provide **only one additional line** as your response, starting with `#FLAG:` followed by a comma-separated list of all categories violated.

Categories:
*   PII: Names, addresses, phone numbers, emails.
*   CONFIDENTIAL_INFO: Schematics, source code, internal documents, proprietary notes.
*   SECURITY_INFO: Passwords, network names, access codes.
*   MACHINE_READABLE_CODE: QR codes, barcodes.
*   BRANDING_LOGOS: Company logos, trademarks, etc.
*   OTHER: Any other clearly sensitive information not covered above.

Ignore references to the FRANKA EMIKA robot arm in this analysis, as it is the subject of the dataset."""


name_pattern = re.compile(r"'name': '([^']+)'")


def get_images_from_predictions(file_path):
    scene_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            names = re.findall(name_pattern, line)
            image_path = line.split(", [")[0]
            path_parts = image_path.split("/")
            scene_id = path_parts[-4]
            camera_id = path_parts[-2]
            timestamp = path_parts[-3]
            full_scene_id = f"{scene_id}/{timestamp}/{camera_id}"
            if names and names != ["person"]:
                if full_scene_id not in scene_dict:
                    scene_dict[full_scene_id] = []
                scene_dict[full_scene_id].append(image_path)
    return scene_dict

def get_json_record(image_paths: list, text_prompt: str, request_id: str = None) -> str:
    """
    Creates a single JSON record suitable for a Gemini Batch JSONL input file.
    The text prompt is placed before the base64 encoded image data for all images.

    Args:
        image_paths: A list of file paths to the PNG images.
        text_prompt: The text prompt to associate with the images.
        request_id: An optional unique identifier for this request. If None, a UUID will be generated.

    Returns:
        A JSON string representing one record for the JSONL file.
    """
    # If no request_id is provided, generate a unique one
    if request_id is None:
        request_id = str(uuid.uuid4())

    # Construct the parts list starting with the text prompt
    parts = [{"text": text_prompt}]
    
    # Encode each image as base64 and add to parts
    for image_path in image_paths:
        encoded_image = ""
        try:
            with open(image_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at: {image_path}")
        except Exception as e:
            raise IOError(f"Error encoding image {image_path}: {e}")
        
        parts.append({
            "inlineData": {
                "mime_type": "image/png",  # All images are PNGs
                "data": encoded_image,
            }
        })

    # Construct the JSON record with text prompt before the images
    record = {
        "key": request_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ]
        },
    }
    return json.dumps(record)


def create_record(scene_item):
    scene_id, image_paths = scene_item
    return get_json_record(image_paths, prompt, request_id=scene_id)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import glob

    output_dir = Path("/usr/project/xtmp/bsc32/ocr_batch_inputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 1000

    predictions_files = []
    if len(sys.argv) > 1:
        predictions_file = Path(sys.argv[1])
        if predictions_file.exists():
            predictions_files = [str(predictions_file)]
        else:
            print(f"Predictions file not found: {predictions_file}")
            sys.exit(1)
    else:
        # Glob predictions files from /usr/project/xtmp/bsc32/droid_yoloe/*/predictions.txt
        predictions_files = glob.glob('/usr/project/xtmp/bsc32/droid_yoloe/*/predictions.txt')
        predictions_files.sort()

    if not predictions_files:
        print("No predictions files found.")
        sys.exit(1)

    for pf in predictions_files[0:1]:
        pf_path = Path(pf)
        output_base = pf_path.parent.name
        output_jsonl_file = Path(f"{output_base}.jsonl")

        scene_dict = get_images_from_predictions(pf)

        print(f"Processing {pf}: Found {len(scene_dict)} filtered scenes.")

        scene_items = list(scene_dict.items())
        files = []

        with ProcessPoolExecutor(max_workers=16) as executor:
            for i in range(0, len(scene_items), BATCH_SIZE):
                batch = scene_items[i:i+BATCH_SIZE]
                json_records = list(tqdm(
                    executor.map(create_record, batch),
                    desc=f"Creating Gemini OCR batch JSONL for {output_base} (batch {i//BATCH_SIZE + 1})",
                    total=len(batch)
                ))

                # Split into files under 2GB each
                max_size = 1.85 * 1024**3  # 1.85GB in bytes
                current_records = []
                current_size = 0

                for record in json_records:
                    record_size = len(record) + 1  # +1 for newline
                    if current_size + record_size > max_size and current_records:
                        # Write current batch to a new file
                        file_name = output_jsonl_file.with_name(f"{output_jsonl_file.stem}_{len(files)}{output_jsonl_file.suffix}")
                        with open(f"{output_dir}/{file_name}", "w") as f:
                            for r in current_records:
                                f.write(r + "\n")
                        files.append(file_name)
                        current_records = []
                        current_size = 0
                    current_records.append(record)
                    current_size += record_size

                # Write the last batch for this scene batch
                if current_records:
                    file_name = output_jsonl_file.with_name(f"{output_jsonl_file.stem}_{len(files)}{output_jsonl_file.suffix}")
                    with open(f"{output_dir}/{file_name}", "w") as f:
                        for r in current_records:
                            f.write(r + "\n")
                    files.append(file_name)

                del json_records
                del current_records
                gc.collect()

        print(f"Wrote Gemini OCR batch JSONL for {output_base} to: {', '.join(str(f) for f in files)}")
        
        del scene_dict
        gc.collect()