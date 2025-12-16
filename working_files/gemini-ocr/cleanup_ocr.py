import os
import glob
from collections import defaultdict
import json
from tqdm import tqdm
import sys

try:
    import orjson
    def load_json_line(line):
        return orjson.loads(line)
    def dump_json_line(data):
        return orjson.dumps(data)
except ImportError:
    print("orjson not found, falling back to standard json library. For better performance, consider `pip install orjson`.")
    def load_json_line(line):
        return json.loads(line)
    def dump_json_line(data):
        return json.dumps(data).encode('utf-8')

def cleanup_ocr_data(base_name=None):
    """
    Processes OCR batch inputs and their corresponding Gemini outputs,
    extracting and merging relevant data, and adding the corresponding file path.
    If base_name is provided, only process that group.
    """
    input_dir = "/usr/project/xtmp/bsc32/ocr_batch_inputs"
    output_dir = "/usr/project/xtmp/bsc32/gemini_ocr_output/new"
    cleaned_dir = "/usr/project/xtmp/bsc32/gemini_ocr_output/clean"
    predictions_base_dir = "/usr/project/xtmp/bsc32/droid_yoloe"

    os.makedirs(cleaned_dir, exist_ok=True)

    # Group input files by base name (e.g., 'AUTOLab' for 'AUTOLab_0.jsonl')
    # filter for AUTOLab only
    input_files = glob.glob(os.path.join(input_dir, "AUTOLab*.jsonl"))
    # input_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    input_files.sort()
    grouped_files = defaultdict(list)
    for f in input_files:
        base = os.path.basename(f).split('_')[0]
        grouped_files[base].append(f)

    if base_name:
        if base_name not in grouped_files:
            print(f"Error: Base name '{base_name}' not found in grouped files.")
            return
        groups_to_process = {base_name: grouped_files[base_name]}
    else:
        groups_to_process = grouped_files

    print(f"Found {len(groups_to_process)} groups of files to process.")

    for base_name, files in tqdm(groups_to_process.items(), desc="Processing groups"):
        cleaned_output_path = os.path.join(cleaned_dir, f"{base_name}.jsonl")

        # --- Load predictions for the current group ---
        predictions_txt_path = os.path.join(predictions_base_dir, base_name, "predictions.txt")
        path_map = defaultdict(list)
        if os.path.exists(predictions_txt_path):
            with open(predictions_txt_path, 'r') as f_pred:
                for line in f_pred:
                    parts = line.split('.png,', 1)
                    if len(parts) == 2:
                        path_part = parts[0]
                        prediction_part = parts[1].strip()
                        if prediction_part != "[]":
                            # Create a searchable key from the path, e.g., '3780459880/Mon_May_29_23:24:39_2023/16787047'
                            path_segments = path_part.split('/')
                            if len(path_segments) > 4:
                                # The key is the last 3 components of the directory path
                                searchable_key = "/".join(path_segments[-4:-1])
                                path_map[searchable_key].append(path_part + ".png")
        else:
            print(f"Warning: Predictions file not found for group {base_name} at {predictions_txt_path}")

        with open(cleaned_output_path, 'wb') as cleaned_file:
            for input_file_path in tqdm(sorted(files), desc=f"Processing files for {base_name}"):
                file_name_no_ext = os.path.splitext(os.path.basename(input_file_path))[0]
                output_jsonl_path = os.path.join(output_dir, file_name_no_ext, "predictions.jsonl")

                if not os.path.exists(output_jsonl_path):
                    print(f"Warning: Output file not found for {input_file_path}, skipping: {output_jsonl_path}")
                    continue

                # Load all responses from the output file into a dictionary keyed by 'key'
                responses = {}
                with open(output_jsonl_path, 'r') as f_out:
                    for line in tqdm(f_out, desc=f"Loading responses for {file_name_no_ext}"):
                        try:
                            data = load_json_line(line)
                            # The output from Gemini Batch has the input 'key' and the 'response'
                            if 'key' in data and 'response' in data and data['response']:
                                try:
                                    text = data['response']['candidates'][0]['content']['parts'][0]['text']
                                    responses[data['key']] = text
                                except (KeyError, IndexError, TypeError):
                                    # Handle cases where the expected structure is not present
                                    responses[data['key']] = None
                            else:
                                responses[data.get('key')] = None

                        except (json.JSONDecodeError, orjson.JSONDecodeError):
                            print(f"Skipping malformed line in {output_jsonl_path}: {line.strip()}")
                            continue
                
                # Iterate through the input file to match keys and write cleaned data
                with open(input_file_path, 'r') as f_in:
                    for line in tqdm(f_in, desc=f"Processing inputs for {file_name_no_ext}"):
                        try:
                            input_data = load_json_line(line)
                            key = input_data.get("key")
                            if key in responses:
                                # make sure file path is found
                                if key not in path_map:
                                    print(f"Warning: No file path found for key {key} in predictions.txt")
                                cleaned_data = {
                                    "key": key,
                                    "text": responses[key],
                                    "file_path": path_map.get(key) # Get file path from map
                                }
                                cleaned_file.write(dump_json_line(cleaned_data) + b'\n')
                        except (json.JSONDecodeError, orjson.JSONDecodeError):
                            print(f"Skipping malformed line in {input_file_path}: {line.strip()}")
                            continue
                            
    print(f"\nProcessing complete. Cleaned files are in {cleaned_dir}")

if __name__ == "__main__":
    base_name = sys.argv[1] if len(sys.argv) > 1 else None
    cleanup_ocr_data(base_name)
