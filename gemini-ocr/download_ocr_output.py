import os
import re
from tqdm import tqdm
from google.cloud import storage
from glob import glob

# --- Configuration ---
BUCKET_NAME = "genai-batch-input"
# This regex matches paths with exactly three directory levels followed by the file.
# e.g., "level1/level2/level3/predictions.jsonl"
PATH_PATTERN = re.compile(r"^[^/]+/[^/]+/[^/]+/predictions\.jsonl$")
LOCAL_DESTINATION = "/usr/project/xtmp/bsc32/gemini_ocr_output/new/"

# --- Script ---
# Initialize the client and get the bucket
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

print(f"Searching for files matching '{PATH_PATTERN.pattern}' in bucket '{BUCKET_NAME}'...")

# List all blobs in the bucket
all_blobs = bucket.list_blobs()

# Filter blobs that match the specific path pattern
matched_blobs = [blob for blob in all_blobs if PATH_PATTERN.match(blob.name)]

print(f"Total files found in bucket: {len(list(bucket.list_blobs()))}")

print("Filtering for AUTOLab on November 18th")
matched_blobs = [blob for blob in matched_blobs if "AUTOLab" in blob.name]
matched_blobs = [blob for blob in matched_blobs if "2025-11-18" in blob.name]
print(f"Files after AUTOLab filter: {len(matched_blobs)}")

if not matched_blobs:
    print("No matching files found.")
else:
    print(f"Found {len(matched_blobs)} matching files. Starting download...")

    # Download each matched blob
    for blob in tqdm(matched_blobs):
        # Create the full local path, including the directory structure
        local_dir_name = blob.name.split('/')[0]
        local_file_path = os.path.join(LOCAL_DESTINATION, local_dir_name, os.path.basename(blob.name))

        # Create the local directory structure if it doesn't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        if os.path.exists(local_file_path):
            print(f"File {local_file_path} already exists. Skipping download.")
            continue

        # Download the file
        print(f"Downloading gs://{BUCKET_NAME}/{blob.name} to {local_file_path}...")
        blob.download_to_filename(local_file_path)

    print("\nDownload complete.")
    print(f"All files downloaded to '{LOCAL_DESTINATION}'.")