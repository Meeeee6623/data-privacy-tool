"""
Submits Gemini batch jobs for all JSONL files in /usr/project/xtmp/bsc32/ocr_batch_inputs.
Each file is uploaded to GCP and a batch job is created.
"""

import os
import time
from google import genai
from google.genai import types
from google.cloud import storage

# Directory containing batch job files
batch_inputs_dir = "/usr/project/xtmp/bsc32/ocr_batch_inputs"

# GCP bucket for uploading
bucket_name = "genai-batch-input"

# Initialize clients
client = genai.Client(
    vertexai=True, project='white-ground-476515-u3', location='us-central1'
)
gcs_client = storage.Client()
bucket = gcs_client.bucket(bucket_name)

# Get list of files in the directory
batch_files = [f for f in os.listdir(batch_inputs_dir) if f.endswith('.jsonl')]
batch_files.sort()


if not batch_files:
    print("No JSONL files found in the directory.")
    exit(1)

# Submit batch jobs for each file
job_names = []
for batch_file in batch_files:
    if 'AUTOLab' not in batch_file:
        continue
    batch_file_path = os.path.join(batch_inputs_dir, batch_file)
    
    # Upload file to GCP bucket
    blob = bucket.blob(batch_file)
    blob.upload_from_filename(batch_file_path)
    blob_uri = f"gs://{bucket.name}/{blob.name}"
    print(f"Uploaded {batch_file} to {blob_uri}")
    
    # Create batch job
    file_batch_job = client.batches.create(
        model="gemini-2.5-flash",
        src=blob_uri,
        config={
            "display_name": f"{batch_file.replace('.jsonl', '')}",
        },
    )
    job_names.append(file_batch_job.name)
    print(f"Created batch job: {file_batch_job.name}")

# Optionally, monitor all jobs (polling each)
completed_states = set(
    [
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    ]
)

for job_name in job_names:
    print(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name)
    while batch_job.state.name not in completed_states:
        print(f"Current state: {batch_job.state.name}")
        time.sleep(30)  # Wait for 30 seconds before polling again
        batch_job = client.batches.get(name=job_name)
    print(f"Job {job_name} finished with state: {batch_job.state.name}")
    if batch_job.state.name == "JOB_STATE_FAILED":
        print(f"Error: {batch_job.error}")
