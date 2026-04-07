#!/bin/bash
set -e

LOCAL_DIR="."

mkdir -p "$LOCAL_DIR"

sudo apt-get update -yq
sudo apt-get install -yq unrar unzip
pip install -q huggingface-hub

echo "=== Starting HMDB51 Processing ==="
mkdir -p "${LOCAL_DIR}/raw_hmdb"

echo "Downloading HMDB51 from Hugging Face mirror..."
python3 -c "
from huggingface_hub import hf_hub_download
import os

hf_hub_download(
    repo_id='jili5044/hmdb51',
    filename='hmdb51.zip',
    repo_type='dataset',
    local_dir='${LOCAL_DIR}',
    token=os.environ['HF_TOKEN']
)
"

echo "Extracting main archive..."
unzip -qo "${LOCAL_DIR}/hmdb51.zip" -d "${LOCAL_DIR}/raw_hmdb/"
rm "${LOCAL_DIR}/hmdb51.zip"