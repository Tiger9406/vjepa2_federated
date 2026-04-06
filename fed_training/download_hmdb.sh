#!/bin/bash
set -e

LOCAL_DIR="/content/fed_hmdb_processing"

mkdir -p "$LOCAL_DIR"

sudo apt-get update -yq
sudo apt-get install -yq aria2 ffmpeg unrar unzip python3
pip install -q huggingface-hub

if nvidia-smi &>/dev/null; then
    echo "GPU detected — using NVENC for transcoding."
    VIDEO_CODEC="-c:v h264_nvenc -preset fast"
else
    echo "No GPU detected — using CPU (libx264)."
    VIDEO_CODEC="-c:v libx264 -preset veryfast -crf 23"
fi

transcode_video() {
    local src="$1"
    local dst_dir="$2"
    local base
    base=$(basename "${src%.*}")
    local dst="${dst_dir}/${base}.mp4"
    ffmpeg -y -i "$src" \
        -vf "scale=256:-2" \
        -r 4 \
        $VIDEO_CODEC \
        -an \
        "$dst" 2>/dev/null
    echo "  -> Created $dst"
}
export -f transcode_video

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

echo "Extracting nested category archives..."
find "${LOCAL_DIR}/raw_hmdb/" -name "*.rar" -execdir unrar x -o+ -idq {} \;

NUM_CLIENTS=4
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    mkdir -p "${LOCAL_DIR}/raw_hmdb/client_split_${i}"
    mkdir -p "${LOCAL_DIR}/client_hmdb_${i}"
done

echo "Distributing videos evenly across $NUM_CLIENTS clients..."
idx=0
find "${LOCAL_DIR}/raw_hmdb/" -name "*.avi" | while read -r vid_file; do
    target=$((idx % NUM_CLIENTS))
    mv "$vid_file" "${LOCAL_DIR}/raw_hmdb/client_split_${target}/"
    idx=$((idx + 1))
done

for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    CLIENT_NAME="client_hmdb_${i}"

    if python3 -c "
import os
from huggingface_hub import list_repo_files
files = list(list_repo_files(repo_id='tiger9406/fed-client-dataset', repo_type='dataset', token=os.environ['HF_TOKEN']))
exit(0 if '${CLIENT_NAME}.tar.gz' in files else 1)
"; then
        echo "[$CLIENT_NAME] Already exists on HF. Skipping."
        continue
    fi

    echo "--- Processing $CLIENT_NAME ---"
    echo "  Transcoding videos (2 concurrent workers)..."
    find "${LOCAL_DIR}/raw_hmdb/client_split_${i}/" -name "*.avi" \
        | xargs -P 2 -I{} bash -c 'transcode_video "$1" "'"${LOCAL_DIR}/${CLIENT_NAME}"'"' _ {}

    echo "  Archiving and saving to Drive..."
    tar -czf "${LOCAL_DIR}/${CLIENT_NAME}.tar.gz" -C "${LOCAL_DIR}/${CLIENT_NAME}" .

    python3 -c "
from huggingface_hub import HfApi, login
import os
login(token=os.environ['HF_TOKEN'])
api = HfApi()
api.upload_file(
    path_or_fileobj='${LOCAL_DIR}/${CLIENT_NAME}.tar.gz',
    path_in_repo='${CLIENT_NAME}.tar.gz',
    repo_id='tiger9406/fed-client-dataset',
    repo_type='dataset'
)
"

    echo "  Cleaning up local files for $CLIENT_NAME..."
    rm -rf "${LOCAL_DIR}/${CLIENT_NAME}.tar.gz" "${LOCAL_DIR}/${CLIENT_NAME}"
done

rm -rf "${LOCAL_DIR}/raw_hmdb"
echo "All HMDB51 clients processed successfully."