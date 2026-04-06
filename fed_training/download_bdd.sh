#!/bin/bash
set -e

LOCAL_DIR="/content/fed_bdd_processing"

mkdir -p "$LOCAL_DIR"

sudo apt-get update -yq
sudo apt-get install -yq aria2 ffmpeg unzip python3
pip install -q huggingface-hub

transcode_video() {
    local src="$1"
    local dst_dir="$2"
    local base
    base=$(basename "${src%.*}")
    local dst="${dst_dir}/${base}.mp4"
    ffmpeg -y -i "$src" \
        -vf "scale=256:-2" \
        -r 4 \
        -c:v h264_nvenc -preset p4 -cq 23 \
        -an \
        "$dst" 2>/dev/null
    echo "  -> Created $dst"
}
export -f transcode_video

for zip_idx in 00 01 02 03 04; do
    CLIENT_NAME="client_bdd_${zip_idx}"

    if python3 -c "
import os
from huggingface_hub import list_repo_files
files = list(list_repo_files(repo_id='tiger9406/fed-client-dataset', repo_type='dataset', token=os.environ['HF_TOKEN']))
exit(0 if '${CLIENT_NAME}.tar.gz' in files else 1)
"; then
        echo "[$CLIENT_NAME] Already exists on HF. Skipping."
        continue
    fi

    echo "=== Processing $CLIENT_NAME ==="

    if [ -f "${LOCAL_DIR}/${CLIENT_NAME}.tar.gz" ]; then
        echo "Final Zip file already here locally; sending to HF"
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
        echo "Cleaning up local files..."
        rm -rf "${LOCAL_DIR}/raw_bdd_${zip_idx}" "${LOCAL_DIR}/${CLIENT_NAME}" "${LOCAL_DIR}/${CLIENT_NAME}.tar.gz"
        echo "Finished $CLIENT_NAME."
        continue
    fi

    mkdir -p "${LOCAL_DIR}/raw_bdd_${zip_idx}"
    mkdir -p "${LOCAL_DIR}/${CLIENT_NAME}"
    if [ -f "${LOCAL_DIR}/bdd_${zip_idx}.zip" ]; then
        echo "Zip ${zip_idx} already exists locally. Skipping download."
    else
        echo "Downloading zip ${zip_idx}..."
        aria2c -x 16 -s 16 -d "${LOCAL_DIR}" -o "bdd_${zip_idx}.zip" \
            "http://128.32.162.150/bdd100k/video_parts/bdd100k_videos_train_${zip_idx}.zip"
    fi

    echo "Extracting..."
    unzip -qo "${LOCAL_DIR}/bdd_${zip_idx}.zip" -d "${LOCAL_DIR}/raw_bdd_${zip_idx}/"
    rm "${LOCAL_DIR}/bdd_${zip_idx}.zip"

    echo "Transcoding (limited to 1200 videos, 2 concurrent workers)..."
    find "${LOCAL_DIR}/raw_bdd_${zip_idx}" \( -name "*.mov" -o -name "*.mp4" \) | head -n 1200 \
        | xargs -P 3 -I{} bash -c 'transcode_video "$1" "'"${LOCAL_DIR}/${CLIENT_NAME}"'"' _ {}
    echo "Archiving and saving to Drive..."
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

    echo "Cleaning up local files..."
    rm -rf "${LOCAL_DIR}/raw_bdd_${zip_idx}" "${LOCAL_DIR}/${CLIENT_NAME}" "${LOCAL_DIR}/${CLIENT_NAME}.tar.gz"
    echo "Finished $CLIENT_NAME."
done

echo "All BDD clients processed successfully."