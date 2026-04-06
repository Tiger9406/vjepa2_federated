#!/bin/bash
set -e

LOCAL_DIR="/content/fed_epic_processing"
mkdir -p "$LOCAL_DIR"

sudo apt-get update -yq
sudo apt-get install -yq aria2 ffmpeg python3
pip install -q huggingface-hub

python3 -c "
import os
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])
"

echo "=== Starting EPIC-KITCHENS Processing ==="

for participant in P01 P02 P04; do
    CLIENT_NAME="client_epic_${participant}"

    if python3 -c "
    from huggingface_hub import HfApi
    api = HfApi()
    files = [f.rfilename for f in api.list_repo_files(repo_id='tiger9406/fed-client-dataset', repo_type='dataset')]
    exit(0 if '${CLIENT_NAME}.tar.gz' in files else 1)
    "; then
        echo "[$CLIENT_NAME] Already exists on HF. Skipping."
        continue
    fi

    echo "--- Processing $CLIENT_NAME ---"
    mkdir -p "${LOCAL_DIR}/raw_epic_${participant}"
    mkdir -p "${LOCAL_DIR}/${CLIENT_NAME}"

    echo "  Downloading up to 15 raw videos..."
    for i in $(seq -w 1 15); do
        aria2c -x 16 -s 16 -d "${LOCAL_DIR}/raw_epic_${participant}" \
        -o "${participant}_${i}.MP4" \
        "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/${participant}/${participant}_${i}.MP4" || true
    done

    echo "  Splitting into 20s chunks and transcoding to 256p @ 4fps..."
    for raw in "${LOCAL_DIR}/raw_epic_${participant}"/*.MP4; do
        [ -e "$raw" ] || continue
        base=$(basename "${raw%.*}")
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$raw")
        num_segments=$(python3 -c "import math; print(math.ceil($duration / 20))")

        echo "  -> Processing $base ($num_segments segments)"
        for i in $(seq 0 $((num_segments - 1))); do
        start=$((i * 20))
        out="${LOCAL_DIR}/${CLIENT_NAME}/${base}_clip_$(printf '%04d' $i).mp4"
        ffmpeg -y -ss "$start" -i "$raw" -t 20 \
            -vf "scale=256:-2" \
            -r 4 \
            -c:v libx264 -preset veryfast -crf 23 \
            -an \
            "$out" 2>/dev/null
        done
        echo "     Finished $base"
    done

    echo "  Archiving and saving to Drive..."
    tar -czf "${LOCAL_DIR}/${CLIENT_NAME}.tar.gz" -C "${LOCAL_DIR}/${CLIENT_NAME}" .

    python3 -c "
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj='${LOCAL_DIR}/${CLIENT_NAME}.tar.gz',
        path_in_repo='${CLIENT_NAME}.tar.gz',
        repo_id='tiger9406/fed-client-dataset',
        repo_type='dataset'
    )
    "

    echo "  Cleaning up local files..."
    rm -rf "${LOCAL_DIR}/raw_epic_${participant}" "${LOCAL_DIR}/${CLIENT_NAME}"
done

echo "All EPIC-KITCHENS clients processed successfully."