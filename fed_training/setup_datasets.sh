#!/bin/bash

set -e

DRIVE_DIR="/content/drive/MyDrive/preprocessed_datasets"
LOCAL_DIR="/content/vjepa2_federated/fed_training"

# helper: transcode one video to 256p @ 4fps 
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
    echo "  transcoded: $dst"
}
export -f transcode_video

# helper: copy dataset from drive to locao
restore_from_drive() {
    local dataset="$1"
    local local_dst="$2"
    echo "Restoring ${dataset} from Drive..."
    mkdir -p "$local_dst"
    cp -r "${DRIVE_DIR}/${dataset}/." "$local_dst/"
    echo "  restored $(find "$local_dst" -name "*.mp4" | wc -l) clips to ${local_dst}"
}

# save dataset to Drive
save_to_drive() {
    local local_src="$1"
    local dataset="$2"
    echo "Saving ${dataset} to Drive..."
    mkdir -p "${DRIVE_DIR}/${dataset}"
    cp -r "${local_src}/." "${DRIVE_DIR}/${dataset}/"
    echo "  saved to ${DRIVE_DIR}/${dataset}"
}

sudo apt-get install -y aria2 ffmpeg

echo "Checking Drive for preprocessed datasets"

# BDD100K
BDD_LOCAL="${LOCAL_DIR}/bdd_256"
BDD_DRIVE="${DRIVE_DIR}/bdd_256"

if [ -d "$BDD_DRIVE" ] && [ "$(ls -A "$BDD_DRIVE")" ]; then
    echo "[bdd_256] Found on Drive; copying locally."
    restore_from_drive "bdd_256" "$BDD_LOCAL"
else
    echo "[bdd_256] Not found on Drive; downloading and processing."
    mkdir -p "${LOCAL_DIR}/bdd/"
    mkdir -p "$BDD_LOCAL"

    aria2c -x 16 -s 16 -d "${LOCAL_DIR}" -o bdd100k_videos_train_00.zip \
        http://128.32.162.150/bdd100k/video_parts/bdd100k_videos_train_00.zip

    echo "Extracting BDD100K..."
    unzip "${LOCAL_DIR}/bdd100k_videos_train_00.zip" -d "${LOCAL_DIR}/bdd/"
    rm "${LOCAL_DIR}/bdd100k_videos_train_00.zip"

    echo "Transcoding BDD100K to 256p @ 4fps; 8 parallel workers"
    find "${LOCAL_DIR}/bdd/" \( -name "*.mov" -o -name "*.mp4" \) \
        | xargs -P 3 -I{} bash -c 'transcode_video "$1" "'"$BDD_LOCAL"'"' _ {}

    rm -rf "${LOCAL_DIR}/bdd/"

    save_to_drive "$BDD_LOCAL" "bdd_256"
fi

echo ""

EPIC_LOCAL="${LOCAL_DIR}/epic/clips"
EPIC_DRIVE="${DRIVE_DIR}/epic_clips"

if [ -d "$EPIC_DRIVE" ] && [ "$(ls -A "$EPIC_DRIVE")" ]; then
    echo "[epic_clips] Found on Drive; copying over locally"
    restore_from_drive "epic_clips" "$EPIC_LOCAL"
else
    echo "[epic_clips] Not found on Drive; download and process"
    mkdir -p "${LOCAL_DIR}/epic/raw/"
    mkdir -p "$EPIC_LOCAL"

    for video_id in P01_01 P01_02 P01_03 P01_04; do
        aria2c -x 16 -s 16 \
            -d "${LOCAL_DIR}/epic/raw/" \
            -o "${video_id}.MP4" \
            "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P01/${video_id}.MP4"
    done

    echo "Splitting and transcoding EPIC-KITCHENS to 20s clips at 256p @ 4fps..."
    job_count=0
    for raw in "${LOCAL_DIR}/epic/raw/"*.MP4; do
        (
            base=$(basename "${raw%.*}")
            duration=$(ffprobe -v error \
                -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 \
                "$raw")
            num_segments=$(python3 -c "import math; print(math.ceil($duration / 20))")
 
            for i in $(seq 0 $((num_segments - 1))); do
                start=$((i * 20))
                out="${EPIC_LOCAL}/${base}_clip_$(printf '%04d' $i).mp4"
                ffmpeg -y \
                    -ss "$start" \
                    -i "$raw" \
                    -t 20 \
                    -vf "scale=256:-2" \
                    -r 4 \
                    -c:v h264_nvenc -preset p4 -cq 23 \
                    -an \
                    "$out" 2>/dev/null
                echo "  ${base} segment $((i+1))/${num_segments}"
            done
        ) &
        ((job_count++))
        if [ "$job_count" -ge 3 ]; then
            wait -n
            ((job_count--))
        fi
    done
    wait

    rm -rf "${LOCAL_DIR}/epic/raw/"

    save_to_drive "$EPIC_LOCAL" "epic_clips"
fi

echo ""

echo "  BDD256 clips : $BDD_LOCAL"
echo "  EPIC clips   : $EPIC_LOCAL"

echo ""