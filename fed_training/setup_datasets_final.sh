#!/bin/bash

set -e

FED_DRIVE="/content/drive/MyDrive/vjepa/fed_client_datasets"
LOCAL_DIR="/content/vjepa2_federated/fed_training"

mkdir -p "$FED_DRIVE"
mkdir -p "$LOCAL_DIR"

sudo apt-get install -y aria2 ffmpeg unrar unzip

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

restore_client() {
  local client_name="$1"
  local local_dst="${LOCAL_DIR}/${client_name}"
  echo "[${client_name}] Found on Drive. Restoring locally..."
  mkdir -p "$local_dst"
  cp -r "${FED_DRIVE}/${client_name}/." "$local_dst/"
  echo "  restored $(find "$local_dst" -name "*.mp4" | wc -l) clips to ${local_dst}"
}

for participant in P01 P02 P04; do
    CLIENT_NAME="client_epic_${participant}"
    CLIENT_DRIVE="${FED_DRIVE}/${CLIENT_NAME}"

    if [ -d "$CLIENT_DRIVE" ] && [ "$(ls -A "$CLIENT_DRIVE")" ]; then
        echo "[$CLIENT_NAME] Already in Drive. copying."
        restore_client "$CLIENT_NAME"
        continue
    fi

    echo "Processing $CLIENT_NAME..."
    mkdir -p "${LOCAL_DIR}/raw_epic_${participant}"
    mkdir -p "${LOCAL_DIR}/${CLIENT_NAME}"

    # Download 15 videos per participant
    for i in $(seq -w 1 15); do
        aria2c -x 16 -s 16 -d "${LOCAL_DIR}/raw_epic_${participant}" \
            -o "${participant}_${i}.MP4" \
            "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/${participant}/${participant}_${i}.MP4" || true
    done

    # Split into 20s chunks and transcode
    for raw in "${LOCAL_DIR}/raw_epic_${participant}"/*.MP4; do
        [ -e "$raw" ] || continue
        base=$(basename "${raw%.*}")
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$raw")
        num_segments=$(python3 -c "import math; print(math.ceil($duration / 20))")

        for i in $(seq 0 $((num_segments - 1))); do
            start=$((i * 20))
            out="${LOCAL_DIR}/${CLIENT_NAME}/${base}_clip_$(printf '%04d' $i).mp4"
            ffmpeg -y -ss "$start" -i "$raw" -t 20 -vf "scale=256:-2" -r 4 -c:v h264_nvenc -preset p4 -cq 23 -an "$out" 2>/dev/null
        done
    done

    # move to drive and clean up raw files
    cp -r "${LOCAL_DIR}/${CLIENT_NAME}/." "${CLIENT_DRIVE}/"
    rm -rf "${LOCAL_DIR}/raw_epic_${participant}"
done

for zip_idx in 00 01 02 03 04; do
    CLIENT_NAME="client_bdd_${zip_idx}"
    CLIENT_DRIVE="${FED_DRIVE}/${CLIENT_NAME}"

    if [ -d "$CLIENT_DRIVE" ] && [ "$(ls -A "$CLIENT_DRIVE")" ]; then
        echo "[$CLIENT_NAME] Already in Drive. copying"
        restore_client "$CLIENT_NAME"
        continue
    fi

    echo "Processing $CLIENT_NAME..."
    mkdir -p "${LOCAL_DIR}/raw_bdd_${zip_idx}"
    mkdir -p "${LOCAL_DIR}/${CLIENT_NAME}"

    aria2c -x 16 -s 16 -d "${LOCAL_DIR}" -o "bdd_${zip_idx}.zip" \
        "http://128.32.162.150/bdd100k/video_parts/bdd100k_videos_train_${zip_idx}.zip"

    unzip -q "${LOCAL_DIR}/bdd_${zip_idx}.zip" -d "${LOCAL_DIR}/raw_bdd_${zip_idx}/"
    rm "${LOCAL_DIR}/bdd_${zip_idx}.zip"

    # Limit to 1200 videos to fit the 200-1500 constraint
    find "${LOCAL_DIR}/raw_bdd_${zip_idx}" \( -name "*.mov" -o -name "*.mp4" \) | head -n 1200 \
        | xargs -P 8 -I{} bash -c 'transcode_video "$1" "'"${LOCAL_DIR}/${CLIENT_NAME}"'"' _ {}

    cp -r "${LOCAL_DIR}/${CLIENT_NAME}/." "${CLIENT_DRIVE}/"
    rm -rf "${LOCAL_DIR}/raw_bdd_${zip_idx}"
done

all_hmdb_local=true
for i in {0..3}; do
  CLIENT_LOCAL="${LOCAL_DIR}/client_hmdb_${i}"
  CLIENT_DRIVE="${FED_DRIVE}/client_hmdb_${i}"
  if [ -d "$CLIENT_DRIVE" ] && [ "$(ls -A "$CLIENT_DRIVE")" ]; then
    restore_client "client_hmdb_${i}"
  else
    all_hmdb_local=false
  fi
done

if [ "$all_hmdb_local" = false ]; then
HMDB_PROCESSED_CHECK="${FED_DRIVE}/client_hmdb_0"
    echo "Processing HMDB Clients..."
    mkdir -p "${LOCAL_DIR}/raw_hmdb"
    
    aria2c -x 16 -s 16 -d "${LOCAL_DIR}" -o hmdb51_org.rar \
        http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

    unrar x -idq "${LOCAL_DIR}/hmdb51_org.rar" "${LOCAL_DIR}/raw_hmdb/"
    rm "${LOCAL_DIR}/hmdb51_org.rar"

    # Setup 4 client folders
    for i in {0..3}; do
        mkdir -p "${LOCAL_DIR}/raw_hmdb/client_split_${i}"
        mkdir -p "${LOCAL_DIR}/client_hmdb_${i}"
    done

    # Round-robin the 51 category .rar files into the 4 client splits
    idx=0
    for cat_rar in "${LOCAL_DIR}/raw_hmdb/"*.rar; do
        target=$((idx % 4))
        mv "$cat_rar" "${LOCAL_DIR}/raw_hmdb/client_split_${target}/"
        idx=$((idx + 1))
    done

    # Extract and transcode each client's chunk
    for i in {0..3}; do
        CLIENT_DRIVE="${FED_DRIVE}/client_hmdb_${i}"
        CLIENT_LOCAL="${LOCAL_DIR}/client_hmdb_${i}"

        if [ -d "$CLIENT_DRIVE" ] && [ "$(ls -A "$CLIENT_DRIVE")" ]; then
            echo "  [client_hmdb_${i}] Already restored above. Skipping processing."
            continue
        fi

        echo "  Extracting and transcoding HMDB client $i..."
        find "${LOCAL_DIR}/raw_hmdb/client_split_${i}/" -name "*.rar" -execdir unrar x -idq {} \;

        find "${LOCAL_DIR}/raw_hmdb/client_split_${i}/" -name "*.avi" \
        | xargs -P 8 -I{} bash -c 'transcode_video "$1" "'"${CLIENT_LOCAL}"'"' _ {}

        mkdir -p "$CLIENT_DRIVE"
        cp -r "${CLIENT_LOCAL}/." "${CLIENT_DRIVE}/"
    done

    rm -rf "${LOCAL_DIR}/raw_hmdb"
fi

echo "All 12 federated clients processed and available locally."