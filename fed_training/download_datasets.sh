#!/bin/bash

# exit if any command fails
set -e

sudo apt-get install -y aria2

echo "Starting dataset downloads for Federated Training..."

# BDD100K
echo "Downloading BDD100K"
mkdir -p ./fed_training/bdd/

echo "Downloading BDD100K Train 00"
# -c flag fir resuming the download if it gets interrupted
aria2c -x 16 -s 16 -d /dev/shm -o bdd100k_videos_train_00.zip http://128.32.162.150/bdd100k/video_parts/bdd100k_videos_train_00.zip

echo "Extracting BDD100K videos"
unzip /dev/shm/bdd100k_videos_train_00.zip -d ./fed_training/bdd/

rm /dev/shm/bdd100k_videos_train_00.zip
echo "BDD100K ready!!"

# Epic kitchen
echo "Downloading EPIC-KITCHENS"
echo "Downloading EPIC-KITCHENS P01 raw videos 1 and 2"

mkdir -p /content/vjepa2_federated/fed_training/epic/EPIC-KITCHENS/P01/videos/

# Download Video 1 (P01_01)
aria2c -x 16 -s 16 \
  -d /content/vjepa2_federated/fed_training/epic/EPIC-KITCHENS/P01/videos/ \
  -o P01_01.MP4 \
  https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P01/P01_01.MP4

# Download Video 2 (P01_02)
aria2c -x 16 -s 16 \
  -d /content/vjepa2_federated/fed_training/epic/EPIC-KITCHENS/P01/videos/ \
  -o P01_02.MP4 \
  https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P01/P01_02.MP4

echo "EPIC-KITCHENS ready."