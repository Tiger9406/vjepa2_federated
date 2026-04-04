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
echo "Downloading EPIC-KITCHENS P01 raw videos..."

git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
cd ./epic-kitchens-download-scripts/
python ./epic_downloader.py --videos --participants P01 --output-path ./../fed_training/epic/

echo "EPIC-KITCHENS ready."