#!/bin/bash
# Downloads and prepares the dataset. Usage: bash scripts/setup_dataset.sh YOUR_KAGGLE_TOKEN
set -e
TOKEN="$1"

mkdir -p ~/.kaggle
echo "$TOKEN" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

kaggle datasets download -d ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes
mkdir -p data/raw
unzip -q -o mri-for-brain-tumor-with-bounding-boxes.zip -d data/raw
python -m src.data.merge_dataset --src data/raw --dst data/yolo
