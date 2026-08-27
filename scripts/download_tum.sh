#!/bin/bash
# Download TUM RGB-D benchmark sequences
# Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
set -e

DATA_DIR="${1:-datasets/TUM}"
SEQUENCE="${2:-rgbd_dataset_freiburg1_desk}"

mkdir -p "$DATA_DIR"

echo "============================================"
echo "TUM RGB-D Dataset Downloader"
echo "Target directory: $DATA_DIR"
echo "Sequence: $SEQUENCE"
echo "============================================"

# Check if already downloaded
if [ -d "$DATA_DIR/$SEQUENCE" ]; then
    echo "Sequence already exists at $DATA_DIR/$SEQUENCE"
    echo "Skipping download."
    exit 0
fi

BASE_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg1"
FILE="$SEQUENCE.tgz"

echo "Downloading $SEQUENCE..."

if command -v wget &> /dev/null; then
    wget -c "$BASE_URL/$FILE" -O "$DATA_DIR/$FILE"
elif command -v curl &> /dev/null; then
    curl -L -C - "$BASE_URL/$FILE" -o "$DATA_DIR/$FILE"
else
    echo "ERROR: wget or curl required"
    exit 1
fi

echo "Extracting..."
cd "$DATA_DIR"
tar xzf "$FILE"
rm -f "$FILE"

# Generate associations file if not present
if [ ! -f "$SEQUENCE/associations.txt" ]; then
    echo "Generating associations.txt..."
    python3 -c "
import os, sys
rgb_dir = '$SEQUENCE/rgb'
depth_dir = '$SEQUENCE/depth'
rgb_files = sorted(os.listdir(rgb_dir))
depth_files = sorted(os.listdir(depth_dir))
with open('$SEQUENCE/associations.txt', 'w') as f:
    for rf, df in zip(rgb_files, depth_files):
        rt = rf.replace('.png', '')
        dt = df.replace('.png', '')
        f.write(f'{rt} rgb/{rf} {dt} depth/{df}\n')
print(f'Generated {min(len(rgb_files), len(depth_files))} associations')
"
fi

echo ""
echo "Done! Sequence at: $DATA_DIR/$SEQUENCE"
echo ""
echo "Usage example:"
echo "  python scripts/run_pipeline.py --config configs/tum_rgbd.yaml --data_path $DATA_DIR/$SEQUENCE --dataset_type tum"
