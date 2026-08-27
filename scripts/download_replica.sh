#!/bin/bash
# Download Replica dataset (iMAP/NICE-SLAM format)
# Reference: https://github.com/cvg/nice-slam
set -e

DATA_DIR="${1:-datasets/Replica}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "Replica Dataset Downloader"
echo "Target directory: $DATA_DIR"
echo "============================================"

# Check if already downloaded
if [ -d "$DATA_DIR/office0/results" ]; then
    echo "Dataset already exists at $DATA_DIR/office0"
    echo "Skipping download."
    exit 0
fi

# Download from NICE-SLAM preprocessed Replica
REPLICA_URL="https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"

echo "Downloading Replica dataset (~5 GB)..."
echo "URL: $REPLICA_URL"

if command -v wget &> /dev/null; then
    wget -c "$REPLICA_URL" -O "$DATA_DIR/Replica.zip"
elif command -v curl &> /dev/null; then
    curl -L -C - "$REPLICA_URL" -o "$DATA_DIR/Replica.zip"
else
    echo "ERROR: wget or curl required"
    exit 1
fi

echo "Extracting..."
cd "$DATA_DIR"
unzip -q Replica.zip
rm -f Replica.zip

echo ""
echo "Done! Available scenes:"
ls -d */ 2>/dev/null || echo "(check $DATA_DIR for extracted scenes)"
echo ""
echo "Usage example:"
echo "  python scripts/run_pipeline.py --config configs/replica.yaml --data_path $DATA_DIR/office0"
