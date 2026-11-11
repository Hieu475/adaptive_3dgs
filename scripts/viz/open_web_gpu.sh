#!/bin/bash
# Launch browser forced onto NVIDIA discrete GPU (RTX 4050) with WebGPU & hardware acceleration
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

URL="${1:-https://playcanvas.com/supersplat/editor}"

echo "=================================================================="
echo "  Launching Browser with NVIDIA GPU Hardware Acceleration"
echo "  Target URL: $URL"
echo "  GPU: NVIDIA GeForce RTX 4050 Laptop GPU"
echo "=================================================================="

if command -v google-chrome &> /dev/null; then
    exec google-chrome \
        --ignore-gpu-blocklist \
        --enable-gpu-rasterization \
        --enable-zero-copy \
        --enable-features=WebGPU \
        "$URL"
elif command -v chromium &> /dev/null; then
    exec chromium \
        --ignore-gpu-blocklist \
        --enable-gpu-rasterization \
        --enable-zero-copy \
        --enable-features=WebGPU \
        "$URL"
elif command -v firefox &> /dev/null; then
    exec firefox "$URL"
else
    echo "No supported browser found."
fi
