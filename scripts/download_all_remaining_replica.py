#!/usr/bin/env python3
"""Download all remaining Replica SLAM scenes (room1, office2, office3, office4)."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES_TO_DOWNLOAD = ["room1", "office2", "office3", "office4"]

def main():
    print("=" * 70)
    print("  DOWNLOADING ALL REMAINING REPLICA SLAM BENCHMARK SCENES")
    print(f"  Target scenes: {', '.join(SCENES_TO_DOWNLOAD)}")
    print("=" * 70)

    for idx, scene in enumerate(SCENES_TO_DOWNLOAD, 1):
        target_dir = REPO_ROOT / "datasets" / "Replica" / scene
        if (target_dir / ".done").exists():
            print(f"\n[{idx}/{len(SCENES_TO_DOWNLOAD)}] Scene '{scene}' already downloaded. Skipping.")
            continue

        print(f"\n[{idx}/{len(SCENES_TO_DOWNLOAD)}] Downloading '{scene}' (120 keyframes)...")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "download_replica_scene.py"),
            "--scene", scene,
            "--n_frames", "120",
            "--workers", "4",
        ]
        ret = subprocess.run(cmd, check=True)
        if ret.returncode != 0:
            print(f"Error downloading {scene}!")
            sys.exit(ret.returncode)
        import time
        time.sleep(3.0)

    print("\n" + "=" * 70)
    print("  ALL REPLICA BENCHMARK SCENES SUCCESSFULLY DOWNLOADED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
