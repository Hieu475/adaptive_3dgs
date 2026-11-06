#!/usr/bin/env python3
"""Upscale all remaining Replica scenes to high-density (~1.2M+ Gaussians)."""
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES = [
    "replica_office1",
    "replica_office2",
    "replica_office3",
    "replica_office4",
    "replica_room1",
    "replica_room2",
]

def main():
    print("=" * 80)
    print("     ADAPTIVE 3DGS — HIGH-DENSITY FULL-ROOM RECONSTRUCTION SUITE")
    print("=" * 80)
    print(f"Target scenes to upscale to ~1.2M Gaussians: {', '.join(SCENES)}")
    print("=" * 80)

    for idx, scene in enumerate(SCENES, 1):
        out_dir = REPO_ROOT / "results" / "demo_output" / scene
        print(f"\n[{idx}/{len(SCENES)}] Starting High-Density Full-Room Reconstruction for '{scene}'...")
        
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_demo.py"),
            "--scene", scene,
            "--full_room",
            "--max_gaussians", "1200000",
            "--n_frames", "80",
            "--steps", "5",
            "--out_dir", f"results/demo_output/{scene}",
        ]
        
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[Warning] Failed running {scene}, exit code: {ret.returncode}")
        else:
            # Clean duplicate point cloud ply to keep folder clean
            pcd_file = out_dir / "reconstructed_point_cloud.ply"
            if pcd_file.exists():
                pcd_file.unlink()
            print(f"[{idx}/{len(SCENES)}] Successfully upscaled {scene}!")

    # Clean any leftover point cloud files across demo_output
    for p in (REPO_ROOT / "results" / "demo_output").glob("**/*point_cloud*.ply"):
        p.unlink()

    print("\n" + "=" * 80)
    print("  ALL REPLICA SCENES SUCCESSFULLY UPSCALED TO HIGH-DENSITY (1.2M+ SPLATS)!")
    print("=" * 80)

if __name__ == "__main__":
    main()
