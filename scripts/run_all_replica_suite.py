#!/usr/bin/env python3
"""Run full Replica benchmark suite across all 8 canonical SLAM sequences."""
import sys
import json
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES = [
    "replica_office0",
    "replica_office1",
    "replica_office2",
    "replica_office3",
    "replica_office4",
    "replica_room0",
    "replica_room1",
    "replica_room2",
]

def main():
    print("=" * 80)
    print("        ADAPTIVE 3DGS — FULL REPLICA BENCHMARK SUITE RUNNER")
    print("=" * 80)
    print(f"Total scenes: {len(SCENES)}")
    print("Scenes:", ", ".join(SCENES))
    print("=" * 80)

    results = {}
    total_start = time.time()

    for idx, scene in enumerate(SCENES, 1):
        out_dir = REPO_ROOT / "results" / "demo_output" / scene
        summary_file = out_dir / "demo_summary.json"
        
        # If already run recently, load summary
        if summary_file.exists():
            print(f"\n[{idx}/{len(SCENES)}] Loading existing run for {scene}...")
            try:
                with open(summary_file) as f:
                    sdata = json.load(f)
                results[scene] = sdata
                print(f"   -> PSNR: {sdata.get('final_psnr_db', sdata.get('mean_psnr_db')):.2f} dB | SSIM: {sdata.get('final_ssim', sdata.get('mean_ssim')):.4f} | FPS: {sdata.get('mean_fps'):.1f} | Splats: {sdata.get('final_gaussians', 0):,}")
                continue
            except Exception:
                pass

        print(f"\n[{idx}/{len(SCENES)}] Running Online Reconstruction on {scene} (50 frames)...")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_demo.py"),
            "--scene", scene,
            "--n_frames", "50",
            "--steps", "5",
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and summary_file.exists():
            with open(summary_file) as f:
                sdata = json.load(f)
            results[scene] = sdata
            print(f"   -> SUCCESS! PSNR: {sdata.get('final_psnr_db'):.2f} dB | SSIM: {sdata.get('final_ssim'):.4f} | FPS: {sdata.get('mean_fps'):.1f}")
        else:
            print(f"   -> Warning: Run failed or summary missing for {scene}")

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"  ALL REPLICA SCENES COMPLETED IN {total_time:.1f}s!")
    print("=" * 80)

    # Print summary table
    print(f"\n{'Scene':<20} | {'Mean PSNR':<10} | {'Final PSNR':<10} | {'SSIM':<8} | {'FPS':<6} | {'Splats':<10}")
    print("-" * 75)
    for s, d in results.items():
        m_psnr = d.get('mean_psnr_db', 0.0)
        f_psnr = d.get('final_psnr_db', 0.0)
        ssim = d.get('final_ssim', d.get('mean_ssim', 0.0))
        fps = d.get('mean_fps', 0.0)
        gauss = d.get('final_gaussians', 0)
        print(f"{s:<20} | {m_psnr:<10.2f} | {f_psnr:<10.2f} | {ssim:<8.4f} | {fps:<6.1f} | {gauss:<10,}")

    # Save suite report
    suite_report_path = REPO_ROOT / "results" / "replica_suite_summary.json"
    with open(suite_report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to: {suite_report_path}")

if __name__ == "__main__":
    main()
