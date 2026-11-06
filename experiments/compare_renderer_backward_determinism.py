#!/usr/bin/env python3
"""Compare repeatability of gsplat and the PyTorch reference renderer backward."""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gaussian_repr import GaussianModel
from research.rasterizer import render
from research.reproducibility import get_git_commit, get_hardware_info


def tensor_comparison(values):
    reference = values[0].float()
    rows = []
    for repeat_idx, value in enumerate(values[1:], start=1):
        current = value.float()
        diff = (current - reference).abs()
        rows.append({
            "repeat": repeat_idx,
            "bitwise_equal": bool(torch.equal(current, reference)),
            "different_elements": int((current != reference).sum().item()),
            "num_elements": int(current.numel()),
            "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
            "relative_l2_diff": float(
                torch.linalg.vector_norm(diff) /
                (torch.linalg.vector_norm(reference) + 1e-30)
            ),
        })
    return rows


def run_backend(backend, repeats, device, n_gaussians, width, height, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # A dense, overlapping scene makes many pixel threads contribute to each Gaussian.
    xy = torch.randn((n_gaussians, 2), generator=generator) * 0.10
    z = 2.0 + torch.rand((n_gaussians, 1), generator=generator) * 0.5
    points = torch.cat([xy, z], dim=-1).to(device)
    colors = (0.15 + 0.70 * torch.rand((n_gaussians, 3), generator=generator)).to(device)
    target = torch.rand((height, width, 3), generator=generator).to(device)
    K = torch.tensor([
        [36.0, 0.0, width / 2.0],
        [0.0, 36.0, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    pose = torch.eye(4, dtype=torch.float32, device=device)
    captured = {"color": [], "depth": [], "xyz_grad": [], "scale_grad": [],
                "opacity_grad": [], "color_param_grad": []}

    for _ in range(repeats):
        model = GaussianModel(sh_degree=0, device=str(device))
        model.initialize_from_points(
            points.clone(), colors=colors.clone(),
            initial_scale=torch.full((n_gaussians,), 0.10, device=device),
            initial_opacity=0.35,
        )
        output = render(
            means3D=model.positions,
            cov3D=model.build_covariance(),
            colors=model.get_colors(),
            opacities=model.opacities.squeeze(-1),
            extrinsics=pose,
            intrinsics=K,
            image_width=width,
            image_height=height,
            tile_size=16,
            backend=backend,
        )
        color_grad = []
        depth_grad = []
        output["color"].register_hook(lambda grad: color_grad.append(grad.detach().cpu().clone()))
        output["depth"].register_hook(lambda grad: depth_grad.append(grad.detach().cpu().clone()))
        loss = ((output["color"] - target) ** 2).mean() + 0.1 * output["depth"].mean()
        loss.backward()
        captured["color"].append(output["color"].detach().cpu().clone())
        captured["depth"].append(output["depth"].detach().cpu().clone())
        captured["xyz_grad"].append(model._xyz.grad.detach().cpu().clone())
        captured["scale_grad"].append(model._scaling.grad.detach().cpu().clone())
        captured["opacity_grad"].append(model._opacity.grad.detach().cpu().clone())
        captured["color_param_grad"].append(model._features_dc.grad.detach().cpu().clone())
        # Output adjoints are included as independent evidence about loss backward.
        captured.setdefault("render_color_grad", []).append(color_grad[0])
        captured.setdefault("render_depth_grad", []).append(depth_grad[0])
        del model, output, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        name: tensor_comparison(values)
        for name, values in captured.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n_gaussians", type=int, default=96)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="results/gsplat_backward_repeat/backend_control.json")
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("At least two repeats are required.")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    results = {}
    for backend in ("gsplat", "reference"):
        if backend == "gsplat" and device.type != "cuda":
            raise RuntimeError("The gsplat backend control requires CUDA.")
        results[backend] = run_backend(
            backend, args.repeats, device, args.n_gaussians,
            args.width, args.height, args.seed,
        )

    payload = {
        "protocol": "same synthetic overlapping Gaussian scene, recreated identically for each repeat",
        "device": str(device),
        "seed": args.seed,
        "repeats": args.repeats,
        "n_gaussians": args.n_gaussians,
        "resolution": [args.width, args.height],
        "git_commit": get_git_commit(),
        "hardware": get_hardware_info(),
        "results": results,
        "caveat": "Synthetic backward repeatability control; not a mapping quality benchmark.",
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    report = [
        "# Renderer backward repeatability control", "",
        f"Synthetic scene: {args.n_gaussians} overlapping Gaussians at {args.width}x{args.height}; "
        f"{args.repeats} runs per backend, same fixed input and seed {args.seed}.",
        "This tests repeatability within each backend, not numerical equivalence between backends.", "",
        "| Backend | Tensor | Bitwise repeatability | Changed elements | Max absolute difference |",
        "|:---|:---|:---|---:|---:|",
    ]
    for backend, tensors in results.items():
        for name, rows in tensors.items():
            total = sum(row["num_elements"] for row in rows)
            changed = sum(row["different_elements"] for row in rows)
            max_diff = max((row["max_abs_diff"] for row in rows), default=0.0)
            same = all(row["bitwise_equal"] for row in rows)
            report.append(
                f"| {backend} | {name} | {'all repeats equal' if same else 'differs'} | "
                f"{changed}/{total} | {max_diff:.9g} |"
            )
    report.extend([
        "",
        "Observed on this synthetic CUDA control: gsplat forward outputs and loss-output adjoints repeat bitwise, "
        "but per-Gaussian gradients differ across repeats. The reference renderer repeats bitwise for all captured "
        "outputs and gradients.",
        "",
        "Scope: this establishes repeatability behavior only for this small synthetic case. It does not establish "
        "reference-renderer throughput or full-pipeline repeatability, nor does it show that the two backends render identically.",
        "",
    ])
    report_path = out_path.with_suffix(".md")
    report_path.write_text("\n".join(report))

    for backend, tensors in results.items():
        print(f"\n{backend}")
        for name, rows in tensors.items():
            print(f"  {name}: {rows}")
    print(f"Saved: {out_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
