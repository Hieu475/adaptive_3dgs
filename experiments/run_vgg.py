#!/usr/bin/env python3
"""Value-gated growth (VGG) vs fixed controls, with per-frame timing.

Conditions: error_off, spawn_cap_always20 (fixed x0.2 cap), vgg, vgg_prop (VGG + map-proportional budget).
Every run is written immediately; theta defaults can be overridden for tuning on development scenes.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_throttle_ablation as abl  # noqa: E402  (installs the proportional-budget wrapper)
from experiments.run_dilution_factorial import summarize  # noqa: E402
from experiments.run_frozen_benchmark import load_benchmark_sequence, run_policy_trajectory  # noqa: E402
from src.control.growth_controller import VGGParams, VGGState, step  # noqa: E402
from src.core.pipeline import OnlineReconstructionPipeline  # noqa: E402
from src.control import time_budget as tb  # noqa: E402

CONDITIONS = ("error_off", "spawn_cap_always20", "error_budget_prop", "vgg", "vgg_prop", "tt", "tt_vgg", "tt_cap", "tts", "tts_vgg", "error_off_opt", "tt_vgg_opt")
STATES = []
TT_STATES = []
_prev = OnlineReconstructionPipeline.process_frame


def _process(self, rgb, depth, gt_pose=None, *a, **k):
    tt = self.config.get("time_target")
    if tt:
        if not hasattr(self, "_tt_state"):
            self._tt_state = tb.TTState()
            self._tt_params = tb.TTParams(**{kk: vv for kk, vv in tt.items() if kk in tb.TTParams.__dataclass_fields__})
            TT_STATES.append(self._tt_state)
        tb.before(self, self._tt_params, self._tt_state)
    gc = self.config.get("growth_controller")
    if gc:
        if not hasattr(self, "_vgg_state"):
            self._vgg_state = VGGState()
            self._vgg_params = VGGParams(**{kk: vv for kk, vv in gc.items() if kk in VGGParams.__dataclass_fields__})
            STATES.append(self._vgg_state)
        step(self, rgb, depth, gt_pose, self._vgg_params, self._vgg_state)
    out = _prev(self, rgb, depth, gt_pose, *a, **k)
    if tt:
        tb.after(self, self._tt_params, self._tt_state)
    return out


OnlineReconstructionPipeline.process_frame = _process


def config_for(cond, seed, budget_ms, device, vgg_overrides=None, tt_overrides=None):
    if cond in ("error_off_opt", "tt_vgg_opt"):  # original budgeted controllers with the steps-regime optimizer
        policy, cfg = config_for(cond[:-4], seed, budget_ms, device, vgg_overrides, tt_overrides)
        tb.apply_steps_regime(cfg)
        return policy, cfg
    if cond in ("error_off", "spawn_cap_always20", "error_budget_prop"):
        return abl.config_for(cond, seed, budget_ms, device)
    if cond in ("tts", "tts_vgg"):
        policy, cfg = config_for("vgg" if cond == "tts_vgg" else "error_off", seed, budget_ms, device, vgg_overrides)
        cfg["scheduler"]["densify_budget_override_ms"] = budget_ms
        cfg["time_target"] = {**(tt_overrides or {}), "mode": "steps"}
        tb.apply_steps_regime(cfg)
        return policy, cfg
    if cond in ("tt", "tt_cap"):
        policy, cfg = abl.config_for("error_off" if cond == "tt" else "spawn_cap_always20", seed, budget_ms, device)
        cfg["scheduler"]["densify_budget_override_ms"] = budget_ms
        cfg["time_target"] = dict(tt_overrides or {})
        return policy, cfg
    if cond == "tt_vgg":
        policy, cfg = config_for("vgg", seed, budget_ms, device, vgg_overrides)
        cfg["scheduler"]["densify_budget_override_ms"] = budget_ms
        cfg["time_target"] = dict(tt_overrides or {})
        return policy, cfg
    policy, cfg = abl.config_for("spawn_cap_always20", seed, budget_ms, device)
    cfg["densification"]["throttle_factor"] = 1.0
    cfg["growth_controller"] = dict(vgg_overrides or {})
    if cond == "vgg_prop":
        cfg["scheduler"]["budget_prop_to_map"] = True
        cfg["scheduler"]["densify_budget_override_ms"] = budget_ms
    return policy, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--theta_lo", type=float, default=0.08)
    ap.add_argument("--theta_hi", type=float, default=0.20)
    ap.add_argument("--g_min", type=float, default=0.1)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--ema", type=float, default=0.7)
    ap.add_argument("--down", type=float, default=0.25)
    ap.add_argument("--up", type=float, default=2.0)
    ap.add_argument("--min_recent", type=int, default=200)
    ap.add_argument("--backoff_max", type=int, default=8)
    ap.add_argument("--T_ms", type=float, default=100.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--m_max", type=float, default=16.0)
    ap.add_argument("--n_micro", type=int, default=None, help="override training.n_micro_steps (max optimization steps per Gaussian per frame)")
    ap.add_argument("--n_max", type=float, default=24.0, help="steps mode: maximum optimization steps per frame")
    ap.add_argument("--lr_mult", type=float, default=1.0, help="multiply all learning rates")
    ap.add_argument("--tag_all", action="store_true", help="append --tag to every condition name")
    ap.add_argument("--n_min", type=float, default=2.0)
    ap.add_argument("--n_init", type=float, default=5.0)
    ap.add_argument("--lr_groups", default="", help="per-group multipliers on top of --lr_mult, e.g. position:2,sh:0.5")
    ap.add_argument("--loss_w", default="", help="loss weight overrides, e.g. ssim:0.4,depth:1")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. densification.max_new_per_frame=1000")
    ap.add_argument("--preset_off", nargs="*", default=[], choices=["lr", "betas", "decay"], help="ablate parts of the steps-regime optimizer preset")
    ap.add_argument("--offset", type=int, default=0, help="start the sequence at this frame (fresh window)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "vgg")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    ov = {"theta_lo": a.theta_lo, "theta_hi": a.theta_hi, "g_min": a.g_min, "every": a.every, "ema": a.ema,
          "down": a.down, "up": a.up, "min_recent": a.min_recent, "backoff_max": a.backoff_max}
    for sc in a.scenes:
        frames, intr = load_benchmark_sequence(scene_name=sc, n_frames=a.offset + a.n_frames + 1, H=240, W=320, device="cuda")
        frames = frames[a.offset:]
        for seed in a.seeds:
            for cond in a.conditions:
                name = cond + (a.tag if (cond.startswith("vgg") or cond.startswith("tt") or a.tag_all) else "")
                target = a.out_dir / sc / f"{name}_seed{seed}.json"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                policy, cfg = config_for(cond, seed, 15.0, "cuda", ov, {"T_ms": a.T_ms, "gamma": a.gamma, "m_max": a.m_max, "n_max": a.n_max, "n_min": a.n_min, "n_init": a.n_init})
                if a.lr_mult != 1.0:
                    lr = cfg["training"].setdefault("learning_rate", {"position": 4.0e-4, "scale": 5e-3, "rotation": 1e-3, "opacity": 5e-2, "sh": 1.0e-2})
                    cfg["training"]["learning_rate"] = {k: v * a.lr_mult for k, v in lr.items()}
                if a.lr_groups:
                    for kv in a.lr_groups.split(","):
                        g, v = kv.split(":")
                        cfg["training"]["learning_rate"][g] *= float(v)
                if a.loss_w:
                    cfg.setdefault("losses", {})
                    for kv in a.loss_w.split(","):
                        g, v = kv.split(":")
                        cfg["losses"]["weight_" + g] = float(v)
                for part in a.preset_off:
                    tr = cfg["training"]
                    if part == "lr":
                        tr["learning_rate"] = {"position": 4.0e-4, "scale": 5e-3, "rotation": 1e-3, "opacity": 5e-2, "sh": 1.0e-2}
                    elif part == "betas":
                        tr.pop("adam_betas", None)
                    elif part == "decay":
                        tr.pop("lr_in_frame", None)
                for kv in a.set:
                    path, val = kv.split("=")
                    node = cfg
                    keys = path.split(".")
                    for k in keys[:-1]:
                        node = node.setdefault(k, {})
                    if "," in val:
                        node[keys[-1]] = [float(x) for x in val.split(",")]
                    else:
                        node[keys[-1]] = type(node.get(keys[-1], 0.0))(float(val)) if not isinstance(node.get(keys[-1]), bool) else bool(float(val))
                if a.n_micro is not None:
                    cfg["training"]["n_micro_steps"] = a.n_micro
                abl.ext.BUDGET_SCALES.clear()
                STATES.clear()
                TT_STATES.clear()
                print(f"RUN {sc} {name} seed={seed}", flush=True)
                summ, logs = run_policy_trajectory(policy=policy, seed=seed, frames=frames, intrinsics=intr,
                                                   budget_ms=15.0, device="cuda", W=320, H=240, custom_config=cfg)
                res = summarize(summ, logs)
                res["mean_frame_ms"] = float(sum(r["frame_time_ms"] for r in logs) / len(logs))
                hist = STATES[0].history if STATES else []
                rec = {"scene": sc, "seed": seed, "condition": name, "summary": res, "overrides": ov if cond.startswith("vgg") else None,
                       "g_history": hist, "n_frames": len(logs), "offset": a.offset,
                       "tt_m_final": (TT_STATES[0].m if TT_STATES else None)}
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec))
                tmp.replace(target)
                gfin = hist[-1]["g"] if hist else 1.0
                print(f"DONE {sc} {name} seed={seed}: PSNR={summ['mean_psnr']:.3f} map={summ['N_final']} "
                      f"frame={res['mean_frame_ms']:.0f}ms g_final={gfin} m_final={(TT_STATES[0].m if TT_STATES else 1.0):.2f}", flush=True)


if __name__ == "__main__":
    main()
