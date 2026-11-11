#!/usr/bin/env python3
"""Compare matched-map E1 pruning selectors on identical held-out Replica seeds."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def read(scene, seed, selector):
    suffix = "heldout" if selector == "lowest_opacity" else "random_prune"
    path = RESULTS / "replica" / f"replica_e1_dynamic_matched_map_{scene}_seed{seed}_{suffix}" / "dynamic_matched_map_results.json"
    d = json.loads(path.read_text())
    s = d["summary"]
    return {
        "scene": scene, "seed": seed, "selector": selector,
        "gate": bool(s["matched_count_gate_passed_every_frame"]),
        "frames": int(s["processed_frames"]), "requested": int(s.get("requested_online_frames", d["online_frames"])),
        "mean_n": s["mean_matched_gaussians"],
        "pre_psnr": s["mean_pre_match_delta_coverage_minus_no_throttle_psnr"],
        "post_psnr": s["mean_delta_coverage_minus_no_throttle_psnr"],
        "prune_no": s["mean_matching_prune_delta_no_throttle_psnr"],
        "prune_cov": s["mean_matching_prune_delta_coverage_throttle_psnr"],
        "depth": s["mean_delta_coverage_minus_no_throttle_depth_l1"],
        "mature": s["mean_delta_coverage_minus_no_throttle_mature_update_coverage"],
        "wins": s["paired_frames_coverage_psnr_wins"],
        "losses": s["paired_frames_coverage_psnr_losses"],
    }


def avg(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows)


def replay_failure(seed):
    path = RESULTS / "replica" / f"replica_e1_dynamic_matched_map_office1_seed{seed}_random_common_target" / "dynamic_matched_map_results.json"
    d = json.loads(path.read_text())
    return d["summary"], d.get("count_schedule_gate_failure")


def main():
    rows = [read("office1", seed, selector)
            for seed in (57, 58) for selector in ("lowest_opacity", "uniform_random")]
    lines = [
        "# E1 matched-map pruning-selector sensitivity: office1", "",
        "Same scene and held-out seeds, same policies, and the same per-frame endogenous lower-count rule. The pruning selector is changed; every run passed its within-run exact count gate. The realized target paths can differ because pruning changes the next-frame maps.", "",
        "| Seed | Prune selector | Gate | Mean N | PSNR Δ before match | PSNR Δ after match | No-throttle prune effect | Coverage prune effect | Depth L1 Δ | Mature coverage Δ | PSNR wins/losses |", "|--:|:--|:--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['selector']} | {'PASS' if r['gate'] else 'FAIL'} | {r['mean_n']:.0f} | {r['pre_psnr']:+.3f} dB | {r['post_psnr']:+.3f} dB | {r['prune_no']:+.3f} dB | {r['prune_cov']:+.3f} dB | {r['depth']:+.5f} m | {r['mature']:+.5f} | {r['wins']}/{r['losses']} |")
    grouped = {selector: [r for r in rows if r["selector"] == selector]
               for selector in ("lowest_opacity", "uniform_random")}
    replay = {seed: replay_failure(seed) for seed in (57, 58)}
    low, rand = grouped["lowest_opacity"], grouped["uniform_random"]
    psnr_flip = avg(low, "post_psnr") - avg(rand, "post_psnr")
    lines += ["", "## Two-seed selector contrast", "",
              f"- Lowest-opacity prune: mean post-match PSNR Δ {avg(low, 'post_psnr'):+.3f} dB; mean pre-match Δ {avg(low, 'pre_psnr'):+.3f} dB; mean no-throttle pruning effect {avg(low, 'prune_no'):+.3f} dB.",
              f"- Uniform-random prune: mean post-match PSNR Δ {avg(rand, 'post_psnr'):+.3f} dB; mean pre-match Δ {avg(rand, 'pre_psnr'):+.3f} dB; mean no-throttle pruning effect {avg(rand, 'prune_no'):+.3f} dB.",
              f"- Post-match PSNR contrast between selectors: {psnr_flip:+.3f} dB (lowest-opacity minus random). Both held-out seeds change the post-match sign when the selector changes.", "",
              "## Same-seed frozen target replay", "",
              "The per-frame target path was copied from the lowest-opacity run for each seed, then replayed with uniform-random pruning. Both runs failed the exact target gate at frame 12; prefix quality is excluded.", "",
              "| Seed | Frames completed | Failure frame | Target | Realized counts |", "|--:|--:|--:|--:|:--|"]
    for seed, (summary, failure) in replay.items():
        lines.append(f"| {seed} | {summary['processed_frames']}/{summary['requested_online_frames']} | {failure['frame']} | {failure['requested_target']} | {failure['realized_counts']} |")
    lines += [
              "## Interpretation", "",
              "The equal-count result is strongly dependent on how extra Gaussians are removed. On both held-out seeds, lowest-opacity pruning yields a positive coverage-minus-no-throttle PSNR delta, while uniform-random pruning yields a negative delta. Random pruning also worsens depth L1 and changes mature-update coverage. The same-seed frozen-target replay fails at frame 12 under random pruning, showing that the reference target path is not feasible after changing the selector. Thus we have selector sensitivity and a feasibility failure, not a completed exact same-target selector ablation. Do not use the opacity-matched positive PSNR as a causal claim that coverage throttling improves quality at equal map size.", "",
              "Uniform-random pruning is a sensitivity control, not a recommended map-pruning policy. This is still one Replica scene and two seeds.", ""]
    out = RESULTS / "replica" / "replica_e1_prune_selector_sensitivity_office1.md"
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
