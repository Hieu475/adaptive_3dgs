# Replication artifacts

The checked-in JSON records, manifests, validation report, and analysis are the
result bundle. Frozen `.pt` prefix checkpoints and candidate batches are kept
out of Git because they add about 100 MB. The run records retain their SHA-256
hashes. Recreate the binary artifacts and runs with:

```bash
python experiments/run_matched_candidate_timing_replay.py \
  --scenes replica_room2 replica_office3 \
  --seeds 58 59 60 \
  --allocation_modes global stale_rescue \
  --stale_fraction 0.50 --stale_after_frames 10 --mature_age_frames 10 \
  --n_frames 119 --candidate_frame 30 --early_insert_frame 30 \
  --late_insert_frame 90 --candidate_count 5000 --micro_updates 5000 \
  --heldout_frames 20 40 60 80 100 119 \
  --out_dir results/matched_candidate_timing_rescue_replication \
  --run_id independent_seed --force
```

The scene datasets and a CUDA-capable PyTorch/gsplat environment are required.
The checked-in run records were generated from a dirty worktree; use the
`source_hashes.json` and the per-run artifact hashes to identify the exact
experiment implementation that produced them.
