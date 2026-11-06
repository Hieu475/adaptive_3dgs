# Within-scene depth interventions

`python experiments/run_depth_intervention.py` runs the two fixed policies on
the same perturbed RGB-D frames, 150 evaluated frames and three seeds. RGB and
poses stay fixed. Replica `office0` receives independent random depth holes at
0, 15, or 30% of pixels plus zero-mean Gaussian depth noise with standard
deviation `0.05 * hole_fraction` meters. TUM `fr2_xyz` uses the standard
loader hygiene, then blends an edge-preserving bilateral filter on valid depth
pixels at strengths 0, 0.5, and 1; its invalid pixels remain masked.

The comparison estimates fixed policy ordering within each perturbed scene.
Random holes are not a full Kinect noise model, and bilateral filtering of
already-hygiened TUM depth does not remove missing measurements. The available
datasets contain no third sensor domain or scene outside the paper's 14-scene
set for independent threshold recalibration.

Each JSON contains the full config, frame trace, and post-intervention hole
rate. `python experiments/analyze_depth_intervention.py` writes paired
seed-bootstrap summaries to `analysis.json`.
