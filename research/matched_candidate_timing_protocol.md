# Matched-candidate timing replay protocol

## Question

For the same set of Gaussian parameters, does inserting the batch earlier
change reconstruction quality because it is available for more of the fixed
optimization trajectory?

## Intervention

Build one corrected-pose prefix through frame 29. At frame 30, sample one
uniform-valid-depth candidate set and materialize its complete Gaussian
parameter batch once. Deep-copy the same prefix into two branches. Insert the
same batch at frame 30 in the early branch and at frame 90 in the late branch.
Disable all other births. Process the same trajectory frames in both branches
with the same exact per-frame Gaussian micro-update quota and the same RNG
state. Exclude the fixed held-out evaluation frames from mapping.

This replay deliberately holds candidate locations and initial Gaussian
parameters fixed. It estimates the effect of giving that fixed batch additional
frames in the map; it does not estimate the natural online value of discovering
new surfaces at later observations.

## Locked outcomes

Primary outcomes are final-map mean PSNR and Depth L1 over the same held-out
views. Secondary outcomes are trajectory mean PSNR/Depth L1, mature update
coverage, synchronized optimization-stage time, and synchronized frame latency.
Report PSNR and depth together. Treat endpoint and trajectory-average values as
separate outcomes.

## Validity gates

- Both arms start from identical prefix map state and RNG state.
- Candidate parameter-batch hashes match exactly.
- Both arms insert exactly the requested candidate count.
- No endogenous births occur in either branch.
- Every mapped frame receives the same exact micro-update count in both arms.
- Both arms process the same non-held-out frames and score the same held-out views.
- Do not require final map cardinality to match: pruning and resulting map size
  can respond to the timing intervention and are outcomes, not baseline controls.
- Report the actual synchronization method and measured latencies separately
  from the modeled optimization budget.

Any failed gate invalidates the pair. A valid single-seed run is a feasibility
pilot, not confirmatory evidence. If the pilot is valid, extend to at least four
seeds on the two Replica scenes and then to prespecified held-out scenes without
changing the dose, timing, or metric definitions.
