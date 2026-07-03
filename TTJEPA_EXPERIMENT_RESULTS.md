# TTJepa Experiment Results Ledger

Last updated: 2026-07-01 PT

This file preserves the full experiment record. The README now keeps the
public-facing motivation plus the current learned dynamic-K result table. This
ledger is broader: it keeps checkpoint families separate, records diagnostic
raw-MSE analyses, checkpoint paths, logs, planner-feature selectors,
whitened/probe-weighted variants, and exploratory training-time regularization
leads.

Important convention: do not mix rows across checkpoint families. The older
raw-MSE diagnostic results use the original recurrent checkpoints under
`analysis/k_refinement_all_20260620_024634`. The current learned dynamic-K main
results use the \(\tau_{\mathrm{rel}}=0.0005\) relative-improvement checkpoint
family. In discussion this setting is often abbreviated as `rel0005`; the
remote four-dataset artifact directories are named `rel00005`.

## Scope Map

| Thread | Included in Paper 1? | Role |
| --- | --- | --- |
| Learned dynamic K with relative marginal MSE supervision, \(\tau_{\mathrm{rel}}=0.0005\) / `rel0005` shorthand | Yes | Current main deployable dynamic-K method; four-dataset remote dirs use `rel00005` |
| Fixed-depth recurrent TTJepa, `K1/K2/K3/K4` | Yes | Same-checkpoint baselines for dynamic K |
| Raw target-latent MSE diagnostic | Yes | Post-hoc diagnostic showing what latent error can and cannot identify |
| Hindsight K1/K4 success chooser | Yes | Analytical upper bound only |
| Depth-allocation histograms and rollout-step traces | Yes | Shows where learned dynamic K spends compute |
| Latent spectrum and state-probe analysis | Yes | Mechanistic/failure analysis |
| Planner-feature selector | No | Diagnostic evidence that planner traces contain useful signal |
| Cube-triple-only directory named `rel0005` with 80% at all depths | No | Auxiliary training-variant record; name collides with the shorthand above |
| Whitened / probe-weighted halt labels | No | Exploratory alternatives |

## Main Learned Dynamic-K Results: Relative Marginal MSE, \(\tau_{\mathrm{rel}}=0.0005\)

This is the current main learned dynamic-K checkpoint family. In prose we call
this setting `rel0005` because the relative margin is 0.0005; the remote
four-dataset directories use the spelling `rel00005`. The continue head is
trained jointly with the recurrent transition predictor. The label is based on
relative marginal raw-MSE improvement:

```text
(MSE_k - MSE_{k+1}) / (MSE_k + eps) > 0.0005
```

At evaluation time, the future target latent is unavailable; the model uses the
continue head to stop or refine. The reported mean K is the average integer
selected depth over all imagined CEM transition predictions.

| Dataset | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Best learned dynamic K | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher | 80% | 80% | 82% | 84% | 82% | **86%@K1.08** | Beats best fixed K3 while using near-K1 compute |
| Cube single | 72% | **84%** | 82% | 82% | 82% | **84%@K1.00** | Ties best fixed K1 and learns not to spend compute |
| Cube double | 66% | 70% | 68% | 68% | 68% | **72%@K1.10** | Beats all fixed depths with selective refinement |
| Cube triple | 74% | 74% | 74% | 72% | 74% | **78%@K1.06** | Beats all fixed depths in this checkpoint family |

The continue threshold `eta` is an evaluation-time sweep parameter. It is not
shown in the main table because the primary reported quantity is the selected
success/mean-depth operating point. The reported operating points use
`eta=0.45` for Reacher, `eta=0.70` for Cube single, and `eta=0.50` for Cube
double and Cube triple.

Depth allocation at each best learned dynamic point:

| Dataset | K1 | K2 | K3 | K4 | Refined beyond K1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 92.188% | 7.649% | 0.156% | 0.007% | 7.812% |
| Cube single | 99.987% | 0.013% | 0% | 0% | 0.013% |
| Cube double | 91.197% | 7.949% | 0.784% | 0.070% | 8.803% |
| Cube triple | 95.205% | 3.244% | 1.502% | 0.049% | 4.795% |

Rollout-step traces:

| Dataset | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 3.95% | 10.23% | 7.02% | 9.30% | 8.56% |
| Cube single | 0.00% | 0.00% | 0.00% | 0.00% | 0.06% |
| Cube double | 9.65% | 8.16% | 10.83% | 7.46% | 7.89% |
| Cube triple | 8.64% | 3.63% | 5.72% | 4.61% | 1.28% |

Local figures:

- `figures/depth_allocation_rel00005.png`
- `figures/depth_by_rollout_step_rel00005.png`
- `figures/depth_by_rollout_step_stacked_rel00005.png`

Remote result directories:

- `/vepfs/zijian/lewm_data/ttjepa_reacher_joint_marginal_rel00005_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_joint_marginal_rel00005_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_double_joint_marginal_rel00005_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e`

Remote train/eval scripts confirming the shared label setting:

- `/vepfs/zijian/TTJepa/logs/ttjepa_joint_rel00005_cross_dataset_20260621_010900.sh`
- `/vepfs/zijian/TTJepa/logs/ttjepa_joint_rel00005_cube_pair_20260621_011000.sh`

### Superseded Learned-Head Sweep: Raw Depth Labels

Before the current \(\tau_{\mathrm{rel}}=0.0005\) joint marginal-improvement
family, we evaluated `ttjepa_*_dynamic_oracle_k4_10e` checkpoints whose
continue targets were built from raw target-MSE depth labels. Those results are
useful historical evidence, but they are not the current main table. The last
column records the current main result for the same dataset.

| Dataset | LeWM baseline | Fixed K1 / K4 in old sweep | Best old learned-head result | Current main \(\tau_{\mathrm{rel}}=0.0005\) result | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Reacher | 80% | 88%@K1 / 86%@K4 | 90%@K1.003 (`t=0.5`) | **86%@K1.08** | Old raw-depth-label run was strong; current main table is the matched joint marginal-improvement family |
| Cube single | 72% | 78.0%@K1 / 77.3%@K4 | 81.33%@K1.13 (`t=0.5`, 3 seeds) | **84%@K1.00** | Current main run is stronger and cleaner: it ties best fixed K1 while spending almost no extra compute |
| Cube double | 66% | 72%@K1 / 70%@K4 | 72%@K1.003--1.020 (`t=0.35/0.5/0.7`) | **72%@K1.10** | Both families show selective refinement can match or exceed fixed-depth baselines |
| Cube triple | 74% | 70%@K1 / 78%@K4 | 74%@K1.40 (`t=0.001` diagnostic) | **78%@K1.06** | Current main run fixes the old under-allocation failure on cube-triple |

## Main Fixed-Depth And Raw-MSE Results

The raw-MSE analysis uses the recurrent checkpoints associated with the
K-refinement rows under `analysis/k_refinement_all_20260620_024634`.

| Dataset / run | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher seed42 | 80% | 88% | n/a | n/a | 86% | Current checkpoint does not need large K |
| Cube single seed42 | 72% | 80% | n/a | n/a | 78% | K4 slightly lower than K1 |
| Cube single seed43 | 72% | 88% | n/a | n/a | 90% | K4 improves by 2 points |
| Cube single seed44 | 72% | 66% | n/a | n/a | 64% | K4 slightly lower than K1 |
| Cube single 3-seed avg | 72% | 78% | n/a | n/a | 77.3% | Seed-level gain but not stable mean gain |
| Cube single original rerun `20260621_refixed_k1234` | 72% | 80% | 76% | 78% | 78% | Same original checkpoint; K1 best |
| Cube double original rerun `20260621_refixed_k1234` | 66% | 72% | 70% | 68% | 70% | Extra depth does not help |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | Cleanest fixed-depth K gain |
| Cube triple whitened checkpoint | 74% | 72% | 72% | 76% | 76% | Exploratory checkpoint; depth still helps |

Important distinction: `LeWM baseline` is the original non-recurrent baseline.
`Fixed K1` is TTJepa's recurrent transition predictor stopped after one
refinement step. They are not the same model.

## Raw Target-Latent MSE Diagnostic

This is a post-hoc diagnostic, not a deployable policy. It uses true target
latents after evaluation to ask whether raw latent error contains information
about which depth would have been useful.

| Dataset | Fixed K1 | Fixed K4 | MSE diagnostic | Outcome upper bound | K1 fail / K4 success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 88%@K1.00 | 86%@K4.00 | 88%@K1.06 to K2.32 | 92%@K1.12 | 2 / 50 |
| Cube single | 78%@K1.00 | 77.3%@K4.00 | 77.3%@K2.72 to K2.96 | 80.7%@K1.08 | 4 / 150 |
| Cube double | 72%@K1.00 | 70%@K4.00 | 72%@K1.00 to K2.62 | 72%@K1.00 | 0 / 50 |
| Cube triple | 70%@K1.00 | 78%@K4.00 | 76%@K2.32 | 82%@K1.36 | 6 / 50 |

Cube-triple threshold sweep:

| Rule | Success | Mean K | Selected episodes | Notes |
| --- | ---: | ---: | ---: | --- |
| Fixed K1 | 70% | 1.00 | 0 / 50 | Shallow baseline |
| Raw latent MSE, tolerance 0 | 76% | 2.32 | 22 / 50 | Recovers part of the K4 gain |
| Raw latent MSE, tolerance 0.001 | 74% | 1.96 | 16 / 50 | Less compute, weaker success |
| Raw latent MSE, tolerance 0.003 | 72% | 1.54 | 9 / 50 | Too conservative |
| Fixed K4 | 78% | 4.00 | 50 / 50 | Stronger but expensive |

Cube-triple K1/K4 episode categories:

| Category | Count |
| --- | ---: |
| K1 fails, K4 succeeds | 6 |
| K1 succeeds, K4 fails | 2 |
| Both succeed | 33 |
| Both fail | 9 |

Local artifacts:

- `analysis/k_refinement_all_20260620_024634/raw_mse_k_gating_sweep.csv`
- `analysis/k_refinement_all_20260620_024634/k_gating_pareto_all.png`
- `analysis/paper1_figures/png_direct/raw_mse_tolerance_pareto.png`
- `analysis/paper1_figures/png_direct/k1_k4_outcome_split.png`
- `analysis/paper1_figures/png_direct/raw_mse_precision_recall_failure.png`

## Mechanistic Analysis: Spectrum And State Probes

Completed analysis pass: `analysis/k_smoothing_20260622`.

Result: there is no strong global latent-collapse signature from K1 to K4.

| Dataset | Spectrum summary | Probe summary |
| --- | --- | --- |
| Reacher | K4/K1 entropy-rank ratio `1.000`, variance ratio `1.000` | qpos R2 `-0.131 -> -0.131`; observation probe is poor |
| Cube single | entropy-rank ratio `1.000`, variance ratio `1.003` | block position R2 `0.991 -> 0.991` |
| Cube double | entropy-rank ratio `1.000`, variance ratio `1.000` | block position R2 `0.946 -> 0.946`; pairwise distance `0.900 -> 0.899` |
| Cube triple | entropy-rank ratio `1.000`, variance ratio `1.000` | block position R2 `0.902 -> 0.902`; pairwise distance `0.895 -> 0.894` |

Figures:

- `analysis/k_smoothing_20260622/figures/spectrum_k1_vs_k4_scatter.png`
- `analysis/k_smoothing_20260622/figures/probe_r2_k1_vs_k4_scatter.png`
- `analysis/k_smoothing_20260622/figures/category_probe_mse_k1_vs_k4_scatter.png`

Interpretation: the failure mode is likely more local than global spectrum or
simple linear state probes can see. The next decisive analysis is CEM candidate
ranking stability.

## Planner-Feature Selector Diagnostic

This is not Paper 1's method. It is a diagnostic showing that planner/result
features can contain stronger information about when K4 is useful.

| Threshold range | Success | Mean K |
| --- | ---: | ---: |
| 0.38 to 0.41 | 80% | 2.53 to 2.63 |
| 0.50 | 74% | 1.80 |
| 0.70 | 72% | 1.12 |

Interpretation: planner trajectory information can predict useful extra depth,
but this was a separate diagnostic selector rather than the learned dynamic-K
method used in the current main table.

## Learned Continue Head / Joint Marginal-Depth Variants

These runs should be kept separate by checkpoint family. The
\(\tau_{\mathrm{rel}}=0.0005\) cross-dataset family above is now the current
main learned dynamic-K method. The table below preserves the earlier
cube-triple-only sweep across different relative-improvement settings. Note the
name collision: the cube-triple-only directory called `rel0005` is not the same
artifact family as the four-dataset main sweep whose remote directories are
named `rel00005`.

| Run | Learned dynamic result | Fixed K1 sanity | Fixed K4 sanity | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `rel00005` four-dataset main family, cube-triple row | 78% at K=1.064 | 74% | 74% | Clean learned-selector gain |
| `rel0002` | 78% at K=1.035 | 78% | 72% | Avoids harmful over-refinement, but does not beat K1 |
| cube-triple-only directory `rel0005` | 80% at K=1.000 to K=1.062 | 80% | 80% | Auxiliary training-variant record; not the four-dataset main table |
| `rel0001` | 74% at K=1.47 | not sanity-checked | not sanity-checked | Weaker setting |
| `rel000` | 66% near K1 | not sanity-checked | not sanity-checked | No-margin target fails |

Preferred wording for the cube-triple-only `rel0005` directory:

> A stronger joint-depth training variant improves all depths to 80%,
> suggesting a separate training-time regularization effect; we exclude it from
> the main dynamic-compute comparison and discuss it separately.

Remote checkpoints:

- `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e/weights_epoch_10.pt`
- `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e/weights_epoch_10.pt`
- `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e/weights_epoch_10.pt`

Remote result directories:

- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e`

Important logs:

- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e_20260620_085100_fine.log`
- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e_20260620_085100_fine.log`
- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e_20260620_085100.log`

## Exploratory Halt-Label Variants

These are alternative label/score designs explored during debugging. They are
not part of Paper 1.

| Variant | Result summary | Status |
| --- | --- | --- |
| Raw learned halt thresholds | Best learned result around 74% at mean depth about 1.40 | Useful diagnosis; weaker than fixed K4 |
| Whitened latent MSE | Fixed K1/K2/K3/K4 = 72/72/76/76; learned best 74% | Whitening did not clearly fix halting |
| Probe-weighted latent MSE | Ran as an exploratory queue; keep result files under remote experiment directory | Needs exact final table copied from remote artifacts |

Known remote directories:

- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_dynamic_whitened_oracle_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_dynamic_probe_weighted_oracle_k4_10e`

Known logs:

- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_whitened_oracle_20260617_20260617_063655.log`
- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_probe_weighted_oracle_20260617_193620.log`

## Remote Layout

Remote machine:

- SSH: `ssh -p 20747 root@115.190.235.210`
- Repo: `/vepfs/zijian/TTJepa`
- Data/results root: `/vepfs/zijian/lewm_data`

Important raw recurrent checkpoint:

- `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt`

Important raw recurrent result directory:

- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_dynamic_oracle_k4_10e`

Local analysis scripts:

- `scripts/k_refinement_analysis.py`
- `scripts/k_raw_mse_sweep.py`
- `scripts/k_smoothing_analysis.py`
