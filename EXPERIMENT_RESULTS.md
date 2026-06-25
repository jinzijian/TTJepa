# RefineJEPA Experiment Results

This file is the compact result ledger for the current RefineJEPA dynamic-`K`
experiments. RefineJEPA is built on the LeWM latent-planning codebase.

## LeWM Reproduction Baselines

| Task | Reproduced LeWM | Official LeWM | Notes |
| --- | ---: | ---: | --- |
| PushT 10e | 92% | 96% | close to ceiling |
| Reacher 10e | 80% | 86% | default 50/25 eval |
| Cube single 10e | 72% | 74% | OGBench cube single |
| TwoRoom 10e | 94-98% | 87% | fixed timestep eval |

Detailed environment and reproduction notes are in
[LEWM_REPRODUCTION_NOTES.md](LEWM_REPRODUCTION_NOTES.md).

## Fixed-Depth RefineJEPA

| Dataset / run | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher seed42 | 80% | 88% | n/a | n/a | 86% | K4 worse than K1 |
| Cube single seed42 | 72% | 80% | n/a | n/a | 78% | K4 slightly lower |
| Cube single seed43 | 72% | 88% | n/a | n/a | 90% | K4 helps |
| Cube single seed44 | 72% | 66% | n/a | n/a | 64% | K4 slightly lower |
| Cube single 3-seed avg | 72% | 78% | n/a | n/a | 77.3% | average K4 slightly lower |
| Cube single original rerun `20260621_refixed_k1234` | 72% | 80% | 76% | 78% | 78% | same-source K1-4 sweep |
| Cube double original rerun `20260621_refixed_k1234` | 66% | 72% | 70% | 68% | 70% | depth does not help |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | clearest depth-positive setting |

Important distinction: LeWM baseline is the original non-recurrent transition
predictor. Fixed `K1` is the recurrent RefineJEPA predictor stopped after one
refinement step.

## Post-Hoc Raw Latent MSE Selection

This analysis selects between evaluated depths using observed raw latent-MSE
improvement. It is a teacher diagnostic, not the deployed learned selector: it
uses the true next latent after evaluation to score whether deeper refinement
helped. The learned-head table below uses a trained continue head at inference,
so the two tables should not be interpreted as duplicate measurements of the
same method.

| Dataset | Fixed K1 | Fixed K4 | Best post-hoc raw-MSE selector | Hindsight K1/K4 chooser | Depth-helped cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 88%@K1.00 | 86%@K4.00 | 88%@K1.06 to K2.32 | 92%@K1.12 | 2 / 50 |
| Cube single | 78%@K1.00 | 77.3%@K4.00 | 77.3%@K2.72 to K2.96 | 80.7%@K1.08 | 4 / 150 |
| Cube double | 72%@K1.00 | 70%@K4.00 | 72%@K1.00 to K2.62 | 72%@K1.00 | 0 / 50 |
| Cube triple | 70%@K1.00 | 78%@K4.00 | 76%@K2.32 | 82%@K1.36 | 6 / 50 |

Summary:

- Raw MSE has signal on cube-triple: `70% -> 76%` at mean `K=2.32`.
- Raw MSE still misses the best fixed-depth and hindsight outcomes.
- The likely limitation is planner alignment: lower latent MSE does not always
  mean better CEM candidate ranking.

## Learned Continue Head From Raw-MSE Targets

These are the four available raw-MSE learned-halting checkpoints on the four
core datasets. Unlike the post-hoc diagnostic above, this is an inference-time
selector: the model sees its recurrent state and predicted continue probability,
not the true target latent MSE.

| Dataset | Checkpoint family | LeWM baseline | Fixed K1 | Fixed K4 | Best learned dynamic K | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Reacher | `ttjepa_reacher_dynamic_oracle_k4_10e` | 80% | 88%@K1.00 | 86%@K4.00 | 90%@K1.003 (`t=0.5`) | small learned dynamic gain |
| Cube single | `ttjepa_cube_dynamic_oracle_k4_10e` | 72% | 78.00%@K1.00 | 77.33%@K4.00 | 81.33%@K1.13 (`t=0.5`, seeds 42/43/44) | strongest learned dynamic result |
| Cube double | `ttjepa_cube_double_dynamic_oracle_k4_10e` | 66% | 72%@K1.00 | 70%@K4.00 | 72%@K1.003-1.020 (`t=0.35/0.5/0.7`) | matches K1; no depth gain |
| Cube triple | `ttjepa_cube_triple_dynamic_oracle_k4_10e` | 74% | 70%@K1.00 | 78%@K4.00 | 74%@K1.40 (`t=0.001` diagnostic) | under-allocates depth; misses fixed K4 |

Cube-single per-seed details from the depth-logged sweep:

| Mode | Seed 42 | Seed 43 | Seed 44 | Mean |
| --- | ---: | ---: | ---: | ---: |
| fixed K1 | 80%@K1.00 | 88%@K1.00 | 66%@K1.00 | 78.00%@K1.00 |
| fixed K4 | 78%@K4.00 | 90%@K4.00 | 64%@K4.00 | 77.33%@K4.00 |
| learned `t=0.5` | 86%@K1.10 | 92%@K1.11 | 66%@K1.16 | 81.33%@K1.13 |

Cube-triple raw learned halt diagnosis:

| Mode | Success | Mean K |
| --- | ---: | ---: |
| fixed K1 | 70% | 1.00 |
| fixed K2 | 76% | 2.00 |
| fixed K3 | 76% | 3.00 |
| fixed K4 | 78% | 4.00 |
| learned `t=0.35` | 72% | 1.0036 |
| learned `t=0.5` | 70% | 1.0010 |
| learned `t=0.7` | 72% | 1.0000 |
| learned `t=0.001`, min-depth 1 | 74% | 1.4007 |

Separate cube-triple joint-depth variants:

| Run | Learned dynamic result | Fixed K1 sanity | Fixed K4 sanity | Notes |
| --- | ---: | ---: | ---: | --- |
| `rel00005` | 78%@K=1.064 | 74% | 74% | clean dynamic-compute result inside this variant |
| `rel0002` | 78%@K=1.035 | 78% | 72% | avoids harmful over-refinement |
| `rel0005` | 80%@K=1.000 to K=1.062 | 80% | 80% | likely training-time regularization |
| `rel0001` | 74%@K=1.47 | n/a | n/a | weaker |
| `rel000` | 66% near K1 | n/a | n/a | no-margin target fails |

## Current Paper Position

The main paper should focus on raw latent MSE dynamic `K`:

1. Show fixed `K` matters but no fixed depth dominates.
2. Show raw latent MSE is a useful first signal.
3. Show learned dynamic `K` can improve success with near-K1 compute.
4. Analyze when MSE helps, when it fails, and why planner benefit is the more
   direct target for future work.
