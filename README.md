# RefineJEPA
### Dynamic test-time transition refinement for LeWM-style latent planning

RefineJEPA is a research fork of
[LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm). The project studies
one question:

> Can a latent world-model planner decide how much computation each imagined
> transition deserves, instead of using the same transition-model depth
> everywhere?

The motivating example is simple. When a robot moves its hand through free
space toward an object, the dynamics are usually easy and one transition-model
pass may be enough. When the robot makes contact, lifts the object, or resolves
which object matters for the goal, the imagined transition can need more
refinement before the planner should trust it. RefineJEPA focuses on this
transition-level compute axis.

LeWM already performs latent model-predictive planning: it encodes observations
and goals into a latent space, rolls out candidate action sequences, and uses
CEM/MPC to choose the action sequence with the best goal-matching cost. RefineJEPA
keeps this planning setup fixed and changes the transition predictor into a
weight-tied recurrent predictor with a variable refinement depth `K`.

## Status

- Base code: LeWM-style JEPA latent planner.
- Method code: active development branch `codex/recurrent-lewm`.
- Paper direction: dynamic test-time compute through transition refinement depth
  `K`.
- Main v0 signal: raw latent MSE improvement, used both as a diagnostic and as
  supervision for a learned continue head.

This README records the current research story and experiment state. The
original LeWM reproduction notes are kept in
[LEWM_REPRODUCTION_NOTES.md](LEWM_REPRODUCTION_NOTES.md).

## Method

RefineJEPA studies the transition predictor inside latent planning:

```text
z_hat[t+1]^(1) -> z_hat[t+1]^(2) -> ... -> z_hat[t+1]^(K)
```

There are two evaluation modes:

1. Fixed `K`: every imagined transition uses the same depth, such as `K=1`,
   `K=2`, or `K=4`.
2. Dynamic `K`: the predictor decides whether another recurrent refinement step
   is worth paying for.

For the current paper draft, the dynamic rule is intentionally simple. During
training, raw latent MSE improvement defines the stop/continue target: continue
when another recurrent step meaningfully improves the next-latent prediction.
At test time, a learned continue head predicts whether to keep refining, so the
model does not need access to the true next observation.

The goal is not to introduce a new action space, a new planner, or a new
dataset-specific controller. The goal is to understand whether `K` is a useful
test-time compute knob for JEPA latent planners.

## Main Experiment Record

The table below separates the original LeWM baseline from recurrent fixed-depth
RefineJEPA checkpoints. `LeWM baseline` and `Fixed K1` are not the same model:
LeWM uses the original non-recurrent transition predictor, while fixed `K1`
uses the recurrent RefineJEPA predictor stopped after its first refinement step.

| Dataset / run | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Observation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher seed42 | 80% | 88% | 86% | 86% | 86% | Extra depth is lower than K1 in this checkpoint |
| Cube single seed42 | 72% | 80% | 78% | 78% | 78% | Extra depth is slightly lower than K1 |
| Cube single seed43 | 72% | 88% | 90% | 90% | 90% | Extra depth improves over K1 by 2 points |
| Cube single seed44 | 72% | 66% | 66% | 62% | 64% | K2 matches K1; deeper K is lower |
| Cube single 3-seed avg | 72% | 78% | 78% | 76.7% | 77.3% | Average K2 matches K1; K3/K4 are slightly lower, but seed43 shows extra depth can help |
| Cube single original rerun `20260621_refixed_k1234` | 72% | 80% | 76% | 78% | 78% | K1 is best; K3/K4 recover to 78% |
| Cube double original rerun `20260621_refixed_k1234` | 66% | 72% | 70% | 68% | 70% | Extra depth does not help in this run |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | Clearest setting where deeper K helps |

The accurate conclusion is not that deeper `K` is always better. The result is
more interesting: `K` changes planning success, but the useful depth is
dataset- and checkpoint-dependent. This is exactly why dynamic allocation is
the central question.

![Fixed K success bars](analysis/readme_figures/fixed_k_success_bars.png)

The fixed-depth picture is already enough to motivate dynamic test-time
compute. Cube Triple is the cleanest positive setting for deeper transition
refinement (`70% -> 78%` from `K1` to `K4`). Reacher and Cube Double move in the
opposite direction or saturate near `K1`, and Cube Single depends on
checkpoint/seed. Thus the paper should not claim that deeper refinement is
universally better; the correct claim is that `K` is a real compute axis whose
utility must be allocated conditionally.

## Post-Hoc Raw Latent MSE Diagnostic

The first analysis asks whether raw latent MSE can identify transitions that
benefit from deeper refinement. This is not a deployable learned policy. It is
a post-hoc teacher diagnostic: after evaluating fixed depths, it uses the true
next latent to measure whether deeper refinement reduced prediction error, then
asks how well that signal would select between shallow and deeper outcomes.

This table should not be read as the same experiment as the learned-head table
below. It uses observed target-latent MSE after the fact; the learned-head table
uses a trained continue head at inference time and does not see the target
latent.

| Dataset | Fixed K1 | Fixed K4 | Best post-hoc raw-MSE selector | Hindsight K1/K4 chooser | Depth-helped cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 88%@K1.00 | 86%@K4.00 | 88%@K1.06 to K2.32 | 92%@K1.12 | 2 / 50 |
| Cube single | 78%@K1.00 | 77.3%@K4.00 | 77.3%@K2.72 to K2.96 | 80.7%@K1.08 | 4 / 150 |
| Cube double | 72%@K1.00 | 70%@K4.00 | 72%@K1.00 to K2.62 | 72%@K1.00 | 0 / 50 |
| Cube triple | 70%@K1.00 | 78%@K4.00 | 76%@K2.32 | 82%@K1.36 | 6 / 50 |

Takeaways:

- Raw latent MSE is a reasonable v0 signal, not a dead end. On cube-triple it
  recovers a real part of the fixed-depth gain: `70% -> 76%` at mean `K=2.32`.
- It still does not reach fixed `K4` (`78%`) or the hindsight chooser
  (`82%@K1.36`), so raw latent MSE is useful but incomplete.
- The weakness is planner alignment. Raw latent MSE measures representation
  error, while CEM cares about whether the predicted latent changes the selected
  action sequence.

### Raw-MSE Allocation Confusion on Cube Triple

The cube-triple outcome split makes the raw-MSE failure mode concrete. With
the tolerance-0 post-hoc selector, raw MSE sends an episode to `K4` whenever
the `K4` prediction has lower target-latent MSE than `K1`.

![Cube Triple raw-MSE allocation confusion](analysis/readme_figures/cube_triple_raw_mse_allocation_confusion.png)

Readout:

- Raw MSE selects `K4` for `3/6` beneficial cases, so it contains real signal.
- It selects `K4` for `17/33` redundant cases, spending most of its extra
  compute where the planning outcome was already successful at `K1`.
- It avoids both harmful cases in this run, but still misses half of the cases
  where deeper refinement changes failure into success.

This is the main diagnostic conclusion: raw latent MSE is not weak, but it is
not the same as planning utility. The selector improves cube-triple from
`70%@K1` to `76%@K2.32`, yet the outcome-based hindsight chooser reaches
`82%@K1.36` because it continues on fewer but more valuable cases.

## Learned Dynamic K With Raw-MSE Supervision

The learned version uses the same raw-MSE idea as training supervision. Each
recurrent state predicts whether another refinement step should be taken. At
evaluation time, the continue head chooses the depth automatically.

This is the deployable version of the idea, but it is a different experiment
from the post-hoc diagnostic above. It uses raw-MSE-derived labels during
training, then relies on the model's predicted continue probability at test
time. Because the selector is learned and the checkpoints differ, the numbers
are not expected to exactly match the post-hoc teacher selector.

The four-dataset learned-head sweep is below. These rows come from the
`ttjepa_*_dynamic_oracle_k4_10e` checkpoints, where the continue target is built
from raw latent-MSE depth labels.

| Dataset | LeWM baseline | Fixed K1 | Fixed K4 | Best learned dynamic K | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Reacher | 80% | 88%@K1.00 | 86%@K4.00 | 90%@K1.003 (`t=0.5`) | Dynamic head slightly beats both fixed depths while using near-K1 compute |
| Cube single | 72% | 78.00%@K1.00 | 77.33%@K4.00 | 81.33%@K1.13 (`t=0.5`, 3 seeds) | Strongest clean learned dynamic-K result |
| Cube double | 66% | 72%@K1.00 | 70%@K4.00 | 72%@K1.003-1.020 (`t=0.35/0.5/0.7`) | Matches K1; no evidence that extra depth helps this checkpoint |
| Cube triple | 74% | 70%@K1.00 | 78%@K4.00 | 74%@K1.40 (`t=0.001` diagnostic) | Raw learned halt under-allocates depth and misses the K4 gain |

The main readout is mixed but useful. Cube single is the clearest positive
learned dynamic result: `81.33%` at mean `K=1.13`, compared with `78.00%` fixed
K1 and `77.33%` fixed K4. Reacher also has a small positive learned result.
Cube double mostly says "do not spend compute." Cube triple exposes the main
failure mode: fixed K4 is useful, but the raw-MSE learned halt head mostly stops
near K1, so it loses the deeper-refinement gain.

There is also a later cube-triple joint-depth variant (`rel00005`, `rel0002`,
`rel0005`, etc.) where the best clean dynamic result reaches `78%@K=1.064`, and
one stronger training variant reaches `80%` at all depths. That batch is useful
for studying training-time regularization, but it is not the four-dataset
raw-MSE learned-head sweep above.

## Analysis Direction

The current paper story should stay focused:

1. Fixed-depth analysis shows that no single `K` is uniformly best.
2. Raw latent MSE is a useful first signal for dynamic `K`, especially on
   cube-triple.
3. A learned continue head can turn this signal into a deployable dynamic
   test-time compute mechanism.
4. Failure analysis explains why latent MSE is incomplete: it is not identical
   to planner benefit.

The main open analysis is to connect deeper refinement with planning decisions:

- Does `K` reduce latent prediction error?
- Does it improve CEM candidate ranking?
- Does it preserve or smooth task-relevant contact details?
- When does extra `K` help, do nothing, or hurt?

## Appendix: Additional Experiment Ledger

This appendix records completed runs that are useful for follow-up analysis but
should not be merged into the main raw-MSE learned-head comparison above. The
main paper story should remain the four-dataset raw-MSE dynamic-`K` result. The
runs below either use planner-aware selection signals or different training
losses, so they answer related but distinct questions.

### Planner-Aware Selectors on Cube Triple

These runs were evaluated on the cube-triple recurrent checkpoint where fixed
`K1` is `70%` and fixed `K4` is `78%`. They test selection rules closer to
planner behavior than raw next-latent MSE.

| Selector family | Best / representative result | Interpretation |
| --- | ---: | --- |
| Learned planner selector | `80%@K2.53-2.63` (`t=0.38-0.41`) | Can exceed fixed K4, but uses substantially more depth than raw-MSE learned head |
| Benefit-cost rule | `78%@K2.85` (`beta=0`) | Recovers fixed K4 success with lower mean depth than uniform K4 |
| Cost-change rule | `78%@K3.38` (`alpha=0.25`) | Also recovers fixed K4 but is compute-heavy |
| Rank-stability rule | `74%@K2.04-2.48` | Does not recover fixed K4 in this run |

These are evidence that planner-aligned selection is promising. They are not
part of the current main table because the paper's first version focuses on the
simpler raw-MSE signal and its failure modes.

![Cube Triple planner alignment trace](analysis/readme_figures/cube_triple_planner_alignment_trace.png)

The planner trace supports the same conclusion from another angle. The trace
records top-30 CEM elite-set overlap between adjacent refinement depths and the
relative improvement in top-30 candidate cost from evaluating the next depth.
Many rollout calls already have very high elite overlap, yet their next-depth
cost benefit can still be positive, negative, or nearly zero. A simple
rank-stability rule therefore reaches only `74%@K2.04`, while a learned
planner-feature selector reaches `80%@K2.59`.

This is useful evidence for the paper's analysis section, but it is not yet the
complete CEM-ranking figure. The current trace logs elite-overlap and top-k cost
benefit; it does **not** yet log full candidate-rank Kendall tau or selected
action changes. The stronger paper figure should add:

- Kendall tau between the full `K1` and `K4` candidate-cost rankings.
- Top-elite overlap by outcome category: beneficial, harmful, redundant, and
  insufficient.
- Whether the best selected candidate/action changes between `K1` and `K4`.
- Whether beneficial cases correspond to ranking corrections and harmful cases
  correspond to unfavorable ranking shifts.

The target conclusion is sharper than "deeper K lowers MSE": the planner cares
about candidate ordering and action selection. Raw latent MSE captures some
prediction-improvement pressure, but planning utility depends on whether the
refined latent rollout changes the CEM decision in the right direction.

### Cube Triple Joint-Depth Variants

These variants change the training objective, so they are better treated as
evidence for a separate training-time regularization story rather than the main
dynamic test-time compute result.

| Run | Fixed K1 | Fixed K4 | Best learned dynamic result | Readout |
| --- | ---: | ---: | ---: | --- |
| `rel00005` | 74% | 74% | `78%@K1.064` | Clean dynamic-compute gain inside this variant |
| `rel0002` | 78% | 72% | `78%@K1.002-1.035` | Dynamic selection avoids harmful over-refinement |
| `rel0005` | 80% | 80% | `80%@K1.000-1.062` | Stronger training effect; likely not a pure selector result |
| `rel0001` | n/a | n/a | `74%@K1.47` | Weaker |
| `rel000` | n/a | n/a | `66%` near K1 | No-margin target fails |

The important note is that `rel0005` reaching `80%` at both fixed depths should
be described as a training-time regularization effect, not as the headline
dynamic-`K` result.

### Cube Double Depth-Loss Variants

These runs were started because cube double was the weakest main-table case for
dynamic `K`. They show that changing the training objective can move the fixed
depth behavior, but they do not give a cleaner raw-MSE dynamic-compute result
than the main checkpoint.

| Run | Fixed-depth behavior | Best learned dynamic result | Readout |
| --- | --- | ---: | --- |
| `finalonly_rel00005` | K1 46%, K2 60%, K3 62%, K4 70% | `76%@K3.53` | Improves success but spends near-deep compute; not a clean efficiency result |
| `joint_marginal_rel00005` | K1 70%, K4 68% | `72%@K1.10-1.60` | Slight dynamic gain, still near the original cube-double ceiling |
| `lightinter01_rel00005` | K1 76%, K4 68% | `72%@K1.15-1.57` | Shallow fixed K1 is strongest; dynamic selector does not improve it |

For Paper 1, cube double should remain a negative or saturation case: the
original raw-MSE learned selector correctly avoids spending much extra compute,
but it does not discover a hidden deep-refinement gain.

### Machine and Git Sync Notes

The GitHub `main` branch contains this RefineJEPA README. The experiment
machines may still show the original LeWM README when checked out on the
development branch:

```text
codex/recurrent-lewm
```

On the new 8xA800 machine, `/vepfs/zijian/TTJepa` also contains local,
uncommitted development artifacts:

```text
M jepa.py
?? analysis/
?? scripts/k_refinement_analysis.py
?? scripts/k_smoothing_analysis.py
?? scripts/train_planner_selector.py
```

Those files contain planner-selector and analysis work that has not been
merged into `main`. Treat GitHub `main` as the current public documentation
state, and treat `/vepfs/zijian/TTJepa` as the active experiment workspace.

## Files

- [LEWM_REPRODUCTION_NOTES.md](LEWM_REPRODUCTION_NOTES.md): LeWM reproduction
  environment, dataset paths, and baseline results.
- [README.zh-CN.md](README.zh-CN.md): Chinese project summary.
- [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md): compact result ledger for the
  current RefineJEPA experiments.
- Method implementation notes and active code live on the development branch
  `codex/recurrent-lewm`.

## Codebase Lineage

This repository is based on LeWM. Please cite the original LeWM work when using
the base world-model code:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
