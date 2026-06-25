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
| Reacher seed42 | 80% | 88% | n/a | n/a | 86% | K4 is worse than K1 in this checkpoint |
| Cube single seed42 | 72% | 80% | n/a | n/a | 78% | K4 is slightly lower than K1 |
| Cube single seed43 | 72% | 88% | n/a | n/a | 90% | K4 improves over K1 by 2 points |
| Cube single seed44 | 72% | 66% | n/a | n/a | 64% | K4 is slightly lower than K1 |
| Cube single 3-seed avg | 72% | 78% | n/a | n/a | 77.3% | Average K4 is slightly lower, but seed43 shows K4 can help |
| Cube single original rerun `20260621_refixed_k1234` | 72% | 80% | 76% | 78% | 78% | K1 is best; K3/K4 recover to 78% |
| Cube double original rerun `20260621_refixed_k1234` | 66% | 72% | 70% | 68% | 70% | Extra depth does not help in this run |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | Clearest setting where deeper K helps |

The accurate conclusion is not that deeper `K` is always better. The result is
more interesting: `K` changes planning success, but the useful depth is
dataset- and checkpoint-dependent. This is exactly why dynamic allocation is
the central question.

## Raw Latent MSE Diagnostic

The first analysis asks whether raw latent MSE can identify transitions that
benefit from deeper refinement. This is a post-hoc diagnostic: it compares
already evaluated fixed-depth outcomes and selects between shallow and deeper
refinement based on observed latent-MSE improvement.

| Dataset | Fixed K1 | Fixed K4 | Best raw-MSE dynamic K | Hindsight K1/K4 chooser | Depth-helped cases |
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

## Learned Dynamic K With Raw-MSE Supervision

The learned version uses the same raw-MSE idea as training supervision. Each
recurrent state predicts whether another refinement step should be taken. At
evaluation time, the continue head chooses the depth automatically.

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
