# Hard Manipulation Benchmark Scouting

## Motivation

TwoRoom and PushT are close to ceiling in the current reproduction. PushT fixed
depth K=4 reached 94% against a reproduced LeWM baseline of 92% and the paper
number of 96%, so one or two episode flips can dominate the comparison. Cube is
more promising: TTJepa recurrent K=1/2/4 all reached 80%, above both our
reproduced Cube baseline of 72% and the paper number of 74%.

The later Cube dynamic-halting run made this direction stronger. A learned
oracle-depth halting checkpoint reached 86% on the first seed at threshold 0.5.
A three-seed depth-logged sweep then showed threshold 0.5 averaging 81.33%
success with mean predictor depth 1.13, compared with fixed K=1 at 78.00% and
fixed K=4 at 77.33%. This is the clearest current evidence that the method is
not merely using more compute: it can use compute close to K=1 while matching or
beating deeper fixed-depth planning.

The next benchmark should be harder than PushT, contact-rich, and still cheap
enough to wire into the existing LeWM training and CEM evaluation loop.

## Ranking

| Priority | Candidate | Why | Main risk |
| --- | --- | --- | --- |
| 1 | OGBench visual cube-double / cube-triple | Same OGB manipulation family as current Cube, more objects and object identity, stable-worldmodel already exposes `swm/OGBCube-v0` with `env_type=double/triple` | Official OGB files are NPZ, so they need conversion to LeWM HDF5 |
| 2 | OGBench visual scene | Adds drawer/window/buttons and mixed contact modes | Eval goal-setting needs more wiring than cube because Scene has multiple target APIs |
| 3 | FetchPickAndPlace / FetchSlide | Already registered by stable-worldmodel, clean grasping sanity task | Need demos or collection pipeline |
| 4 | RoboCasa single-stage PnP / drawer tasks | Richer kitchen manipulation and stable-worldmodel wrapper exists | Heavier environment/data footprint |
| 5 | robomimic Lift / Can / Square / Transport | Classic manipulation datasets in HDF5 | Requires robosuite eval adapter and goal-setting work |
| 6 | LIBERO / CALVIN / ManiSkill / FurnitureBench | Strong long-horizon benchmark story | Too much systems work for the immediate TTJepa claim |

## First Targets

Start with `visual-cube-double-play-v0`, then extend to
`visual-cube-triple-play-v0`.

Reasoning after the Cube dynamic result:

- The method's strongest signal is already on OGBench Cube, not PushT.
- `cube-double` preserves the same simulator family, action space, and contact
  mechanism while adding object identity and harder goal binding.
- `cube-triple` is the next lowest-friction step after double: same converter,
  same `swm/OGBCube-v0` wrapper, and only one more `set_target_pos` callable.
  It is harder than double but still much cheaper to wire than Scene, RoboCasa,
  robomimic, or LIBERO-style tasks.

Known remote facts:

- Raw val file downloaded:
  `/home/robotuser/zijian/lewm_data/ogbench_scout/raw/visual-cube-double-play-v0-val.npz`
- Schema:
  `observations (N, 64, 64, 3) uint8`, `actions (N, 5) float32`,
  `terminals`, `qpos (N, 28)`, `qvel (N, 26)`.
- For cube-double, cube poses occupy the last 14 `qpos` entries:
  cube 0 starts at `qpos[:, 14:21]`, cube 1 starts at `qpos[:, 21:28]`.
- For cube-triple, use the same rule with `--num-cubes 3`: cube poses occupy
  the last 21 `qpos` entries, and the converter writes
  `goal_privileged_block_{0,1,2}_pos/quat`.

The converter writes:

- `pixels <- observations`
- `action <- actions`
- `observation <- zeros((N, 19 + 9 * num_cubes))` to match the OGBCube
  step-time info shape: single 28, double 37, triple 46
- `qpos`, `qvel`, `ep_idx`, `step_idx`, `ep_len`, `ep_offset`
- `goal_privileged_block_{i}_pos`
- `goal_privileged_block_{i}_quat`

## Cube-Double Commands

Download the train split:

```bash
mkdir -p /home/robotuser/zijian/lewm_data/ogbench_scout/raw
cd /home/robotuser/zijian/lewm_data/ogbench_scout/raw
curl -L --fail -O https://rail.eecs.berkeley.edu/datasets/ogbench/visual-cube-double-play-v0.npz
```

Convert to LeWM HDF5:

```bash
cd /home/robotuser/zijian/TTJepa
source /home/robotuser/zijian/le-wm/.venv/bin/activate
export STABLEWM_HOME=/home/robotuser/zijian/lewm_data
/home/robotuser/.local/bin/uv run --active python scripts/convert_ogbench_npz_to_hdf5.py \
  /home/robotuser/zijian/lewm_data/ogbench_scout/raw/visual-cube-double-play-v0.npz \
  /home/robotuser/zijian/lewm_data/datasets/ogbench/visual_cube_double_play.h5 \
  --num-cubes 2
```

Smoke train:

```bash
/home/robotuser/.local/bin/uv run --active python train.py --config-name lewm_recurrent \
  data=ogb_cube_double_visual \
  trainer.max_epochs=1 \
  loader.batch_size=8 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  output_model_name=ttjepa_cube_double_smoke \
  subdir=ttjepa_cube_double_smoke \
  recurrent.max_depth=4
```

Smoke eval:

```bash
/home/robotuser/.local/bin/uv run --active python eval.py --config-name cube_double \
  policy=ttjepa_cube_double_smoke/weights_epoch_1.pt \
  eval.num_eval=5 \
  planner.predictor_depth=4 \
  output.filename=ttjepa_cube_double_smoke_results.txt
```

## Cube-Triple Commands

Download the train split:

```bash
mkdir -p /home/robotuser/zijian/lewm_data/ogbench_scout/raw
cd /home/robotuser/zijian/lewm_data/ogbench_scout/raw
curl -L --fail -O https://rail.eecs.berkeley.edu/datasets/ogbench/visual-cube-triple-play-v0.npz
```

Convert to LeWM HDF5:

```bash
cd /home/robotuser/zijian/TTJepa
source /home/robotuser/zijian/le-wm/.venv/bin/activate
export STABLEWM_HOME=/home/robotuser/zijian/lewm_data
/home/robotuser/.local/bin/uv run --active python scripts/convert_ogbench_npz_to_hdf5.py \
  /home/robotuser/zijian/lewm_data/ogbench_scout/raw/visual-cube-triple-play-v0.npz \
  /home/robotuser/zijian/lewm_data/datasets/ogbench/visual_cube_triple_play.h5 \
  --num-cubes 3
```

Formal baseline and dynamic runs should mirror cube-double:

```bash
# LeWM baseline
/home/robotuser/.local/bin/uv run --active python train.py --config-name lewm \
  data=ogb_cube_triple_visual \
  trainer.max_epochs=10 \
  output_model_name=lewm_cube_triple_baseline_10e \
  subdir=lewm_cube_triple_baseline_10e

/home/robotuser/.local/bin/uv run --active python eval.py --config-name cube_triple \
  policy=lewm_cube_triple_baseline_10e/weights_epoch_10.pt \
  output.filename=lewm_cube_triple_baseline_10e_results.txt

# TTJepa dynamic-oracle
/home/robotuser/.local/bin/uv run --active python train.py --config-name lewm_recurrent_dynamic \
  data=ogb_cube_triple_visual \
  trainer.max_epochs=10 \
  recurrent.max_depth=4 \
  output_model_name=ttjepa_cube_triple_dynamic_oracle_k4_10e \
  subdir=ttjepa_cube_triple_dynamic_oracle_k4_10e

/home/robotuser/.local/bin/uv run --active python eval.py --config-name cube_triple \
  policy=ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt \
  planner.predictor_depth=1 \
  planner.return_depth_stats=true \
  output.filename=ttjepa_cube_triple_dynamic_fixed_k1_results.txt

/home/robotuser/.local/bin/uv run --active python eval.py --config-name cube_triple \
  policy=ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt \
  planner.predictor_depth=4 \
  planner.return_depth_stats=true \
  output.filename=ttjepa_cube_triple_dynamic_fixed_k4_results.txt

/home/robotuser/.local/bin/uv run --active python eval.py --config-name cube_triple \
  policy=ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt \
  planner.halt_mode=learned \
  planner.halt_threshold=0.5 \
  planner.min_depth=1 \
  planner.return_depth_stats=true \
  output.filename=ttjepa_cube_triple_dynamic_learned_t0_5_results.txt
```

## Cube-Triple Results and Halt Diagnosis

The first full cube-triple run finished successfully, but the learned halt
policy did not preserve the fixed-depth gain.

Result directories:

```text
/home/robotuser/zijian/lewm_data/lewm_cube_triple_baseline_10e
/home/robotuser/zijian/lewm_data/ttjepa_cube_triple_dynamic_oracle_k4_10e
```

Checkpoints:

```text
/home/robotuser/zijian/lewm_data/checkpoints/lewm_cube_triple_baseline_10e/weights_epoch_10.pt
/home/robotuser/zijian/lewm_data/checkpoints/ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt
```

Main results:

| Mode | Success rate | Mean predictor depth |
|---|---:|---:|
| LeWM baseline | 74% | n/a |
| TTJepa fixed K=1 | 70% | 1.00 |
| TTJepa fixed K=4 | 78% | 4.00 |
| TTJepa learned threshold 0.35 | 72% | 1.0036 |
| TTJepa learned threshold 0.5 | 70% | 1.0010 |
| TTJepa learned threshold 0.7 | 72% | 1.0000 |

Short read:

- Cube-triple is harder than Cube and cube-double, but still gives a positive
  recurrent-depth signal: fixed K=4 reaches `78%`, beating the `74%` LeWM
  baseline.
- Learned halting fails because it almost never uses extra depth. It behaves
  like K=1 compute and therefore lands around `70%-72%`.
- This is different from the Cube result, where learned threshold `0.5`
  reached `81.33%` mean success over three seeds with mean depth `1.13`, and
  a best single run of `86%`.

The halt diagnosis sweep added fixed K=2/K=3 and lower halt thresholds:

```text
/home/robotuser/zijian/TTJepa/logs/ttjepa_cube_triple_halt_diagnosis_20260617_045824.log
```

| Diagnostic mode | Success rate | Mean predictor depth |
|---|---:|---:|
| fixed K=2 | 76% | 2.00 |
| fixed K=3 | 76% | 3.00 |
| learned threshold 0.001, min_depth=1 | 74% | 1.4007 |
| learned threshold 0.003, min_depth=1 | 72% | 1.2828 |
| learned threshold 0.005, min_depth=1 | 70% | 1.2378 |
| learned threshold 0.01, min_depth=1 | 74% | 1.1779 |
| learned threshold 0.35, min_depth=2 | 76% | 2.0000 |
| learned threshold 0.35, min_depth=3 | 76% | 3.0000 |

Diagnosis:

- The recurrent predictor has useful capacity: K=2/K=3/K=4 all beat K=1.
- The learned halt target is poorly calibrated. With raw latent MSE,
  `continue_target_rate` is only about `0.7%-1.0%`, `halt_depth_mean` is about
  `1.03`, and `continue_prob_mean` tracks the same early-stop prior.
- The most likely reason is latent anisotropy / partial JEPA collapse: raw
  Euclidean latent MSE can say depth 1 is close enough even when low-variance
  task-critical factors still need deeper rollout.

The whitened follow-up and the stronger probe-weighted follow-up are now complete. Their final metrics are below, and the old live-queue state should be treated as superseded.

## Cube-Triple Alternative Halt Oracles

The whitened and probe-weighted follow-ups are complete. They test whether the
learned halt failure was caused by the raw latent-MSE oracle rather than by a
lack of recurrent predictor capacity.

Whitened oracle paths:

```text
Config: config/train/lewm_recurrent_dynamic_whitened.yaml
Log: /home/robotuser/zijian/TTJepa/logs/ttjepa_cube_triple_whitened_oracle_20260617_20260617_063655.log
Checkpoint: /home/robotuser/zijian/lewm_data/checkpoints/ttjepa_cube_triple_dynamic_whitened_oracle_k4_10e/weights_epoch_10.pt
Results: /home/robotuser/zijian/lewm_data/ttjepa_cube_triple_dynamic_whitened_oracle_k4_10e
```

| Whitened mode | Success rate | Mean predictor depth |
|---|---:|---:|
| fixed K=1 | 72% | 1.00 |
| fixed K=2 | 72% | 2.00 |
| fixed K=3 | 76% | 3.00 |
| fixed K=4 | 76% | 4.00 |
| learned 0.0005 | 68% | 2.0049 |
| learned 0.001 | 70% | 1.8594 |
| learned 0.003 | 74% | 1.6307 |
| learned 0.005 | 70% | 1.5205 |
| learned 0.01 | 74% | 1.3644 |
| learned 0.03 | 72% | 1.1904 |
| learned 0.1 | 70% | 1.0854 |
| learned 0.35 | 74% | 1.0189 |

Probe-weighted oracle paths:

```text
Config: config/train/lewm_recurrent_dynamic_probe_weighted.yaml
Log: /home/robotuser/zijian/TTJepa/logs/ttjepa_cube_triple_probe_weighted_oracle_20260617_193620.log
Checkpoint: /home/robotuser/zijian/lewm_data/checkpoints/ttjepa_cube_triple_dynamic_probe_weighted_oracle_k4_10e/weights_epoch_10.pt
Results: /home/robotuser/zijian/lewm_data/ttjepa_cube_triple_dynamic_probe_weighted_oracle_k4_10e
```

| Probe-weighted mode | Success rate | Mean predictor depth |
|---|---:|---:|
| fixed K=1 | 72% | 1.00 |
| fixed K=2 | 66% | 2.00 |
| fixed K=3 | 72% | 3.00 |
| fixed K=4 | 70% | 4.00 |
| learned 0.0005 | 72% | 1.5639 |
| learned 0.001 | 74% | 1.4775 |
| learned 0.003 | 70% | 1.3914 |
| learned 0.005 | 74% | 1.3146 |
| learned 0.01 | 72% | 1.2369 |
| learned 0.03 | 70% | 1.1576 |
| learned 0.1 | 76% | 1.0864 |
| learned 0.35 | 74% | 1.0222 |
| learned 0.5 | 76% | 1.0108 |
| learned 0.7 | 72% | 1.0009 |

Readout:

- Whitened labels change depth usage but do not improve learned success beyond
  `74%`; they are not enough on cube-triple.
- Probe-weighted labels improve learned halting to `76%` while using almost K=1
  compute, so the halt target is moving in the right direction.
- The probe-weighted checkpoint does not improve the underlying fixed-depth
  predictor: fixed K=4 is only `70%`. The strongest cube-triple recurrent result
  is still the raw dynamic-oracle checkpoint at fixed K=4 (`78%`).
- The next step should be decision-aware halting: continue when deeper rollout
  changes CEM action choice, value ranking, or goal-progress prediction, instead
  of relying on latent reconstruction distance alone.
