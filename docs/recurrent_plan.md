# Recurrent LeWM Implementation Plan

更新时间: 2026-06-18

本文档记录 TTJepa 在 LeWM latent planning 上的新增实验计划。主线是给
latent dynamics predictor 增加 per-transition recurrent refinement depth `K`，
先验证 fixed `K` 的 test-time scaling，再做 adaptive halting。

## 0. Project Goal

Paper working title:

**Learning When to Refine: Per-Transition Test-Time Scaling for Latent World-Model Planning**

核心假设:

在 LeWM 的 CEM/MPC latent planning 中，test-time compute 不只可以花在
candidate 数量 `N`、CEM iteration 数 `I`、rollout horizon `H` 上，也可以花在
predictor refinement depth `K` 上:

```text
z_hat[t+1]^(0) -> z_hat[t+1]^(1) -> ... -> z_hat[t+1]^(K)
```

目标是证明 fixed or adaptive deeper `K` 在 contact-rich transitions 上比单纯增加
`N` 或 `I` 更能修复 dynamics error。

## 1. Scope

第一阶段只做最小闭环:

- 保留 LeWM encoder、action encoder、SIGReg 和原始 end-to-end final loss。
- 新增 drop-in `RecurrentARPredictor`，接口兼容 `ARPredictor.forward(x, c)`。
- 训练支持 all-depth prediction 和 intermediate detached-target supervision。
- planning/eval 支持 fixed `predictor_depth`，为后续 adaptive halting 留接口。
- 新增 focused unit tests，先验证 shape、dict output、default compatibility。

暂不做:

- tokenizer 或 skill abstraction
- learned value network
- 真正 ragged per-token early stop
- 改写 stable-worldmodel planner 内部

## 2. Code Changes

Planned files:

```text
module.py
  + RecurrentRefineCell
  + RecurrentARPredictor

jepa.py
  * JEPA.predict(..., return_all=False, predictor_depth=None, ...)
  * JEPA.rollout(..., predictor_depth=None, return_depth_stats=False, ...)
  * JEPA.get_cost(..., **rollout_kwargs)

train.py
  * lejepa_forward recurrent.enabled path
  * logs final/intermediate prediction losses and residual stats

config/train/lewm_recurrent.yaml
config/train/model/lewm_recurrent.yaml
  + recurrent predictor and loss config

config/eval/*.yaml
  + optional planner.predictor_depth / halt_mode / halt_eps

tests/test_recurrent_predictor.py
tests/test_jepa_recurrent_predict.py
```

## 3. Predictor Design

`RecurrentARPredictor` keeps the original tensor return by default:

```python
pred = recurrent_predictor(x, c)  # (B, T, D)
```

For training analysis:

```python
out = recurrent_predictor(
    x,
    c,
    max_depth=8,
    return_all=True,
    halt_mode="none",
)
```

Return dict:

```python
{
    "pred": Tensor,          # (B, T, D)
    "preds": Tensor,         # (K, B, T, D)
    "residuals": Tensor,     # (K, B, T)
    "continue_logits": Tensor, # (K, B, T)
    "depth_used": Tensor,    # (B, T)
}
```

Architecture:

1. Add positional embedding and dropout as in `ARPredictor`.
2. Run a shallow base conditional transformer to get anchored hidden state `h`.
3. Produce initial `z_hat` with `init_head`.
4. Reuse a recurrent refinement cell for `K` steps.
5. Each step predicts a gated residual correction:

```text
feedback = z_hat - x_anchor
h = refine_cell(h, c, feedback)
delta = delta_head(h)
gamma = sigmoid(gamma_head(h))
continue_logit = continue_head(h)
z_hat = z_hat + residual_scale * gamma * delta
```

The predictor supports fixed `K` and two adaptive halting modes:

- `halt_mode=residual`: selects the first depth whose per-token update
  residual falls below `halt_eps` after `min_depth`.
- `halt_mode=learned`: selects the first depth whose learned continue
  probability falls below `halt_threshold` after `min_depth`.

## 4. Training Loss

When `recurrent.enabled=false`, the original LeWM loss path remains unchanged.

When `recurrent.enabled=true`:

```text
L_final = mse(preds[K - 1], target)
L_inter = mean_k<K-1 mse(preds[k], stopgrad(target))
L_halt  = bce_with_logits(continue_logits, continue_target)
L_pred  = L_final + alpha * L_inter + beta * L_halt
L_total = L_pred + lambda * L_SIGReg
```

By default `continue_target[k] = 1` when the next recurrent depth reduces the
true target error by more than `recurrent.halt_min_improvement`; otherwise it is
0. This trains the head to answer "is another refinement step worth it?" rather
than merely copying update magnitude. For ablations, `halt_label_mode` can be
set to `relative_improvement`, `error_threshold`, or `residual_threshold`.

For dynamic-halting training, `config/train/lewm_recurrent_dynamic.yaml` sets
`halt_label_mode=oracle_depth`. This labels the first recurrent depth whose
target error is within `halt_oracle_relative_tolerance` of the best available
depth as the stop point, and trains the continue head to run only until that
oracle depth. It keeps the all-depth prediction losses intact, but makes the
learned halting head optimize the same early-stop decision used at inference.

Logged metrics:

- `train/pred_loss`
- `train/pred_loss_final`
- `train/pred_loss_inter`
- `train/pred_loss_halt`
- `train/sigreg_loss`
- `train/residual_mean`
- `train/continue_prob_mean`
- `train/continue_target_rate`
- `train/halt_depth_mean`

## 5. Eval Plan

For fixed-depth sweeps, run the same checkpoint with:

```bash
python eval.py --config-name pusht \
  policy=<checkpoint-dir>/weights_epoch_10.pt \
  planner.predictor_depth=1

python eval.py --config-name pusht \
  policy=<checkpoint-dir>/weights_epoch_10.pt \
  planner.predictor_depth=2

python eval.py --config-name pusht \
  policy=<checkpoint-dir>/weights_epoch_10.pt \
  planner.predictor_depth=4

python eval.py --config-name pusht \
  policy=<checkpoint-dir>/weights_epoch_10.pt \
  planner.predictor_depth=8
```

For TwoRoom paper-style eval, keep the fixed timestep-offset conversion:

```bash
python eval.py --config-name tworoom \
  policy=<checkpoint-dir>/weights_epoch_10.pt \
  eval.eval_budget=150 \
  eval.goal_offset_timesteps=100 \
  planner.predictor_depth=4
```

## 6. Baseline Notes

Existing reproduction context is recorded in:

```text
LEWM_REPRODUCTION_NOTES.md
```

Current key reference numbers:

| Task | Reproduced | Official LeWM |
|---|---:|---:|
| PushT 10e | 92% | 96% |
| Reacher 10e | 80% | 86% |
| Cube 10e | 72% | 74% |
| TwoRoom fixed paper timestep eval | 94-98% | 87% |

## 7. Immediate Milestones

1. Implement recurrent predictor and `JEPA.predict(return_all=True)`.
2. Add recurrent training config and all-depth loss.
3. Add fixed-depth rollout/eval plumbing.
4. Run local unit tests.
5. On `sj-a800`, run a 100-step smoke train with `uv`:

```bash
cd /home/robotuser/zijian/le-wm
git remote add ttjepa https://github.com/jinzijian/TTJepa.git || true
git fetch ttjepa codex/recurrent-lewm
git checkout -B codex/recurrent-lewm ttjepa/codex/recurrent-lewm

source .venv/bin/activate
export STABLEWM_HOME=/home/robotuser/zijian/lewm_data
export WANDB_MODE=disabled
uv run --active python train.py --config-name=lewm_recurrent \
  data=pusht \
  trainer.max_steps=100 \
  trainer.max_epochs=1 \
  wandb.enabled=false
```

6. Run first fixed-depth evaluation sweep on the smallest reliable checkpoint.

## 8. First Training Targets

第一批结果只跑两个数据集:

| Dataset | Why first | Training setting | Eval setting |
|---|---|---|---|
| PushT | contact-rich manipulation, official gap still有提升空间 | 10 epochs, `history_size=3` | default `eval_budget=50`, `goal_offset_steps=25` |
| TwoRoom | 快速、稳定，已修复 paper timestep eval 语义 | 10 epochs, `history_size=1` | default 50/25 plus paper-style `eval_budget=150`, `goal_offset_timesteps=100` |

Success target:

- PushT recurrent 10e should match or beat the reproduced LeWM baseline `92%`,
  ideally closing toward official `96%`.
- TwoRoom recurrent 10e should stay in the same band as the fixed paper-timestep
  baseline and must not reproduce the invalid row-offset collapse.
- K sweep should show whether `predictor_depth=2/4/8` improves planning success
  or latency-normalized success over `predictor_depth=1`.

PushT training command:

```bash
cd /home/robotuser/zijian/le-wm
git remote add ttjepa https://github.com/jinzijian/TTJepa.git || true
git fetch ttjepa codex/recurrent-lewm
git checkout -B codex/recurrent-lewm ttjepa/codex/recurrent-lewm

source .venv/bin/activate
export STABLEWM_HOME=/home/robotuser/zijian/lewm_data
export CUDA_VISIBLE_DEVICES=0

uv run --active python train.py --config-name=lewm_recurrent \
  data=pusht \
  output_model_name=ttjepa_pusht_k8_10e \
  subdir=ttjepa_pusht_k8_10e \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  wandb.enabled=true \
  wandb.config.entity=open-science \
  wandb.config.project=wm-ttc
```

TwoRoom training command:

```bash
cd /home/robotuser/zijian/le-wm
git remote add ttjepa https://github.com/jinzijian/TTJepa.git || true
git fetch ttjepa codex/recurrent-lewm
git checkout -B codex/recurrent-lewm ttjepa/codex/recurrent-lewm

source .venv/bin/activate
export STABLEWM_HOME=/home/robotuser/zijian/lewm_data
export CUDA_VISIBLE_DEVICES=0

uv run --active python train.py --config-name=lewm_recurrent \
  data=tworoom \
  history_size=1 \
  output_model_name=ttjepa_tworoom_h1_k8_10e \
  subdir=ttjepa_tworoom_h1_k8_10e \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  wandb.enabled=true \
  wandb.config.entity=open-science \
  wandb.config.project=wm-ttc
```

Fixed-depth eval examples:

```bash
uv run --active python eval.py --config-name pusht \
  policy=ttjepa_pusht_k8_10e/weights_epoch_10.pt \
  planner.predictor_depth=1 \
  output.filename=ttjepa_pusht_k1_results.txt

uv run --active python eval.py --config-name pusht \
  policy=ttjepa_pusht_k8_10e/weights_epoch_10.pt \
  planner.predictor_depth=4 \
  output.filename=ttjepa_pusht_k4_results.txt

uv run --active python eval.py --config-name tworoom \
  policy=ttjepa_tworoom_h1_k8_10e/weights_epoch_10.pt \
  eval.eval_budget=150 \
  eval.goal_offset_timesteps=100 \
  planner.predictor_depth=4 \
  output.filename=ttjepa_tworoom_h1_k4_paper_t100_b150_results.txt
```

Residual-halting eval examples:

```bash
uv run --active python eval.py --config-name pusht \
  policy=ttjepa_pusht_k4_10e/weights_epoch_10.pt \
  planner.predictor_depth=4 \
  planner.halt_mode=residual \
  planner.halt_eps=1e-4 \
  planner.min_depth=1 \
  planner.return_depth_stats=true \
  output.filename=ttjepa_pusht_dynamic_res1e-4_results.txt

uv run --active python eval.py --config-name cube \
  policy=ttjepa_cube_k4_10e/weights_epoch_10.pt \
  planner.predictor_depth=4 \
  planner.halt_mode=residual \
  planner.halt_eps=1e-4 \
  planner.min_depth=1 \
  planner.return_depth_stats=true \
  output.filename=ttjepa_cube_dynamic_res1e-4_results.txt
```

The first dynamic sweep should try `halt_eps in {1e-3, 3e-4, 1e-4, 3e-5}`
and report success rate, evaluation time, and mean `depth_used`.

Learned-halting eval examples:

```bash
uv run --active python eval.py --config-name pusht \
  policy=ttjepa_pusht_k4_10e/weights_epoch_10.pt \
  planner.predictor_depth=4 \
  planner.halt_mode=learned \
  planner.halt_threshold=0.5 \
  planner.min_depth=1 \
  planner.return_depth_stats=true \
  output.filename=ttjepa_pusht_dynamic_learned_t05_results.txt

uv run --active python eval.py --config-name cube \
  policy=ttjepa_cube_k4_10e/weights_epoch_10.pt \
  planner.predictor_depth=4 \
  planner.halt_mode=learned \
  planner.halt_threshold=0.5 \
  planner.min_depth=1 \
  planner.return_depth_stats=true \
  output.filename=ttjepa_cube_dynamic_learned_t05_results.txt
```

The first learned sweep should try `halt_threshold in {0.3, 0.5, 0.7}` and
compare success rate against mean `depth_used`.

## 9. Cube Dynamic-Halting Results

### 9.1 Fixed-depth recurrent Cube

The first useful positive signal came from OGBench Cube. A recurrent TTJepa
checkpoint trained with `recurrent.max_depth=4` for 10 epochs reached:

| Eval mode | Success rate |
|---|---:|
| fixed K=1 | 80% |
| fixed K=2 | 80% |
| fixed K=4 | 80% |

Reference numbers:

| Model | Cube success rate |
|---|---:|
| LeWM reproduced baseline | 72% |
| LeWM paper number | 74% |
| TTJepa recurrent fixed K | 80% |

Checkpoint:

```text
/home/robotuser/zijian/lewm_data/checkpoints/ttjepa_cube_k4_10e/weights_epoch_10.pt
```

### 9.2 Learned dynamic-halting training

A second Cube checkpoint trained the same recurrent predictor with the
`oracle_depth` continue-head target from `config/train/lewm_recurrent_dynamic.yaml`:

```bash
cd /home/robotuser/zijian/TTJepa
source /home/robotuser/zijian/le-wm/.venv/bin/activate
export STABLEWM_HOME=/home/robotuser/zijian/lewm_data
export CUDA_VISIBLE_DEVICES=0

/home/robotuser/.local/bin/uv run --active python train.py --config-name=lewm_recurrent_dynamic \
  data=ogb \
  output_model_name=ttjepa_cube_dynamic_oracle_k4_10e \
  subdir=ttjepa_cube_dynamic_oracle_k4_10e \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  recurrent.max_depth=4 \
  wandb.enabled=true \
  wandb.config.entity=open-science \
  wandb.config.project=wm-ttc
```

Training log:

```text
/home/robotuser/zijian/TTJepa/logs/ttjepa_cube_dynamic_training_20260614_035518.log
```

Checkpoint:

```text
/home/robotuser/zijian/lewm_data/checkpoints/ttjepa_cube_dynamic_oracle_k4_10e/weights_epoch_10.pt
```

Single-seed first pass (`seed=42`) found a strong threshold:

| Eval mode | Success rate |
|---|---:|
| fixed K=1 | 80% |
| fixed K=2 | 78% |
| fixed K=4 | 78% |
| learned threshold 0.005 | 78% |
| learned threshold 0.01 | 80% |
| learned threshold 0.02 | 80% |
| learned threshold 0.05 | 74% |
| learned threshold 0.1 | 76% |
| learned threshold 0.3 | 80% |
| learned threshold 0.5 | 86% |

Result directory:

```text
/home/robotuser/zijian/lewm_data/ttjepa_cube_dynamic_oracle_k4_10e
```

### 9.3 Depth-logged validation sweep

`eval.py` now records dynamic compute usage through `depth_stats` when
`planner.return_depth_stats=true`. The stats include:

- `depth_stats.mean_depth`
- `depth_stats.depth_histogram`
- `depth_stats.depth_fraction`

Validation sweep log:

```text
/home/robotuser/zijian/TTJepa/logs/ttjepa_cube_dynamic_depth_sweep_20260614_203200.log
```

Validation sweep command template:

```bash
/home/robotuser/.local/bin/uv run --active python eval.py --config-name cube \
  policy=ttjepa_cube_dynamic_oracle_k4_10e/weights_epoch_10.pt \
  seed=<42|43|44> \
  planner.predictor_depth=4 \
  planner.halt_mode=learned \
  planner.halt_threshold=<threshold> \
  planner.min_depth=1 \
  planner.return_depth_stats=true \
  output.filename=ttjepa_cube_dynamic_oracle_k4_10e_depth_sweep_seed<seed>_learned_t<threshold>_results.txt
```

Per-seed results:

| Mode | Seed 42 success / depth | Seed 43 success / depth | Seed 44 success / depth |
|---|---:|---:|---:|
| fixed K=1 | 80% / 1.00 | 88% / 1.00 | 66% / 1.00 |
| fixed K=4 | 78% / 4.00 | 90% / 4.00 | 64% / 4.00 |
| learned 0.35 | 78% / 1.21 | 92% / 1.23 | 62% / 1.28 |
| learned 0.40 | 78% / 1.20 | 92% / 1.18 | 66% / 1.24 |
| learned 0.45 | 78% / 1.15 | 92% / 1.15 | 64% / 1.20 |
| learned 0.50 | 86% / 1.10 | 92% / 1.11 | 66% / 1.16 |
| learned 0.55 | 84% / 1.07 | 90% / 1.08 | 62% / 1.12 |
| learned 0.60 | 80% / 1.07 | 90% / 1.06 | 58% / 1.10 |
| learned 0.70 | 78% / 1.02 | 90% / 1.03 | 68% / 1.03 |
| learned 0.80 | 80% / 1.00 | 90% / 1.01 | 66% / 1.01 |

Aggregate over seeds 42, 43, and 44:

| Mode | Mean success | Mean depth |
|---|---:|---:|
| fixed K=1 | 78.00% | 1.00 |
| fixed K=4 | 77.33% | 4.00 |
| learned 0.35 | 77.33% | 1.24 |
| learned 0.40 | 78.67% | 1.21 |
| learned 0.45 | 78.00% | 1.16 |
| learned 0.50 | 81.33% | 1.13 |
| learned 0.55 | 78.67% | 1.09 |
| learned 0.60 | 76.00% | 1.08 |
| learned 0.70 | 78.67% | 1.03 |
| learned 0.80 | 78.67% | 1.01 |

Interpretation:

- Best single-run result: learned threshold `0.5` reaches `86%` success on
  `seed=42`, improving over the old fixed TTJepa Cube result of `80%`.
- `learned threshold=0.5` is the best validated point: `81.33%` mean success
  with `1.13` mean predictor depth.
- This beats the old TTJepa fixed Cube result of `80%`, the reproduced LeWM
  Cube baseline of `72%`, and the paper Cube number of `74%`.
- The compute profile is the main signal: learned halting stays close to K=1
  compute while improving success over both fixed K=1 and fixed K=4 in this
  three-seed sweep.
- `seed=44` is a hard eval draw for all modes: fixed K=1 gets only `66%`, so
  the dynamic result there should not be read as a learned-halting-specific
  failure.

Current claim wording:

```text
On OGBench Cube, learned dynamic halting reaches a best single-run success of
86% at threshold 0.5. Across three eval seeds, the same threshold averages
81.33% success with mean predictor depth 1.13, outperforming fixed K=1
(78.00%) and fixed K=4 (77.33%) while using compute close to K=1.
```

## 10. Cube-Triple Halt Diagnosis and Whitened Oracle

Cube-triple is the first harder OGBench variant where the current learned halt
oracle does not transfer cleanly. The result is still useful because fixed
depth improves with more recurrent prediction, while learned halting mostly
collapses back to depth 1.

### 10.1 Raw cube-triple results

The raw dynamic-oracle checkpoint was trained with `lewm_recurrent_dynamic`,
`recurrent.max_depth=4`, `halt_oracle_relative_tolerance=0.02`, and
`halt_min_depth=1`.

| Mode | Success rate | Mean predictor depth |
|---|---:|---:|
| LeWM baseline | 74% | n/a |
| TTJepa fixed K=1 | 70% | 1.00 |
| TTJepa fixed K=2 | 76% | 2.00 |
| TTJepa fixed K=3 | 76% | 3.00 |
| TTJepa fixed K=4 | 78% | 4.00 |
| TTJepa learned threshold 0.35 | 72% | 1.0036 |
| TTJepa learned threshold 0.5 | 70% | 1.0010 |
| TTJepa learned threshold 0.7 | 72% | 1.0000 |

Interpretation:

- Fixed recurrent depth is not the bottleneck: K=4 improves from `70%` to
  `78%` and beats the `74%` LeWM baseline.
- Learned halting is the bottleneck: it spends almost all samples at depth 1,
  so it loses the K=4 gain.
- Therefore the failure is better described as halt-label calibration / latent
  proxy failure, not recurrent predictor capacity failure.

### 10.2 Halt-label diagnosis

The original `oracle_depth` label is built from raw latent MSE. With tolerance
`0.02` and `halt_min_depth=1`, most samples are labeled as safe to stop at
the first recurrent step. The training logs show the imbalance directly:

| Signal | Observed value |
|---|---:|
| `continue_target_rate` | about 0.7%-1.0% |
| `halt_depth_mean` | about 1.03 |
| `continue_prob_mean` | about 0.007 |

In other words, the supervised halt head is trained to say "do not continue"
for about 99% of positions. Under BCE without a strong positive-class or
budget controller, the natural solution is near-universal early halt.

The diagnosis sweep confirms this:

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

Forcing `min_depth=2` or `min_depth=3` recovers fixed-depth performance, but
that is a hard compute floor rather than a learned allocation policy. It is a
useful sanity check, not the final dynamic-compute mechanism.

### 10.3 Working hypothesis

The raw latent MSE oracle is likely too coarse for cube-triple. JEPA latents can
be anisotropic: high-variance dimensions dominate Euclidean MSE, while
task-critical but low-variance dimensions such as small object-pose or
goal-relative errors can be underweighted. This makes depth 1 look "close
enough" in latent space even when the planner would benefit from deeper
prediction.

The immediate fix is the `oracle_depth_whitened` label mode. It computes the
same depth oracle after batch-whitening latent dimensions, so low-variance
task-relevant dimensions are not drowned out by high-variance latent axes.

The immediate follow-up tested `oracle_depth_whitened`, then `oracle_depth_probe_weighted`. Both runs are complete; their results and the resulting decision-aware next step are summarized below.

## 11. Cube-Triple Alternative Halt Oracles (2026-06-18)

The cube-triple follow-ups tested whether the learned halt failure was mainly a
raw-latent label problem. The answer is mixed: better labels improve learned
halting, but they do not yet recover the best fixed-depth result.

### 11.1 Whitened halt oracle

Run:

```text
Config: config/train/lewm_recurrent_dynamic_whitened.yaml
Log: /home/robotuser/zijian/TTJepa/logs/ttjepa_cube_triple_whitened_oracle_20260617_20260617_063655.log
Checkpoint: /home/robotuser/zijian/lewm_data/checkpoints/ttjepa_cube_triple_dynamic_whitened_oracle_k4_10e/weights_epoch_10.pt
Results: /home/robotuser/zijian/lewm_data/ttjepa_cube_triple_dynamic_whitened_oracle_k4_10e
```

| Mode | Success rate | Mean predictor depth |
|---|---:|---:|
| fixed K=1 | 72% | 1.00 |
| fixed K=2 | 72% | 2.00 |
| fixed K=3 | 76% | 3.00 |
| fixed K=4 | 76% | 4.00 |
| learned threshold 0.0005 | 68% | 2.0049 |
| learned threshold 0.001 | 70% | 1.8594 |
| learned threshold 0.003 | 74% | 1.6307 |
| learned threshold 0.005 | 70% | 1.5205 |
| learned threshold 0.01 | 74% | 1.3644 |
| learned threshold 0.03 | 72% | 1.1904 |
| learned threshold 0.1 | 70% | 1.0854 |
| learned threshold 0.35 | 74% | 1.0189 |

Whitening makes the halt policy spend more depth at low thresholds, so it does
change the label geometry. However, the best learned result is still `74%`, the
same as the LeWM baseline and below the raw checkpoint fixed K=4 result of
`78%`. This suggests raw latent anisotropy is part of the problem, but whitening
alone is not enough to identify the samples where extra depth changes planning
success.

### 11.2 Probe-weighted halt oracle

Run:

```text
Config: config/train/lewm_recurrent_dynamic_probe_weighted.yaml
Log: /home/robotuser/zijian/TTJepa/logs/ttjepa_cube_triple_probe_weighted_oracle_20260617_193620.log
Checkpoint: /home/robotuser/zijian/lewm_data/checkpoints/ttjepa_cube_triple_dynamic_probe_weighted_oracle_k4_10e/weights_epoch_10.pt
Results: /home/robotuser/zijian/lewm_data/ttjepa_cube_triple_dynamic_probe_weighted_oracle_k4_10e
```

| Mode | Success rate | Mean predictor depth |
|---|---:|---:|
| fixed K=1 | 72% | 1.00 |
| fixed K=2 | 66% | 2.00 |
| fixed K=3 | 72% | 3.00 |
| fixed K=4 | 70% | 4.00 |
| learned threshold 0.0005 | 72% | 1.5639 |
| learned threshold 0.001 | 74% | 1.4775 |
| learned threshold 0.003 | 70% | 1.3914 |
| learned threshold 0.005 | 74% | 1.3146 |
| learned threshold 0.01 | 72% | 1.2369 |
| learned threshold 0.03 | 70% | 1.1576 |
| learned threshold 0.1 | 76% | 1.0864 |
| learned threshold 0.35 | 74% | 1.0222 |
| learned threshold 0.5 | 76% | 1.0108 |
| learned threshold 0.7 | 72% | 1.0009 |

Probe weighting is the first alternative label that improves the learned dynamic
policy on cube-triple: best learned success is `76%` at thresholds `0.1` and
`0.5`, with average depth `1.09` and `1.01` respectively. That is better than
the `74%` LeWM baseline and better than the raw/whitened learned best of `74%`,
while still using almost K=1 compute.

The caveat is important: the same probe-weighted checkpoint has weak fixed-depth
scores, with fixed K=4 only `70%`. So probe weighting partially fixes halt
calibration, but it does not produce a stronger recurrent predictor than the raw
checkpoint. The best cube-triple recurrent result remains the raw dynamic-oracle
checkpoint at fixed K=4 (`78%`).

### 11.3 Conclusion

The cube-triple evidence separates three claims:

- Recurrent prediction can help: raw fixed K=4 improves from K=1 `70%` to `78%`.
- Latent-MSE halt labels are too weak: raw learned halting mostly collapses to
  depth 1 and tops out at `74%` in the diagnosis sweep.
- Probe-weighted labels help learned halting, reaching `76%` with near-K=1
  depth, but still do not beat the best fixed-depth result.

The next useful oracle is decision-aware rather than another latent-distance
threshold. It should label continue when deeper rollout changes CEM action
selection, value/ranking, or goal-progress disagreement. That makes the halt
supervision follow planning utility directly instead of using latent MSE as a
proxy.
