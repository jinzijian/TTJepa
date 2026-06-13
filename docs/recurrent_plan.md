# Recurrent LeWM Implementation Plan

更新时间: 2026-06-12

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

Logged metrics:

- `train/pred_loss`
- `train/pred_loss_final`
- `train/pred_loss_inter`
- `train/pred_loss_halt`
- `train/sigreg_loss`
- `train/residual_mean`
- `train/continue_prob_mean`
- `train/continue_target_rate`

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
