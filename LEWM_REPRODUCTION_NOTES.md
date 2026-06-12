# LeWM Appendix E 10-Epoch Reproduction Notes

更新时间: 2026-06-12

本文档记录在 `sj-a800` 上复现 `lucas-maes/le-wm` 论文 Appendix E 10-epoch 设置的过程、结果、已修复问题和后续注意事项。

## 路径和环境

- 远端机器: `sj-a800`
- 远端 repo: `/home/robotuser/zijian/le-wm`
- 数据根目录: `/home/robotuser/zijian/lewm_data`
- 环境: `/home/robotuser/zijian/le-wm/.venv`
- 安装方式: `uv`
- W&B project: `open-science/wm-ttc`

关键依赖调整:

- 官方安装入口:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

- 为了适配机器 CUDA/driver 和代码 import，实际固定过:
  - `torch==2.6.0+cu124`
  - `torchvision==0.21.0+cu124`
  - `transformers==4.48.3`
  - `hdf5plugin==6.0.0`
- Reacher eval 曾因 `dm-control 1.0.41` 与 `mujoco 3.9.0` 不兼容失败，错误为:

```text
AttributeError: 'MjModel' object has no attribute 'flex_bandwidth'
```

已用 `uv` 降级为:

```bash
uv pip install mujoco==3.8.1
```

并验证 Reacher env 可以 reset/render。

## 数据

数据位于:

```text
/home/robotuser/zijian/lewm_data
```

主要 HDF5 文件:

| Dataset | File | Size |
|---|---|---:|
| PushT | `pusht_expert_train.h5` | 46.3 GB |
| TwoRoom | `tworoom.h5` | 12.8 GB |
| Reacher | `reacher.h5` | 98.9 GB |
| Cube | `cube_single_expert.h5` | 101.9 GB |

数据集 loader smoke:

| Dataset name | Length | Notes |
|---|---:|---|
| `pusht_expert_train` | 1,981,721 | action dim 2 |
| `tworoom` | 920,809 | action dim 2, proprio dim 2 |
| `dmc/reacher_random` | 1,820,000 | action dim 2, observation dim 6 |
| `ogbench/cube_single_expert` | 1,820,000 | action dim 5, observation dim 28 |

TwoRoom 数据分布:

- episodes: `10000`
- average length: `92.0809`
- max `step_idx`: `100`
- max episode rows: `101`

## 论文设置

论文 Appendix E:

- TwoRoom: 10 epochs
- PushT: 10 epochs
- OGBench-Cube: 10 epochs
- Reacher: 10 epochs

论文 Appendix D:

- frame-skip: `5`
- sub-trajectory size: `4`
- PushT and OGBench-Cube history length: `3`
- TwoRoom history length: `1`

论文 Appendix F.1:

- TwoRoom: evaluation budget `150 steps`, goal `100 timesteps` in future
- PushT: evaluation budget `50`, goal `25 timesteps` in future
- OGBench-Cube and Reacher: evaluation budget `50`, goal `25 timesteps` in future

论文 Figure 6 LeWM 数字:

| Task | Official LeWM |
|---|---:|
| TwoRoom | 87% |
| Reacher | 86% |
| PushT | 96% |
| OGBench-Cube | 74% |

论文链接:

- Appendix D/F.1/Figure 6: https://arxiv.org/html/2603.19312v1

## 训练和 eval 入口

训练需要先 import HDF5 format:

```bash
cd /home/robotuser/zijian/le-wm
source .venv/bin/activate
export STABLEWM_HOME=/home/robotuser/zijian/lewm_data
export CUDA_VISIBLE_DEVICES=0

python -c 'import stable_worldmodel.data.formats.hdf5, runpy; runpy.run_path("train.py", run_name="__main__")' \
  data=<pusht|tworoom|dmc|ogb> \
  data.dataset.name=<dataset-name>.h5 \
  output_model_name=<model-name> \
  subdir=<model-name> \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  wandb.enabled=true \
  wandb.config.entity=open-science \
  wandb.config.project=wm-ttc
```

Eval:

```bash
cd /home/robotuser/zijian/le-wm
source .venv/bin/activate
export STABLEWM_HOME=/home/robotuser/zijian/lewm_data
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

python eval.py --config-name <pusht|tworoom|reacher|cube> \
  policy=<checkpoint-dir>/weights_epoch_10.pt \
  eval.num_eval=50 \
  output.filename=<result-file>.txt
```

## 最终结果

| Task | Checkpoint | Eval setting | Result | Official | Gap |
|---|---|---|---:|---:|---:|
| PushT fresh 10e | `lewm_pusht_paper10_fresh/weights_epoch_10.pt` | default 50/25 | 92% | 96% | -4 pp |
| Reacher 10e | `lewm_reacher_paper10/weights_epoch_10.pt` | default 50/25 | 80% | 86% | -6 pp |
| Cube 10e | `lewm_cube_paper10/weights_epoch_10.pt` | default 50/25 | 72% | 74% | -2 pp |
| TwoRoom old 10e | `lewm_tworoom_paper10/weights_epoch_10.pt` | default 50/25 | 88% | 87% | +1 pp |
| TwoRoom h1 10e | `lewm_tworoom_paper10_h1/weights_epoch_10.pt` | default 50/25 | 92% | 87% | +5 pp |
| TwoRoom old 10e | literal row offset 100, budget 150 | 22% | 87% | invalid comparison |
| TwoRoom h1 10e | literal row offset 100, budget 150 | 40% | 87% | invalid comparison |
| TwoRoom old 10e | fixed paper timestep 100, budget 150 | 94% | 87% | +7 pp |
| TwoRoom h1 10e | fixed paper timestep 100, budget 150 | 98% | 87% | +11 pp |

主要结果文件:

```text
/home/robotuser/zijian/lewm_data/lewm_pusht_paper10_fresh/pusht_paper10_fresh_epoch10_results.txt
/home/robotuser/zijian/lewm_data/lewm_reacher_paper10/reacher_paper10_epoch10_results.txt
/home/robotuser/zijian/lewm_data/lewm_cube_paper10/cube_paper10_epoch10_results.txt
/home/robotuser/zijian/lewm_data/lewm_tworoom_paper10/tworoom_paper10_epoch10_results.txt
/home/robotuser/zijian/lewm_data/lewm_tworoom_paper10_h1/tworoom_h1_epoch10_default_results.txt
/home/robotuser/zijian/lewm_data/lewm_tworoom_paper10_h1/tworoom_h1_epoch10_paper_timestep100_budget150_fixed_results.txt
/home/robotuser/zijian/lewm_data/lewm_tworoom_paper10/tworoom_old_epoch10_paper_timestep100_budget150_fixed_results.txt
```

## TwoRoom 异常和修复

### 问题现象

直接按照字面运行:

```bash
python eval.py --config-name tworoom \
  policy=lewm_tworoom_paper10_h1/weights_epoch_10.pt \
  eval.num_eval=50 \
  eval.eval_budget=150 \
  eval.goal_offset_steps=100
```

得到:

| Model | Result |
|---|---:|
| old `history_size=3` | 22% |
| h1 `history_size=1` | 40% |

这和 Figure 6 的 TwoRoom LeWM `87%` 不一致。

### 定位

`eval.py` 原先把 `eval.goal_offset_steps` 直接当 HDF5 row offset:

```python
goal = start_step + goal_offset_steps
```

但 TwoRoom HDF5 中:

- max `step_idx` 是 `100`
- episode max length 是 `101`
- 当 `goal_offset_steps=100` 时，合法 start 只能是 `step_idx=0`

实际采样退化为:

```text
selected start step min/mean/max: 0 / 0.0 / 0
```

这不符合 Appendix F.1 的描述:

> The initial state is chosen by randomly sampling a state from a trajectory in the dataset.

更合理的解释是:

- 论文写的 `100 timesteps` 是 raw timestep
- released HDF5 是 frame-skipped row
- Appendix D 写了 frame-skip = 5
- 因此 `100 timesteps` 应转换为 `100 / 5 = 20` HDF5 rows

### 诊断结果

固定 h1 checkpoint 和 `eval_budget=150`，只改变 HDF5 row offset:

| HDF5 row offset | Result |
|---:|---:|
| 20 | 98% |
| 25 | 92% |
| 50 | 62% |
| 75 | 56% |
| 100 | 40% |

这说明失败不是 `eval_budget=150` 导致，而是 literal `goal_offset_steps=100` 在当前 HDF5 row 单位里过远，并且改变了起点采样分布。

### 修复

在远端 repo 中修改:

```text
/home/robotuser/zijian/le-wm/eval.py
/home/robotuser/zijian/le-wm/config/eval/tworoom.yaml
```

新增配置:

```yaml
eval:
  goal_offset_steps: 25
  goal_offset_timesteps: null
  dataset_frameskip: 5
```

新增逻辑:

- `goal_offset_steps` 保持旧行为，仍解释为 HDF5 row offset
- `goal_offset_timesteps` 表示论文中的 raw timestep offset
- 当设置 `goal_offset_timesteps=100` 时，`eval.py` 会按 `dataset_frameskip=5` 转换为 HDF5 row offset `20`

运行时会打印:

```text
Resolved goal_offset_timesteps=100 with dataset_frameskip=5 to goal_offset_steps=20.
```

修复后命令:

```bash
python eval.py --config-name tworoom \
  policy=lewm_tworoom_paper10_h1/weights_epoch_10.pt \
  eval.num_eval=50 \
  eval.eval_budget=150 \
  eval.goal_offset_timesteps=100 \
  output.filename=tworoom_h1_epoch10_paper_timestep100_budget150_fixed_results.txt
```

修复后结果:

| Model | Fixed result |
|---|---:|
| old `history_size=3` | 94% |
| h1 `history_size=1` | 98% |

因此 TwoRoom 的异常已经修复。之后表达论文的 “100 timesteps” 时，应使用:

```bash
eval.goal_offset_timesteps=100
```

不要再使用:

```bash
eval.goal_offset_steps=100
```

后者仍然表示 literal HDF5 row offset，会回到异常评估分布。

## 当前远端 git 状态说明

当前远端 repo 有以下我们关心的改动:

```text
M config/eval/tworoom.yaml
M eval.py
```

另有已有或运行产生的非核心项:

```text
M config/train/launcher/local.yaml
?? logs/
?? outputs/
?? wandb/
?? wandb_resume.json
```

其中 `config/train/launcher/local.yaml` 不是本次 TwoRoom 修复引入的核心变更。`logs/`, `outputs/`, `wandb/`, `wandb_resume.json` 是运行产物。

## 总结

- PushT/Reacher/Cube 基本复现，结果比官方低 `2-6 pp`，属于可接受范围。
- TwoRoom default eval 原本已经接近官方。
- TwoRoom `150/100` 异常的原因是 timestep 与 HDF5 row 单位错配。
- 修复后 TwoRoom paper timestep eval 结果为 `94-98%`，不再异常。
- 后续复现实验若要严格表达论文 F.1 的 TwoRoom `100 timesteps`，应使用 `eval.goal_offset_timesteps=100`。
