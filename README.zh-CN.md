# RefineJEPA：用于 LeWM 潜在规划的动态测试时计算

RefineJEPA 是基于
[LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm) 代码库的研究分支。
当前问题很明确：

> 在 latent world-model planning 里，模型能不能自己决定每个 imagined
> transition 需要算多少层 `K`，而不是所有 transition 都用同一个固定深度？

一个直观例子：机器人把手伸向物体的自由空间运动通常比较简单，不需要想太多；
但真正接触、抓取、抬起、或者分辨多个物体时，transition 更难，应该允许 dynamics
predictor 多做几轮 refinement。

## 方法

LeWM 的 planner 会把当前视觉状态和 goal 编码到 latent space，在 latent space
里 rollout candidate action sequences，然后用 CEM/MPC 选 terminal goal cost 最好的
动作序列。RefineJEPA 不改 action space、不改 planner，只研究 transition predictor
内部的 recurrent refinement depth：

```text
z_hat[t+1]^(1) -> z_hat[t+1]^(2) -> ... -> z_hat[t+1]^(K)
```

当前主线是 raw latent MSE：

- fixed `K`：所有 imagined transitions 都用同一个深度，比如 `K=1/2/4`。
- dynamic `K`：模型自己预测是否需要继续 refine。
- 训练时用 raw latent MSE improvement 生成 stop/continue 监督信号。
- 测试时只用 learned continue head 自动停，不看真实 next latent。

## 主要结果记录

`LeWM baseline` 和 `Fixed K1` 不是同一个模型。LeWM baseline 是原始非 recurrent
transition predictor；Fixed K1 是 RefineJEPA recurrent predictor 只跑第一层
refinement。

| Dataset / run | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher seed42 | 80% | 88% | n/a | n/a | 86% | 这个 checkpoint 里 K4 比 K1 差 |
| Cube single seed42 | 72% | 80% | n/a | n/a | 78% | K4 略低于 K1 |
| Cube single seed43 | 72% | 88% | n/a | n/a | 90% | K4 比 K1 高 2 点 |
| Cube single seed44 | 72% | 66% | n/a | n/a | 64% | K4 略低于 K1 |
| Cube single 3-seed avg | 72% | 78% | n/a | n/a | 77.3% | 平均略低，但 seed43 说明 K4 可以有帮助 |
| Cube single original rerun `20260621_refixed_k1234` | 72% | 80% | 76% | 78% | 78% | K1 最好，K3/K4 回到 78% |
| Cube double original rerun `20260621_refixed_k1234` | 66% | 72% | 70% | 68% | 70% | 这个 run 里深层没有帮助 |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | 最清楚的正例，K2/K4 明显有用 |

结论不是“越深越好”，而是：`K` 确实会改变 planning success，但不同数据集、
不同 checkpoint 需要的深度不同，所以 dynamic allocation 是合理问题。

## Raw Latent MSE 诊断

第一版分析直接看 raw latent MSE：如果更深的 recurrent step 明显降低 next-latent
prediction error，就认为这个 transition 值得继续 refine。这个表是 post-hoc
诊断，不是最终部署方法。

| Dataset | Fixed K1 | Fixed K4 | Best raw-MSE dynamic K | Hindsight K1/K4 chooser | Depth-helped cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 88%@K1.00 | 86%@K4.00 | 88%@K1.06 to K2.32 | 92%@K1.12 | 2 / 50 |
| Cube single | 78%@K1.00 | 77.3%@K4.00 | 77.3%@K2.72 to K2.96 | 80.7%@K1.08 | 4 / 150 |
| Cube double | 72%@K1.00 | 70%@K4.00 | 72%@K1.00 to K2.62 | 72%@K1.00 | 0 / 50 |
| Cube triple | 70%@K1.00 | 78%@K4.00 | 76%@K2.32 | 82%@K1.36 | 6 / 50 |

这个结果说明 raw latent MSE 不弱。Cube triple 里它能从 `70%@K1` 提到
`76%@K2.32`，说明它确实有信号；但它追不上 fixed `K4=78%`，也追不上 hindsight
chooser `82%@K1.36`。弱点是它衡量的是 latent prediction error，不一定等于
planner 真正在乎的 action ranking / planner benefit。

## Learned Dynamic K

更干净的版本是把 raw latent MSE improvement 当训练监督，让 recurrent state
上的 continue head 学会预测要不要继续。测试时模型自动选 `K`。

四个数据集的 raw-MSE learned-halting 结果如下：

| Dataset | LeWM baseline | Fixed K1 | Fixed K4 | Best learned dynamic K | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| Reacher | 80% | 88%@K1.00 | 86%@K4.00 | 90%@K1.003 (`t=0.5`) | learned head 略高于两个 fixed depth，几乎 K1 compute |
| Cube single | 72% | 78.00%@K1.00 | 77.33%@K4.00 | 81.33%@K1.13 (`t=0.5`, 3 seeds) | 最干净的 learned dynamic-K 正结果 |
| Cube double | 66% | 72%@K1.00 | 70%@K4.00 | 72%@K1.003-1.020 (`t=0.35/0.5/0.7`) | 基本等于 K1；这个 checkpoint 不需要深层 |
| Cube triple | 74% | 70%@K1.00 | 78%@K4.00 | 74%@K1.40 (`t=0.001` diagnosis) | raw learned halt 分配深度太少，没拿回 K4 gain |

这个表比单看 cube-triple 更准确：Cube single 是最强正例，Reacher 有小正例，
Cube double 说明 dynamic head 可以学成“别多算”，Cube triple 则暴露 raw latent
MSE label 的失败模式。也就是说 learned dynamic K 不是完全没学会，而是在最需要深层
的 cube-triple 上没有把需要继续的样本挑出来。

另有一批 cube-triple joint-depth variant：`rel00005` 可到 `78%@K=1.064`，
`rel0005` 能让 K1/dynamic/K4 都到 `80%`。这批更像训练时正则化或缓解 smoothing
的线索，不作为四数据集 learned-halting 主表。

## 当前论文主线

1. 固定 K 分析：没有一个 fixed depth 永远最好。
2. Raw latent MSE：是合理 v0，也能在 cube-triple 上拿到实际提升。
3. Learned continue head：把 raw MSE supervision 变成可部署的 dynamic K。
4. 失败分析：latent MSE 不等于 planner benefit，所以后续要分析 CEM ranking、
   contact detail、latent smoothing。

## 和 LeWM 的关系

RefineJEPA 基于 LeWM。使用 base world-model 代码时请引用 LeWM：

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
