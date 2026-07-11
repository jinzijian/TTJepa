# RefineJEPA：用于 JEPA 潜在规划的动态 K

RefineJEPA 研究的是 **latent world-model planning 里的动态测试时计算**：
不是让每个 imagined transition 都用同样的 transition-model depth，而是让模型学习哪些 transition 值得多做几次 recurrent refinement。

当前核心问题：

> 在 latent MPC/CEM planning 中，哪些 imagined transitions 值得更深地 refine？

这个仓库是 RefineJEPA / TTJepa 的代码和实验账本。本地源码包含 recurrent transition predictor、learned continue-head 训练、eval hook 和论文结果记录；大规模训练和评估主要在远端 TTJepa workspace 上执行：

- SSH：`ssh -p 20747 root@115.190.235.210`
- repo：`/vepfs/zijian/TTJepa`
- 数据/结果：`/vepfs/zijian/lewm_data`
- 分支：`codex/recurrent-lewm`

原始 LeWM README 内容保留在 [LEWM_REPRODUCTION_NOTES.md](LEWM_REPRODUCTION_NOTES.md)。当前 README 是 RefineJEPA-first 的项目入口。

## Motivation

想象一个机器人去拿桌上的物体。

把手伸过去时，大部分 free-space motion 都比较容易预测，粗一点的 latent transition 通常就够了。但到了接触、抓取、抬起、滑动或者多个物体相互影响的时候，小的 dynamics error 可能会改变 planner 最后选择的动作。

所以合理的 test-time compute 分配不是“每一步都想很久”，也不是“永远只想一步”，而是：

- 简单的 imagined transition 少算；
- 可能影响动作选择的 contact / interaction transition 多 refine 几次。

RefineJEPA 关注的正是这个 transition-level compute axis：动态选择每个 imagined transition 的 recurrent refinement depth \(K\)。

## 方法概念

RefineJEPA 基于 LeWM-style latent planning：

1. 编码当前视觉 observation 和 goal；
2. 在 latent space 中 rollout CEM 候选 action sequences；
3. 根据 terminal goal-matching cost 选择动作；
4. 保持外部 planner 不变，只替换 transition predictor。

transition predictor 是 weight-tied recurrent predictor。同一个 imagined transition 可以得到：

\[
\hat z_{t+1}^{(1)}, \hat z_{t+1}^{(2)}, \ldots, \hat z_{t+1}^{(K_{\max})}.
\]

每一层后都有一个轻量 continue head，判断是否继续 refine。当前主实验使用 relative marginal MSE supervision：

\[
y_k =
\mathbb{I}
\left[
\frac{e_k - e_{k+1}}{e_k+\epsilon}
>
\tau_{\mathrm{rel}}
\right],
\qquad
\tau_{\mathrm{rel}}=5\times10^{-4}.
\]

训练时 \(e_k\) 是 depth \(k\) 的 raw latent MSE。测试时未来 latent 不可用，模型只能用 continue head 自己决定停还是继续。

主实验配置：

| 设置 | 值 |
| --- | ---: |
| 最大 refinement depth | \(K_{\max}=4\) |
| continue label | relative marginal MSE improvement |
| \(\tau_{\mathrm{rel}}\) | \(5\times10^{-4}\) |
| continue-head loss weight | \(0.2\) |
| minimum depth | \(1\) |

这里的 mean \(K\) 是所有 CEM imagined transition predictions 上选中整数 depth 的平均值，不是每个 transition 内部真的用了小数 K。

## 实现细节：Continue Head 和 Action Conditioning

**latent continue head 的监督信号是什么？** 当前 learned dynamic-\(K\) head 用的是 relative marginal raw latent-MSE improvement。训练时 predictor 会一次性跑出所有 depth，并把每个 depth 的 prediction 和真实下一步 latent 比较。depth \(k\) 的 continue label 是：

\[
y_k =
\mathbb{I}
\left[
\frac{e_k - e_{k+1}}{e_k+\epsilon}
>
5\times10^{-4}
\right].
\]

也就是说，如果多 refine 一层能让 next-latent MSE 有足够相对下降，就标成 continue；否则标成 stop。测试时没有未来 target latent，所以只用 learned continue head 自己决定是否继续。

**transition layer 是什么结构？** 当前 transition predictor 是 recurrent、action-conditioned transformer predictor，不是单纯 MLP。它包括：

- 一个 action-conditioned base transformer，`base_depth=2`；
- 一个共享的 recurrent refinement cell，`refine_depth=1`；
- 一个线性 `init_head` 生成初始预测；
- 线性 `delta_head` 和 `gamma_head` 做 residual latent update；
- 一个线性 `continue_head` 预测 stop / continue。

refinement cell 在不同 depth 之间共享参数，所以增加 \(K\) 是重复使用同一个 cell，而不是堆一个更深、参数更多的 untied model。

**continue head 是 MLP 还是 transformer？** continue head 本身只是一个线性 classifier：

\[
p_k=\sigma(W h_k+b).
\]

它不是 MLP，也不是 transformer。它能工作是因为输入 \(h_k\) 已经经过 action-conditioned transformer/refinement blocks，里面包含了 latent history、candidate action context 和当前 refinement feedback。

**它怎么 based on action？** raw action 先经过 action encoder 变成 action embedding，然后这些 action embeddings 会 condition base transformer 和每一层 recurrent refinement cell。也就是说 head 表面上只看 \(h_k\)，但 \(h_k\) 已经是 action-conditioned 的：

\[
p(\mathrm{continue})
=
g(h_k),
\qquad
h_k = F(h_{k-1}, z_{\mathrm{hist}}, a_{\mathrm{hist}}, \hat z^{(k-1)}-z_{\mathrm{anchor}}).
\]

所以不同 CEM candidate action sequences 会产生不同 hidden states，也就可以选择不同的 refinement depth。

## 主结果：Learned Dynamic K

下面是当前主实验结果：四个数据集都使用
\(\tau_{\mathrm{rel}}=0.0005\) 的 relative-improvement continue target。讨论时可以把这组设置简称为 `rel0005`；远端四数据集实验目录实际命名为 `rel00005`。Fixed \(K\) 和 learned dynamic \(K\) 都在同一个 checkpoint family 内评估。

| Dataset | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Best learned dynamic K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reacher | 81.3% | 84.0% | 82.0% | 83.3% | 82.7% | **85.3%@K1.03** |
| Cube Single | 72.0% | **79.3%** | 78.7% | 78.0% | 78.0% | **79.3%@K1.00** |
| Cube Double | **74.7%** | 72.0% | 72.0% | 72.7% | 72.0% | 74.0%@K1.26 |
| Cube Triple | 74.0% | 74.0% | 74.0% | 73.3% | 74.0% | **77.3%@K1.22** |

解读：

- Reacher 和 Cube Triple 上，learned dynamic \(K\) 超过 LeWM 和所有 fixed-depth baseline，同时平均 compute 接近 \(K=1\)。
- Cube Double 上，learned dynamic \(K\) 超过所有 fixed-depth RefineJEPA evaluation，但这版 3-seed 平均里原始 LeWM baseline 仍略高。这说明 recurrent predictor 的质量和 dynamic allocation 的质量要分开讨论。
- Cube Single 是一个很有用的反例/控制组：这个 checkpoint 中 fixed K1 已经最好，learned selector 基本全停在 K1，并没有乱花 compute。
- 主结论不是 “K 越深越好”。主结论是：transition depth 是真实有效的 compute axis，应该动态分配。

这张表取代旧的 `ttjepa_*_dynamic_oracle_k4_10e` learned-head sweep。旧 sweep 使用 raw target-MSE depth labels，作为历史对照保留在实验账本里。

continue threshold \(\eta\) 是 test-time sweep 参数，只用于选择 reported success/compute point；主表里不把它作为核心结果展示。当前 reported operating points 使用：Reacher \(\eta=0.45\)，Cube Single \(\eta=0.70\)，Cube Double \(\eta=0.45\)，Cube Triple \(\eta=0.30\)。

## Depth Allocation

learned dynamic policy 只在少数 imagined transitions 上花额外 compute。

| Dataset | Best dynamic | K=1 | K=2 | K=3 | K=4 | 超过 K1 的比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reacher | 85.3%@K1.03 | 96.77% | 3.11% | 0.12% | 0.003% | 3.23% |
| Cube Single | 79.3%@K1.00 | 99.984% | 0.017% | 0.0001% | 0% | 0.017% |
| Cube Double | 74.0%@K1.26 | 76.49% | 21.09% | 1.69% | 0.72% | 23.51% |
| Cube Triple | 77.3%@K1.22 | 89.41% | 3.71% | 2.64% | 4.25% | 10.59% |

![Depth allocation](figures/depth_allocation_rel00005.png)

rollout-step trace 显示，额外 refinement 在 CEM imagined rollout 内的位置也不是均匀的。

![Depth by rollout step](figures/depth_by_rollout_step_rel00005.png)

![Depth by rollout step breakdown](figures/depth_by_rollout_step_stacked_rel00005.png)

### PushT 成功例子的定性 trace

上面的 rollout-step 图是 aggregate 统计。我们还单独 trace 了一个
PushT H10/goal50 的成功 episode：`global=1474257`，threshold
\(\eta=0.30\)。这个 episode 在 evaluator 下 success=True，同时并不是全程都深算；
learned policy 的 mean \(K=1.287\)，其中 \(21.8\%\) 的 imagined
predictions 使用 \(K>1\)。最明显的 extra-refinement 区域集中在 imagined
rollout step \(+3\) 附近的一段 planning window。

![PushT success trace with high-refinement decisions](figures/pusht_success_highk_env6_trace_panel.png)

成功动图预览：

![PushT success dynamic K preview](figures/pusht_success_highk_env6_annotated.gif)

MP4：[pusht_success_highk_env6_annotated.mp4](figures/pusht_success_highk_env6_annotated.mp4)

## Current Raw Target-Latent MSE Diagnostic

下面的 raw target-latent MSE diagnostic 是当前 `rel00005` checkpoint family 上的 50-episode 机制分析切片，不是上面 3-seed 主表的同一统计口径。这个 diagnostic 在评估之后比较 `K1` 与 `K4` 的真实 target-latent MSE：只有当 `K4` 的 target MSE 比 `K1` 更低并超过某个 tolerance 时，才选择 `K4`。

它仍然**不能部署**，因为测试时未来 target latent 不可用；它的作用是分析 raw latent error 里到底有多少 refinement-allocation signal。

| Dataset | Fixed K1 | Fixed K4 | MSE diagnostic | Hindsight K1/K4 chooser | K1 fail / K4 success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 80%@K1.00 | 82%@K4.00 | 80%@K1.00 to K2.20 | 86%@K1.18 | 3 / 50 |
| Cube Single | 84%@K1.00 | 82%@K4.00 | 84%@K1.00 to K1.12 | 84%@K1.00 | 0 / 50 |
| Cube Double | 70%@K1.00 | 68%@K4.00 | 70%@K1.00 to K2.14 | 70%@K1.00 | 0 / 50 |
| Cube Triple | 74%@K1.00 | 74%@K4.00 | 74%@K1.00 to K1.84 | 76%@K1.06 | 1 / 50 |

结论：

- 在当前 checkpoint 上，raw target-MSE diagnostic 本身没有带来正收益；最好的 success 基本就是保持 shallow `K1`。
- 但 hindsight `K1/K4` chooser 仍然显示有少量 episode 确实需要更深 depth：Reacher 有 `3/50`，Cube Triple 有 `1/50`。
- 这让结论更清楚：target-latent MSE 是有用的分析工具，但它不等价于 planner utility。planner 真正关心的是 candidate ranking 和 selected action。

## 机制分析

我们测试过一个自然解释：更深的 recurrent refinement 是否只是把 latent 变平滑了，导致任务相关的 contact detail 丢失？

第一版分析不支持“全局 collapse”这个强结论。

分析路径：`analysis/k_smoothing_20260622`

![Spectrum K1 vs K4](analysis/k_smoothing_20260622/figures/spectrum_k1_vs_k4_scatter.png)

![Probe R2 K1 vs K4](analysis/k_smoothing_20260622/figures/probe_r2_k1_vs_k4_scatter.png)

![Category probe MSE K1 vs K4](analysis/k_smoothing_20260622/figures/category_probe_mse_k1_vs_k4_scatter.png)

观察：

- K1 到 K4 的 global latent spectrum 基本不变。
- linear state-probe performance 也基本不变。
- depth-helped 和 depth-hurt subset 没有明显 global-rank collapse signature。

更可能的问题是局部 planner alignment：某些很小的 latent 改变可能会影响 CEM candidate ranking 或 selected action，但不会明显改变全局 spectrum 或简单 probe。

## Future Work：Planner-Aware Continue Supervision

当前 continue head 学的是 latent prediction improvement。下一步可以让同一个 head 学更 planner-aware 的监督信号。

之前尝试过的 CEM-aware diagnostic 是离线做的：对同一批 CEM candidates 分别用 \(K=1,2,3,4\) 打 cost，然后从 top-k cost、elite set overlap、ranking change 等 planner features 里训练一个单独 MLP selector，判断是否继续到更深 \(K\)。这个实验说明 CEM ranking 里确实有 allocation signal，但它不是当前主方法：它没有集成进 transition predictor，而且当前实现需要先算多个 depth 的 CEM cost，不是真正便宜的 test-time stopping rule。

更干净的 future work 是：

1. 训练时采样一小批 candidate action sequences；
2. 用多个 depth rollout 它们；
3. 用 top-k CEM cost improvement 或 elite-ranking change 生成 planner-aware continue label；
4. 用这个 label 训练现有的 `continue_head(h_k)`，planner-derived label 走 stop-gradient；
5. 测试时仍然只用轻量 latent continue head 自动决定 K。

这样相当于把昂贵的 CEM-aware teacher distill 到 transition-level head 里。目标是把监督问题从“多 refine 一层是否降低 latent MSE”推进到更贴近 planner 的问题：“多 refine 一层是否改善 CEM 实际使用的 candidate ranking 或 action selection”。

## 实验账本

完整实验记录、路径和日志见：

- [TTJEPA_EXPERIMENT_RESULTS.md](TTJEPA_EXPERIMENT_RESULTS.md)
- [TTJEPA_DYNAMIC_K_RESEARCH.md](TTJEPA_DYNAMIC_K_RESEARCH.md)
- [paper.md](paper.md)

重要本地 artifacts：

- `analysis/k_refinement_rel00005_current/`
- `analysis/k_refinement_all_20260620_024634/`
- `analysis/k_smoothing_20260622/`
- `analysis/paper1_figures/`
- `figures/depth_allocation_rel00005.png`
- `figures/depth_by_rollout_step_rel00005.png`
- `figures/depth_by_rollout_step_stacked_rel00005.png`

命名说明：主结果是 \(\tau_{\mathrm{rel}}=0.0005\) 的四数据集 sweep。有些讨论里把它简称为 `rel0005`，但远端四数据集目录名是 `rel00005`。另有一个只在 Cube Triple 上跑过、目录名正好叫 `rel0005` 的辅助训练变体；它保留在实验账本里，不是这张四数据集主表。

## 当前清理状态

这个 repo 现在被整理成 RefineJEPA-first 的研究入口。当前主要文件分工是：

1. `jepa.py`：JEPA model 和 rollout/eval hook；
2. `module.py`：recurrent predictor 和 continue head；
3. `recurrent_halting.py`：continue-label construction；
4. `train.py`：training losses 和 logging；
5. `eval.py`：CEM/MPC evaluation entrypoint；
6. `config/`：train/eval Hydra configs；
7. `TTJEPA_EXPERIMENT_RESULTS.md`：完整实验账本；
8. `LEWM_REPRODUCTION_NOTES.md`：原始 LeWM 复现实验说明。

README 保持当前论文主线，长表格、失败变体和路径细节放进实验账本。
