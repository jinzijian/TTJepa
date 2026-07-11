# RefineJEPA: Dynamic K for JEPA Latent Planning

中文说明见 [README.zh-CN.md](README.zh-CN.md).

RefineJEPA studies **dynamic test-time compute inside latent world-model
planning**. A weight-tied recurrent transition predictor can refine the same
imagined transition multiple times, while a jointly trained continue head
decides when to stop. The outer LeWM-style MPC/CEM planner is unchanged.

The central question is:

> In latent MPC/CEM planning, which imagined transitions are worth refining
> more deeply?

## Motivation

Moving a robot arm through free space is often easy to predict. Contact,
lifting, sliding, and multi-object interaction are more planning-critical: a
small dynamics error can change which action sequence CEM selects. Applying the
same transition-model depth everywhere therefore wastes computation on easy
predictions and may still under-refine the few decisions that matter.

![Motivation: adaptive transition refinement](analysis/readme_figures/motivation_dynamic_transition_refinement.png)

RefineJEPA introduces recurrent refinement depth \(K\) as a transition-level
compute axis. Each imagined transition uses an integer depth in
\(\{1,2,3,4\}\), but different CEM candidates and rollout steps may stop at
different depths.

## Method

RefineJEPA starts from LeWM-style latent planning: encode the current visual
observation and goal, roll out candidate action sequences in latent space, and
score each sequence by the terminal latent-to-goal distance. CEM repeatedly
keeps low-cost elite candidates, updates its action distribution, and finally
executes the first action of the selected sequence.

![Method: RefineJEPA dynamic K](analysis/readme_figures/method_refinejepa_dynamic_k.png)

The transition predictor contains a two-layer action-conditioned transformer
followed by one shared recurrent refinement block. Reusing this block produces

\[
\hat z_{t+1}^{(1)},\hat z_{t+1}^{(2)},\ldots,
\hat z_{t+1}^{(K_{\max})}, \qquad K_{\max}=4.
\]

The recurrent block is weight-tied, so increasing \(K\) increases inference
depth without increasing parameter count. Actions are encoded and injected into
both the base transformer and every refinement step. A shared linear continue
head reads the action-conditioned recurrent state:

\[
p_k = \sigma(W h_k+b).
\]

At evaluation time, refinement stops when \(p_k\leq\eta\); otherwise the same
cell runs again, up to \(K_{\max}\). The future observation is never used for
this test-time decision.

### What supervises the recurrent predictor and continue head?

The main four-task model is trained jointly, rather than attaching a selector
after training. Its objective is

\[
\mathcal L =
\mathcal L_{\mathrm{final}}
+0.5\,\mathcal L_{\mathrm{inter}}
+0.2\,\mathcal L_{\mathrm{cont}}
+0.09\,\mathcal L_{\mathrm{SIGReg}}.
\]

The four terms play different roles:

- **Final-depth prediction.**
  \(\mathcal L_{\mathrm{final}}=\|\hat z_{t+1}^{(4)}-z^*_{t+1}\|_2^2\).
  The target is not detached, preserving LeWM's end-to-end encoder/predictor
  objective.
- **Intermediate-depth prediction.**
  \(\mathcal L_{\mathrm{inter}}\) averages the prediction MSE at depths
  \(1,2,3\) against \(\operatorname{sg}(z^*_{t+1})\). This makes shallow exits
  useful while preventing every intermediate loss from directly moving the
  target encoder.
- **Continue supervision.** Let
  \(e_k=\|\hat z_{t+1}^{(k)}-z^*_{t+1}\|_2^2\), computed under stop-gradient
  for label construction. The binary label is

  \[
  y_k=\mathbb I\left[
  \frac{e_k-e_{k+1}}{e_k+\epsilon}>5\times10^{-4}
  \right].
  \]

  Thus \(y_k=1\) means that one more recurrent step produces a sufficiently
  large **relative marginal reduction in raw target-latent MSE**. The linear
  continue head is trained with binary cross-entropy,
  \(\mathcal L_{\mathrm{cont}}=\operatorname{BCEWithLogits}(Wh_k+b,y_k)\),
  without class reweighting in the main run. Because this loss backpropagates
  through \(h_k\), the selector and recurrent predictor are learned jointly.
- **Representation regularization.** The original LeWM SIGReg term remains in
  the objective to regularize encoder representations.

The current selector therefore learns a deployable proxy for an MSE-derived
training signal. It does **not** observe future target MSE or run CEM comparisons
at test time.

## Main Results

The table reports success over three paired train/evaluation seed pairs
(`3072/42`, `3073/43`, and `3074/44`), with 50 evaluation episodes per pair.
Each success entry is mean \(\pm\) sample standard deviation across the three
seeds. LeWM is the original non-recurrent baseline; fixed \(K=1,2,3,4\) and
dynamic \(K\) are evaluated from the same RefineJEPA checkpoint family.

| Dataset | LeWM | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Learned dynamic K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reacher | 81.3\(\pm\)4.2 | 84.0\(\pm\)5.3 | 82.0\(\pm\)2.0 | 83.3\(\pm\)1.2 | 82.7\(\pm\)1.2 | **85.3\(\pm\)4.2 @ K1.03** |
| Cube Single | 72.0\(\pm\)12.0 | **79.3\(\pm\)8.1** | 78.7\(\pm\)7.6 | 78.0\(\pm\)8.7 | 78.0\(\pm\)8.7 | **79.3\(\pm\)8.1 @ K1.00** |
| Cube Double | **74.7\(\pm\)7.6** | 72.0\(\pm\)3.5 | 72.0\(\pm\)5.3 | 72.7\(\pm\)5.0 | 72.0\(\pm\)5.3 | 74.0\(\pm\)3.5 @ K1.26 |
| Cube Triple | 74.0\(\pm\)8.0 | 74.0\(\pm\)4.0 | 74.0\(\pm\)0.0 | 73.3\(\pm\)5.0 | 74.0\(\pm\)6.0 | **77.3\(\pm\)7.6 @ K1.22** |

![Main success comparison](analysis/readme_figures/main_success_vs_lewm.png)

The main readout is deliberately conditional:

- Reacher improves by 4.0 points over LeWM and 1.3 points over the best fixed
  RefineJEPA depth.
- Cube Triple improves by 3.3 points over both LeWM and every fixed depth.
- Cube Single is a negative/control case: fixed \(K=1\) is already strongest,
  and the learned selector almost always stops at \(K=1\).
- Cube Double improves over all fixed RefineJEPA depths but remains 0.7 points
  below the original LeWM baseline.

The seed-level variation is substantial, so these results support the current
method trend rather than a claim of statistical dominance on every task.

### Threshold-selection protocol

The displayed dynamic entries are the best observed success/mean-\(K\)
operating points from a test-time threshold sweep. The selected thresholds are
\(\eta=0.45\) for Reacher, \(0.70\) for Cube Single, \(0.45\) for Cube Double,
and \(0.30\) for Cube Triple. Because these thresholds were selected on the
evaluation sweep rather than a held-out validation set, the table should be
read as a **post-hoc Pareto envelope**, not as a final unbiased test estimate.
The final paper protocol should select \(\eta\) on held-out validation episodes
and evaluate it once on the test set.

## Where Does the Computation Go?

The learned policies allocate extra recurrent steps to only a subset of imagined
transitions.

| Dataset | Success / mean K | K=1 | K=2 | K=3 | K=4 | Refined beyond K1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reacher | 85.3% / 1.03 | 96.77% | 3.11% | 0.12% | 0.003% | 3.23% |
| Cube Single | 79.3% / 1.00 | 99.984% | 0.017% | 0.0001% | 0% | 0.017% |
| Cube Double | 74.0% / 1.26 | 76.49% | 21.09% | 1.69% | 0.72% | 23.51% |
| Cube Triple | 77.3% / 1.22 | 89.41% | 3.71% | 2.64% | 4.25% | 10.59% |

![Depth allocation](figures/depth_allocation_rel00005.png)

Mean \(K\) measures recurrent-cell evaluations per imagined transition. It is a
model-compute proxy, not yet a wall-clock latency measurement: batched CEM
execution can reduce or amplify the runtime effect of heterogeneous depths.
Measured success-latency and throughput curves remain necessary for the final
compute claim.

Extra refinement is also non-uniform within a rollout. Cube Triple uses it most
often near the first imagined transition, then progressively less later.

![Depth by rollout step](figures/depth_by_rollout_step_rel00005.png)

### Qualitative PushT trace

The following is one successful PushT H10/goal50 episode, included as a
qualitative trace rather than aggregate benchmark evidence. For
`global=1474257` at \(\eta=0.30\), mean \(K=1.287\), 21.8% of imagined
predictions are refined beyond \(K=1\), and the strongest band appears around
imagined rollout step \(+3\).

![PushT success trace with high-refinement decisions](figures/pusht_success_highk_env6_trace_panel.png)

![PushT success dynamic K preview](figures/pusht_success_highk_env6_annotated.gif)

MP4: [pusht_success_highk_env6_annotated.mp4](figures/pusht_success_highk_env6_annotated.mp4)

## Why Not Use Target MSE Directly at Test Time?

A 50-episode post-hoc diagnostic compares true target-latent errors at fixed
\(K=1\) and \(K=4\). It is not deployable because the future target latent is
unavailable during planning, and it is not directly comparable to the learned
head: the diagnostic makes one episode-level K1/K4 choice, whereas RefineJEPA
makes marginal K-to-K+1 decisions for every imagined transition.

| Dataset | Fixed K1 | Fixed K4 | Target-MSE diagnostic | Hindsight K1/K4 ceiling |
| --- | ---: | ---: | ---: | ---: |
| Reacher | 80%@K1.00 | 82%@K4.00 | 80%@K1.00--K2.20 | 86%@K1.18 |
| Cube Single | 84%@K1.00 | 82%@K4.00 | 84%@K1.00--K1.12 | 84%@K1.00 |
| Cube Double | 70%@K1.00 | 68%@K4.00 | 70%@K1.00--K2.14 | 70%@K1.00 |
| Cube Triple | 74%@K1.00 | 74%@K4.00 | 74%@K1.00--K1.84 | 76%@K1.06 |

![Raw target-MSE stopping Pareto](analysis/readme_figures/raw_mse_tolerance_pareto.png)

The diagnostic shows that raw target-latent error contains refinement signal,
but is not the planner's objective. CEM ultimately depends on candidate ranking
and selected actions. A lower latent MSE may leave the selected action unchanged,
while a small latent change near a decision boundary can alter the plan.

## Mechanistic Checks

We tested whether deeper recurrence simply causes broad latent smoothing or
collapse. The current global analyses do not support that explanation:

- the latent spectrum changes little from \(K=1\) to \(K=4\);
- linear state-probe quality is nearly unchanged;
- depth-helped and depth-hurt subsets do not show a clean global-collapse
  signature.

![Spectrum K1 versus K4](analysis/k_smoothing_20260622/figures/spectrum_k1_vs_k4_scatter.png)

![Probe R2 K1 versus K4](analysis/k_smoothing_20260622/figures/probe_r2_k1_vs_k4_scatter.png)

This points toward a more local, planner-facing mechanism: refinement matters
when it changes candidate ordering or action selection. Full Kendall-tau,
elite-overlap, selected-action, and wall-clock analyses are the highest-priority
remaining experiments.

## Scope and Experiment Records

This README contains the current paper-facing result. Longer tables, exploratory
training variants, planner-aware diagnostics, paths, and failed runs are kept in:

- [TTJEPA_EXPERIMENT_RESULTS.md](TTJEPA_EXPERIMENT_RESULTS.md)
- [TTJEPA_DYNAMIC_K_RESEARCH.md](TTJEPA_DYNAMIC_K_RESEARCH.md)
- [paper.md](paper.md)
- [LEWM_REPRODUCTION_NOTES.md](LEWM_REPRODUCTION_NOTES.md)

The active recurrent implementation currently lives on the
`codex/recurrent-lewm` development branch. Before an artifact release, it still
needs to be merged into `main` together with a pinned `uv` environment, tests,
checkpoint/data instructions, and exact train/eval commands. The repository is
therefore an active research release rather than a frozen reproducibility
artifact.

## Citation and Provenance

RefineJEPA builds on JEPA-style representation learning and LeWM-style latent
world-model planning. It modifies the LeWM transition predictor and inference
path to study dynamically allocated recurrent refinement depth \(K\). Please
cite the corresponding upstream works; a RefineJEPA BibTeX entry will be added
with the paper release.

The repository is distributed under the terms in [LICENSE](LICENSE).
