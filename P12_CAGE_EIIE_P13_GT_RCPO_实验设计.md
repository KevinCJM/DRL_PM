# P12 CAGE-EIIE 与 P13 GT-RCPO-lite 实验设计

生成日期：2026-05-24  
适用项目：`/Users/chenjunming/Desktop/DRL_PM`  
继承协议：`core13_v2_full_reset_20260522`  
模型扩展 ID：`core13_v2_p12_p13_20260524`  
文档状态：**新增模型实验设计 / runbook candidate**  
建议主线：**P12 CAGE-EIIE**  
建议探索线：**P13 GT-RCPO-lite**

---

## 0. 执行裁决

本轮新增实验不建议继续包装旧主模型 `full_dqn_gated_multitask_cnn_ppo` 为收益优势模型。已有结果已经说明：

```text
1. eiie_native 在主 HPO 表中累计收益均值更高。
2. ppo_dqn_hierarchical_reimplementation 在 main_hpo_plus_p9 合并范围中也高于旧主模型。
3. 旧主模型的可保留优势主要是换手、成本和 execution discipline。
```

因此新增实验的目标不是“继续强化 PPO + DQN gate”，而是重新设计 allocation 与 execution 的分工。

最终裁决：

```text
P12：CAGE-EIIE 作为新主线候选。
P13：GT-RCPO-lite 作为探索性 scout，不直接进入 full formal。
```

CAGE-EIIE 的核心假设：

```text
EIIE 负责 allocation alpha：买什么。
Gate / controller 负责 execution discipline：何时买、买多少。
Distributional risk critic 负责 downside-risk awareness：何时减弱执行。
```

GT-RCPO-lite 的核心假设：

```text
Graph / Transformer 能更好捕捉 Core-13 跨资产关系，
并通过 constrained actor-critic 同时控制成本、换手和尾部风险。
```

但是 GT-RCPO-lite 工程复杂度高、训练不稳定风险高、ablation 负担重。因此它只作为 P13 scout，除非在 validation-only pilot 中明显胜出，否则不进入 5-seed formal HPO。

---

## 1. 与既有 Core-13 v2 协议的关系

P12/P13 不重置 Core-13 v2 的数据、执行和估值协议，但它们引入新模型和新 HPO search space，因此必须作为 `core13_v2_full_reset_20260522` 之上的 registered model-extension addendum 单独登记。

```text
base_protocol_id = core13_v2_full_reset_20260522
model_extension_id = core13_v2_p12_p13_20260524
post_hoc_development_disclosure = true
test_used_for_model_selection = false
```

解释：

```text
1. 不重新下载数据，不改变 Core-13 asset universe、valuation/execution split、cost model 或 test split。
2. 不覆盖 P7/P9 既有正式主表。
3. P12/P13 的 search-space manifest、promotion gate、formal aggregation、artifact bundle 必须单独记录。
4. 论文中必须披露：P12/P13 是基于已完成 Core-13 v2 结果提出的 second-stage model extension。
```

必须继承以下规则：

```text
asset_universe_id = core13_v2
protocol_id = core13_v2_full_reset_20260522
data_cutoff_date = 2026-05-20
data_mode = availability_mask
return_source = adj_nav
valuation_source = adj_nav
reward_return_source = adj_nav
metrics_return_source = adj_nav
execution_price_source = ohlcv
valuation_execution_split = true
execution_price = next_open
rebalance base mode = monthly
model gate may only choose rho > 0 on scheduler-allowed rebalance dates
non-allowed rebalance dates must force rho = 0 and rebalance_action = 0
cost_model.proportional_cost = 0.001
cost_model.slippage = 0.0005
market_impact_enabled = true
market_impact_coef = 0.10
```

P12/P13 不能混入：

```text
data/processed/asset_universe.csv
configs/data/etf_lof_universe.yaml
data/metrics_factory/all_metrics_features.parquet
旧 17 资产池结果
旧 Core-13 非 reset 结果
旧 HPO trial / checkpoint
proxy / external original 结果
```

---

## 2. 当前新增实验的核心问题

已有正式结果显示，旧主模型不是累计收益第一。因此新增实验要回答下面三个问题。

### Q1. CAGE-EIIE 是否能改善 EIIE 的收益-成本权衡？

检验对象：

```text
cage_eiie_frozen_gate
cage_eiie_multilevel_gate
cage_eiie_distributional
```

核心比较：

```text
eiie_native
pgportfolio_eiie_native
full_dqn_gated_multitask_cnn_ppo
ppo_dqn_hierarchical_reimplementation
cnn_ppo_native
```

主要判据：

```text
在不显著损失累计收益的前提下，降低 turnover / transaction cost / downside risk。
若累计收益也超过 eiie_native，则可以成为新的收益主张候选。
```

### Q2. GT-RCPO-lite 是否值得进入正式大实验？

检验对象：

```text
graph_transformer_risk_constrained_actor_critic_lite
```

核心比较：

```text
cage_eiie_distributional
eiie_native
ppo_dqn_hierarchical_reimplementation
full_dqn_gated_multitask_cnn_ppo
```

主要判据：

```text
必须在 validation-only pilot 上同时体现收益、风险和成本优势，
否则只保留为 appendix diagnostic / future work。
```

### Q3. 新模型能否形成可信论文贡献，而不是 test-set chasing？

控制方式：

```text
1. 所有新模型先走 smoke。
2. pilot 只用 validation selection，不用 final test ranking 选模型。
3. formal evaluation 只对晋级模型执行。
4. smoke / pilot 必须保留 run_manifest 或 sidecar manifest、daily artifacts 和 validation report。
5. formal P12/P13/P14 必须保留 search-space manifest、run_manifest、paired statistics 和 artifact lineage。
```

---

## 3. 新实验阶段总览

| 阶段 | 名称 | 目标 | rankability | 是否进入主论文表 |
|---|---|---|---|---|
| P12.0 | CAGE implementation gate | 检查代码、配置、manifest、数据口径 | no | no |
| P12.1 | CAGE smoke | 单 seed 单 trial，验证能跑通 | diagnostic | no |
| P12.2 | CAGE validation pilot | seed=42，5-trial HPO，validation-only selection | diagnostic / promotion candidate | no |
| P12.3 | CAGE ablation pilot | 多档 gate、CVaR、drawdown、joint/frozen ablation | diagnostic | appendix / method table |
| P12.4 | CAGE formal HPO | 5-seed equal-budget HPO，默认 50 trials/model；若少于 P7/P9 formal budget 只能 appendix diagnostic | formal | yes |
| P12.5 | CAGE formal aggregation | 主表、paired stats、risk-cost 表 | formal | yes |
| P13.1 | GT-RCPO-lite smoke | 单 seed 单 trial，验证稳定性 | diagnostic | no |
| P13.2 | GT-RCPO-lite validation pilot | seed=42，5-trial HPO，判断是否晋级 | diagnostic / promotion candidate | no |
| P13.3 | GT-RCPO-lite formal scout | 仅在 pilot 明显胜出时执行；预算必须与 P12/P7/P9 formal 可比 | formal candidate | conditional |
| P14 | New-model final comparison | CAGE/P13 晋级模型与现有强 baseline 合并比较 | formal | yes |

---

## 4. P12 主线模型：CAGE-EIIE

### 4.1 模型名称

代码名：

```text
cage_eiie_distributional
```

论文名：

```text
CAGE-EIIE: Cost-Aware Gated EIIE with Distributional Rebalance Control
```

### 4.2 模型思想

EIIE 产生候选组合权重：

```math
\tilde{w}_{t}^{\mathrm{EIIE}} = f_{\theta}^{\mathrm{EIIE}}(s_t)
```

Gate 产生调仓执行强度：

```math
\rho_t \in \{0, 0.25, 0.50, 0.75, 1.00\}
```

最终目标权重：

```math
w_t^{\mathrm{target}}
=
(1-\rho_t)w_{t^-}
+
\rho_t\tilde{w}_{t}^{\mathrm{EIIE}}
```

其中：

```text
w_{t^-}：执行前组合权重，考虑价格漂移后的 current weights。
\tilde{w}_{t}^{EIIE}：EIIE candidate weights。
\rho_t：调仓强度，由 cost-aware gate 决定。
```

### 4.3 设计动机

CAGE-EIIE 明确解耦两件事：

```text
allocation alpha：由 EIIE/PVM 提供。
execution discipline：由 gate / controller 提供。
```

现有结果支持这个方向：

```text
EIIE native 收益强，但可能换手和成本较高。
旧主模型成本控制较好，但 PPO candidate allocation 不够强。
CAGE-EIIE 用 EIIE 替换 PPO actor 的弱项，同时保留多档 gate 的成本控制能力。
```

---

## 5. CAGE-EIIE 模型结构

### 5.1 Allocation expert

模块：

```text
EIIE / PVM allocation expert
```

输入：

```text
Core-13 feature tensor
availability_mask
previous portfolio vector
optional cash / residual position, if existing system supports cash asset
```

输出：

```text
eiie_candidate_weights: shape = [n_assets]
```

约束：

```text
1. long-only。
2. simplex weights。
3. unavailable asset weight = 0。
4. max_weight 可选，不得破坏与 baseline 的公平比较。
```

### 5.2 Execution gate

Gate 动作空间：

```text
0: hold
1: rebalance_25
2: rebalance_50
3: rebalance_75
4: rebalance_100
```

对应：

```text
rho = [0.00, 0.25, 0.50, 0.75, 1.00]
```

Scheduler 规则：

```text
非 scheduler-allowed rebalance date 强制 rho=0、rebalance_action=0。
这些日期可写入 replay / diagnostics，但必须标记 forced_hold_reason=scheduler_blocked；不得作为模型自由选择的 rebalance action 学习样本。
正式 gate action 分析必须区分 model_chosen_hold 与 scheduler_forced_hold。
```

推荐实现：

```text
Dueling Double DQN gate
或 categorical DQN gate
```

不建议第一版直接使用复杂 hierarchical option-critic，因为当前目标是验证 CAGE 假设，而不是引入过多训练不稳定因素。

### 5.3 Distributional risk critic

建议实现两档：

```text
v1: ordinary value critic + rolling downside features
v2: quantile critic / distributional critic
```

Distributional critic 输出：

```text
quantile_values: [n_quantiles]
expected_net_return
CVaR_5_estimate
CVaR_10_estimate
drawdown_risk_score
```

CVaR 估计：

```math
\mathrm{CVaR}_{\alpha}(R)
=
\mathbb{E}[R \mid R \leq \mathrm{VaR}_{\alpha}(R)]
```

如果使用 quantile critic，近似为：

```math
\widehat{\mathrm{CVaR}}_{\alpha}
=
\frac{1}{K_{\alpha}}
\sum_{k: \tau_k \leq \alpha} Z_{\tau_k}(s,a)
```

### 5.4 Cost-aware controller features

Gate 的状态向量必须包含至少以下信息：

```text
decision_time_current_weights
eiie_candidate_weights
delta_weights = eiie_candidate_weights - decision_time_current_weights
estimated_turnover
estimated_transaction_cost
estimated_market_impact
rolling_volatility
rolling_correlation_risk
recent_drawdown
recent_net_return_mean
recent_net_return_std
recent_net_return_skew, optional
liquidity_features: amount / volume / turnover_rate
availability_mask
number_of_available_assets
calendar / rebalance eligibility flag
```

Leakage guard：

```text
1. Gate observation 不得包含真实 next-open / execution-time price drift。
2. actual pre-execution drifted weights 只能由 PortfolioExecutionCore 在执行时计算，不得进入模型输入。
3. rolling_volatility、rolling_correlation_risk、recent_return_distribution 只能使用 decision date 及以前的滚动窗口。
4. 不得使用全样本相关矩阵、test-window 统计量或 post-hoc regime label 作为训练、HPO 或 gate 输入。
```

### 5.5 最终执行权重

正式实现优先沿用现有 execution core 协议：

```text
model output:
  candidate_weights = EIIE candidate weights
  rebalance_intensity = rho
  gate_action_index in {0,1,2,3,4}

execution core:
  根据 rebalance_intensity 在 execution-time current weights 与 candidate_weights 之间插值。
```

禁止双重降档：

```text
如果模型已经输出 mixed target weights，则 rebalance_intensity 必须固定为 1.0，并记录 execution_weight_mode=pre_mixed。
正式 P12 推荐不使用 pre_mixed；推荐输出 EIIE candidate，由 execution core 执行 rho 插值。
```

概念上，execution core 执行的目标等价于：

```math
w_t^{executed}
=
\Pi_{\Delta(\mathcal{A}_t)}
\left((1-\rho_t)w_{t^-}^{exec}+\rho_t\tilde{w}_{t}^{EIIE}\right)
```

其中：

```text
\Pi：simplex projection。
\mathcal{A}_t：当前可交易 / 可持有资产集合。
w_{t^-}^{exec}：execution core 内部计算的执行前权重，不进入 decision-time observation。
```

硬约束：

```text
unavailable_asset_weight_abs_max = 0.0
sum(weights) = 1.0, unless cash is explicitly modeled
weights >= 0
finite weights
```

---

## 6. CAGE-EIIE 训练目标

### 6.1 避免重复扣成本

如果环境输出的 `net_log_return` 已经是 after-cost return，则 reward 不应再次扣完整 transaction cost。建议定义为：

```math
r_t
=
\log\left(\frac{V_{t+1}}{V_t}\right)
-
\lambda_{turnover}\cdot \tau_t
-
\lambda_{cvar}\cdot \max(0, \widehat{\mathrm{CVaR}}^{loss}_{\alpha,t}-cvar^{loss}_{budget})
-
\lambda_{dd}\cdot \Delta DD_t^+
```

其中：

```text
log(V_{t+1}/V_t)：after-cost portfolio log return。
τ_t：turnover。
CVaR 使用 loss 口径，\widehat{CVaR}^{loss}_{\alpha,t} >= 0；penalty 只在 loss 超过 budget 时触发。
ΔDD_t^+：drawdown 增量。
```

CVaR 符号约定：

```text
return-tail CVaR 通常为负数；训练和日志统一转换为 loss 正数。
CVaR_loss_5 = max(0, -CVaR_return_5)
CVaR_penalty = max(0, CVaR_loss_5 - cvar_loss_budget)
```

如果环境只输出 before-cost return，则可以使用：

```math
r_t
=
\log(1+r_t^{gross})
-
\lambda_{cost}\cdot cost_t
-
\lambda_{turnover}\cdot \tau_t
-
\lambda_{cvar}\cdot CVaRPenalty_t
-
\lambda_{dd}\cdot DrawdownPenalty_t
```

但正式口径必须与 Core-13 v2 的 after-cost daily returns 对齐。

### 6.2 Gate Q-learning target

Gate 的 action-value 可定义为：

```math
Q_{\phi}(s_t, \rho_t)
```

DQN target：

```math
y_t
=
r_t
+
\gamma Q_{\phi^-}(s_{t+1}, \arg\max_{\rho'}Q_{\phi}(s_{t+1},\rho'))
```

建议使用：

```text
Double DQN
Dueling head
target network
prioritized replay, optional
```

### 6.3 EIIE expert 训练方式

允许三种版本。

#### Version A: frozen candidate

```text
1. 每个 seed / fold 内训练 EIIE expert。
2. 冻结 EIIE 权重。
3. 只训练 gate。
```

预算归属：

```text
CAGE-EIIE 是一个完整 trainable model。
EIIE warm-up、frozen expert training、gate training、validation selection 的总 env_steps / gradient_updates / hpo_trials 必须计入同一个 per-model formal budget。
不得把“EIIE 训练预算 + 额外 gate 训练预算”与 eiie_native 直接同表比较，除非 baseline 也获得同等预算或表格标注为 unequal-budget diagnostic。
```

优点：

```text
假设最清晰。
最容易判断 gate 是否改善 EIIE 的 return-cost trade-off。
训练稳定。
```

缺点：

```text
上限受 frozen EIIE 限制。
```

#### Version B: joint-light

```text
1. EIIE expert 与 gate 联合训练。
2. EIIE learning rate 较低。
3. gate learning rate 较高。
4. 可以先 warm-up EIIE，再训练 gate。
```

优点：

```text
可能适应 gate 的执行强度。
```

缺点：

```text
归因较弱。
训练更不稳定。
```

#### Version C: distributional full

```text
1. EIIE expert。
2. Multi-level gate。
3. Quantile / CVaR critic。
4. Drawdown-aware penalty。
```

优点：

```text
最完整，论文贡献最强。
```

缺点：

```text
需要更多 ablation。
```

---

## 7. P12 模型变体

### 7.1 必跑变体

| 模型名 | 说明 | 目的 |
|---|---|---|
| `cage_eiie_frozen_gate` | 冻结 EIIE，只训练多档 gate | 验证 execution overlay 是否有效 |
| `cage_eiie_fixed_rho_25` | 固定执行 25% EIIE 目标 | 排除 DQN gate 是否只是弱化 EIIE |
| `cage_eiie_fixed_rho_50` | 固定执行 50% EIIE 目标 | 同上 |
| `cage_eiie_multilevel_gate` | DQN 选择 5 档执行强度 | 验证 adaptive gate |
| `cage_eiie_no_cvar` | 无 CVaR critic | 验证 risk critic 贡献 |
| `cage_eiie_distributional` | 完整版本 | 新主线候选 |

### 7.2 选跑变体

| 模型名 | 说明 | 使用条件 |
|---|---|---|
| `cage_eiie_joint_light` | EIIE + gate 联合训练 | frozen gate 成功后再跑 |
| `cage_eiie_binary_gate` | 仅 hold / full rebalance | 对照旧 binary gate |
| `cage_eiie_cost_only` | 只输入成本特征，无 CVaR/DD | 判断风险模块是否必要 |
| `cage_eiie_risk_only` | 输入风险特征，弱化成本特征 | 判断成本模块是否主导 |

---

## 8. P13 探索模型：GT-RCPO-lite

### 8.1 模型名称

代码名：

```text
graph_transformer_risk_constrained_actor_critic_lite
```

论文名：

```text
GT-RCPO-lite: Graph-Transformer Risk-Constrained Portfolio Optimization
```

### 8.2 为什么只做 lite

Full GT-RCPO 包含：

```text
Temporal Transformer
Graph Transformer
Distributional / CVaR critic
Constrained actor-critic
Lagrangian multipliers
Option gate
Risk-off mode
多种动态图边
```

这会导致：

```text
1. 参数量过大。
2. 训练稳定性风险高。
3. ablation 成本高。
4. Core-13 资产数量较小，Graph Transformer 的优势未必能充分体现。
```

因此 P13 只做 lite scout。

### 8.3 GT-RCPO-lite 结构

保留：

```text
Temporal Transformer encoder
rolling-correlation graph attention
simplex actor head
turnover-cost Lagrangian constraint
4-action option gate
basic CVaR critic, optional
```

Graph / correlation leakage guard：

```text
rolling-correlation graph attention 只能基于 decision date 及以前的滚动窗口构图。
不得使用全样本协方差/相关矩阵、test-period 统计量或 post-hoc regime labels。
每个 fold / seed 内的图统计必须在对应 train/validation 路径内独立生成并记录到 manifest。
```

暂不加入：

```text
复杂 cash/risk-off 子策略
多层 Graph Transformer
多个动态图边类型联合训练
完整 distributional actor-critic
复杂 CPO trust-region 更新
```

### 8.4 Actor 输出

```math
z_t = f_{\theta}^{GT}(s_t)
```

```math
w_t^{candidate} = \mathrm{softmax}(g_{\theta}(z_t))
```

或使用 simplex projection：

```math
w_t^{candidate}=\Pi_{\Delta(\mathcal{A}_t)}(g_{\theta}(z_t))
```

### 8.5 Option gate

动作空间：

```text
0: hold
1: conservative rebalance, rho = 0.25
2: normal rebalance, rho = 0.50
3: aggressive rebalance, rho = 1.00
4: defensive, optional, only if cash / risk-off asset is explicitly supported
```

如果系统没有 cash asset，第一版不启用 `defensive`，避免伪造无法执行的 risk-off 行为。

### 8.6 Lagrangian constraint

约束目标：

```math
\mathbb{E}[turnover_t] \leq \bar{\tau}
```

```math
\mathbb{E}[cost_t] \leq \bar{c}
```

Lagrangian reward：

```math
\mathcal{L}
=
\mathbb{E}[r_t]
-
\lambda_{\tau}(\mathbb{E}[\tau_t]-\bar{\tau})
-
\lambda_{c}(\mathbb{E}[c_t]-\bar{c})
```

Multiplier update：

```math
\lambda_{\tau} \leftarrow [\lambda_{\tau}+\eta_{\lambda}(\mathbb{E}[\tau_t]-\bar{\tau})]_+
```

```math
\lambda_{c} \leftarrow [\lambda_{c}+\eta_{\lambda}(\mathbb{E}[c_t]-\bar{c})]_+
```

---

## 9. 数据、时间切分和 leakage 防线

### 9.1 固定数据

P12/P13 使用与 Core-13 v2 完全一致的数据：

```text
data/processed/core13_wide_adj_nav_tushare.parquet
data/processed/core13_wide_log_return.parquet
data/processed/core13_wide_open.parquet
data/processed/core13_wide_close.parquet
data/processed/core13_wide_amount.parquet
data/processed/core13_wide_vol.parquet
data/processed/core13_wide_turnover_rate.parquet
data/metrics_factory/core13_all_metrics_features.parquet
```

### 9.2 时间切分

必须复用现有正式 train / validation / test split。若当前配置未显式导出 split manifest，P12/P13 前必须生成：

```text
data/reports/core13_split_manifest.json
```

至少记录：

```json
{
  "protocol_id": "core13_v2_full_reset_20260522",
  "train_start": "YYYY-MM-DD",
  "train_end": "YYYY-MM-DD",
  "validation_start": "YYYY-MM-DD",
  "validation_end": "YYYY-MM-DD",
  "test_start": "YYYY-MM-DD",
  "test_end": "YYYY-MM-DD",
  "selection_metric": "validation_return_cost_risk_utility",
  "test_used_for_model_selection": false
}
```

### 9.3 不允许的行为

```text
1. 不得用 final test ranking 调整 P12/P13 超参数。
2. 不得用 test period 训练 EIIE candidate。
3. 不得用 test period 训练 gate replay buffer。
4. 不得用 test period 选择 rho action bins。
5. 不得在看到 CAGE test 结果后再新增奖励项并继续声称同一 formal protocol。
```

### 9.4 允许的行为

```text
1. 使用既有 P7/P9 结果确定研究问题：EIIE allocation 强、旧主模型成本控制强。
2. 使用 train + validation 做新模型开发。
3. 使用 validation pilot 决定是否进入 formal。
4. formal test 只用于最终报告。
```

---

## 10. Metrics 体系

P12/P13 的主表不能只看累计收益。必须同时报告收益、风险、成本和稳定性。

### 10.1 收益指标

```text
cumulative_return
annualized_return
mean_daily_return
median_daily_return
positive_day_ratio
```

### 10.2 风险指标

```text
volatility
sharpe
sortino
max_drawdown
calmar
CVaR_5
CVaR_10
VaR_5
worst_5_day_return
```

注意：如果 risk metrics 仍出现 NaN，不能写风险调整收益改进结论。

### 10.3 成本与交易指标

```text
turnover_mean
turnover_median
turnover_p95
transaction_cost_total
transaction_cost_mean
market_impact_total
rebalance_count
hold_ratio
rho_mean
rho_distribution
return_per_turnover
return_per_cost
```

### 10.4 约束指标

```text
unavailable_asset_weight_abs_max
min_available_assets_per_date
daily_returns_finite
daily_nav_finite
weight_sum_abs_error_max
negative_weight_abs_max
max_single_asset_weight
constraint_violation_count
```

### 10.5 论文推荐主效用指标

为了避免只追逐 raw return，新增一个预注册 utility：

```math
U
=
\mathrm{CumRet}
-
\alpha_{mdd}\cdot |\mathrm{MDD}|
-
\alpha_{cvar}\cdot \mathrm{CVaR}^{loss}_{5}
-
\alpha_{turnover}\cdot \overline{Turnover}
-
\alpha_{cost}\cdot CostTotal
```

建议只用于 validation selection，不作为唯一论文排名依据。论文主表仍分列所有原始指标。

默认系数必须在 pilot config 或 `validation_selection_report.csv` 中固定记录；若未显式配置，使用：

```text
alpha_mdd = 0.25
alpha_cvar = 0.25
alpha_turnover = 2.0
alpha_cost = 10.0
```

这些系数只能用 train/validation 预注册，不得根据 final test ranking 调整。

Selection metric 口径：

```text
promotion_metric = validation_return_cost_risk_utility
formal_hpo_objective = validation_sharpe_minus_drawdown_turnover_penalty, default for P14 same-objective comparison
```

若 P12/P13 formal HPO 继续使用 `validation_return_cost_risk_utility` 作为 objective，则 P14 必须标注为 different-selection-objective extension；或者需要把 EIIE、旧主模型、P9 hierarchy 等强基准也用同一 utility 重新 HPO，才能称为同 objective 公平比较。

---

## 11. Promotion Gate：晋级规则

P12/P13 promotion gate 只能使用 validation split。P12 pilot 前必须生成同 split、同数据协议、同 validation metric 的 reference report：

```text
results/paper_tables/p12_p13_validation_references/
  validation_reference_comparison.csv
  validation_reference_daily_returns.csv
  validation_selection_report.csv
  validation_reference_manifest.json
```

最少包含：

```text
eiie_native
full_dqn_gated_multitask_cnn_ppo
ppo_dqn_hierarchical_reimplementation
cnn_ppo_native
pgportfolio_eiie_native
```

这些 reference 不得从既有 final-test aggregation 反推；必须来自 validation split 的 BacktestEngine / HPO validation outputs。

### 11.1 CAGE-EIIE 晋级 formal 的条件

P12 validation pilot 满足以下任一条件即可晋级。

#### 条件 A：收益直接胜出

```text
validation cumulative_return > eiie_native validation cumulative_return
```

#### 条件 B：Pareto 改善

```text
validation cumulative_return >= eiie_native_validation_cumulative_return - 0.01
且 turnover_mean <= 0.85 * eiie_native_validation_turnover_mean
且 transaction_cost_total <= 0.85 * eiie_native_validation_transaction_cost_total
且 max_drawdown <= eiie_native_validation_max_drawdown + 0.005
且 CVaR_loss_5 <= eiie_native_validation_CVaR_loss_5 + 0.005
```

#### 条件 C：相对旧主模型显著改善

```text
validation cumulative_return > full_dqn_gated_multitask_cnn_ppo
且 turnover_mean <= 1.05 * full_dqn_gated_multitask_cnn_ppo_validation_turnover_mean
且 transaction_cost_total <= 1.05 * full_dqn_gated_multitask_cnn_ppo_validation_transaction_cost_total
```

### 11.2 GT-RCPO-lite 晋级 formal 的条件

GT-RCPO-lite 工程成本更高，因此门槛更高：

```text
1. validation cumulative_return >= eiie_native_validation_cumulative_return。
2. validation_utility >= best_promoted_CAGE_validation_utility。
3. turnover_mean <= eiie_native_validation_turnover_mean。
4. transaction_cost_total <= eiie_native_validation_transaction_cost_total。
5. failed_trial_rate <= 0.20。
6. finite_artifact_rate == 1.0。
7. daily_returns / weights / NAV 全 finite。
```

若不满足，P13 只作为 diagnostic / future work。

### 11.3 失败处理

如果 P12/P13 都未晋级，不得把它们包装成新主模型。论文应保留当前结果，并将新方向作为 appendix diagnostic 或 future work。

---

## 12. 实验配置文件规划

### 12.1 P12 configs

必须新增：

```text
configs/paper/p12_cage_eiie_smoke.yaml
configs/paper/p12_cage_eiie_pilot.yaml
configs/paper/p12_cage_eiie_ablation.yaml
configs/paper/p12_cage_eiie_formal_seed_runner.yaml
configs/paper/p12_cage_eiie_formal_comparison.yaml
```

可选新增：

```text
configs/paper/p12_cage_eiie_joint_light_pilot.yaml
configs/paper/p12_cage_eiie_distributional_pilot.yaml
configs/paper/p12_cage_eiie_fixed_rho_ablation.yaml
```

### 12.2 P13 configs

必须新增：

```text
configs/paper/p13_gt_rcpo_lite_smoke.yaml
configs/paper/p13_gt_rcpo_lite_pilot.yaml
```

只有晋级后再新增：

```text
configs/paper/p13_gt_rcpo_lite_formal_seed_runner.yaml
configs/paper/p13_gt_rcpo_lite_formal_comparison.yaml
```

### 12.3 Config 必须包含的逻辑字段

如果当前 ConfigLoader 不支持这些字段，不能直接写入 YAML；必须通过支持的 schema 或 sidecar manifest 实现。

目标字段：

```yaml
new_model_protocol:
  phase: P12
  model_family: CAGE-EIIE
  protocol_id: core13_v2_full_reset_20260522
  model_extension_id: core13_v2_p12_p13_20260524
  post_hoc_development_disclosure: true
  data_mode: availability_mask
  selection_split: validation
  test_used_for_model_selection: false

rankability:
  # smoke / pilot:
  #   rankable_in_unified_table: false
  #   diagnostic_status: diagnostic
  # formal only:
  #   rankable_in_unified_table: true
  #   diagnostic_status: formal
  rankable_in_unified_table: phase_dependent
  diagnostic_status: phase_dependent

model:
  name: cage_eiie_distributional
  allocation_expert: eiie_pvm
  gate_type: multilevel_dqn
  rho_actions: [0.0, 0.25, 0.5, 0.75, 1.0]
  distributional_critic: true

paper_run_guard:
  require_core13_paths: true
  require_valuation_execution_split: true
  require_availability_mask_contract_if_mask_mode: true
  forbid_legacy_17_asset_paths: true
```

如果这些字段尚未被 `src/config.py` 接受，则必须写入：

```text
results/<run_name>/logs/new_model_sidecar_manifest.json
```

### 12.4 当前可执行性裁决

截至本文档审核时，以下 P12/P13 配置和模型注册属于 target-after-implementation：

```text
configs/paper/p12_cage_eiie_smoke.yaml
configs/paper/p12_cage_eiie_pilot.yaml
configs/paper/p12_cage_eiie_ablation.yaml
configs/paper/p12_cage_eiie_formal_seed_runner.yaml
configs/paper/p13_gt_rcpo_lite_smoke.yaml
configs/paper/p13_gt_rcpo_lite_pilot.yaml
model registry: cage_eiie_* / graph_transformer_risk_constrained_actor_critic_lite
```

在上述配置、模型注册、manifest writer 和 artifact validator 未实现前，本文件不是 current-runnable runbook，不能直接启动 P12/P13 formal。

---

## 13. Search space 设计

### 13.1 P12 CAGE search space

| 参数 | 类型 | 候选 / 范围 | 说明 |
|---|---|---|---|
| `gate_lr` | log-uniform | `[1e-5, 3e-4]` | Gate 学习率 |
| `eiie_lr` | log-uniform | `[1e-5, 1e-4]` | joint-light 才启用 |
| `gamma` | choice | `[0.95, 0.97, 0.99]` | DQN discount |
| `lambda_turnover` | log-uniform | `[1e-4, 1e-1]` | 换手惩罚 |
| `lambda_cvar` | log-uniform | `[1e-4, 1e-1]` | CVaR 惩罚 |
| `lambda_dd` | log-uniform | `[1e-4, 1e-1]` | Drawdown 惩罚 |
| `n_quantiles` | choice | `[16, 32, 51]` | distributional critic |
| `replay_size` | choice | `[10000, 50000]` | gate replay buffer |
| `target_update_freq` | choice | `[100, 500, 1000]` | DQN target update |
| `epsilon_final` | choice | `[0.01, 0.05, 0.10]` | exploration floor |
| `gate_hidden_dim` | choice | `[64, 128, 256]` | gate hidden size |

### 13.2 P13 GT-RCPO-lite search space

| 参数 | 类型 | 候选 / 范围 | 说明 |
|---|---|---|---|
| `transformer_d_model` | choice | `[32, 64, 128]` | 控制参数量 |
| `transformer_heads` | choice | `[2, 4]` | Core-13 不建议过大 |
| `transformer_layers` | choice | `[1, 2]` | lite 版本限制 |
| `graph_heads` | choice | `[1, 2, 4]` | graph attention |
| `actor_lr` | log-uniform | `[1e-5, 3e-4]` | actor 学习率 |
| `critic_lr` | log-uniform | `[1e-5, 3e-4]` | critic 学习率 |
| `lambda_lr` | log-uniform | `[1e-4, 1e-2]` | Lagrange multiplier 学习率 |
| `turnover_budget` | choice | `[0.05, 0.10, 0.15, 0.20]` | 平均换手约束 |
| `cost_budget` | choice | `[0.0005, 0.001, 0.002]` | 平均成本约束 |

### 13.3 Search-space manifest

每个 formal HPO 必须输出：

```text
results/<run_name>/logs/hpo_search_space_manifest.csv
```

字段：

```text
model_name
param_name
param_type
low
high
choices
log_scale
is_shared_across_models
is_model_specific
rationale
```

Formal budget：

```text
P12/P13 若要进入 P14 与 P7/P9 同表比较，必须使用与 P7/P9 formal 可比的 equal-budget：
  seeds = [42, 123, 2024, 3407, 9999]
  n_trials_per_model = 50
  selection_split = validation
  final_report_split = test

若因资源限制使用 20-30 trials/model，必须在表格中标注 lower-budget formal extension；不得与 P7/P9 50-trial results 直接做“同预算”主排名。
少于 20 trials/model 的结果只能作为 smoke / pilot / diagnostic。
```

---

## 14. 执行命令模板

说明：以下命令是 runbook target templates。若对应 config、model registry、manifest writer 或 CLI 尚未实现，则只能作为 target-after-implementation，不能直接冒充 current-runnable。

### 14.1 P12.1 CAGE smoke

```bash
.venv/bin/python -m src.experiments.run_experiment \
  --config configs/paper/p12_cage_eiie_smoke.yaml \
  --seed 42 \
  --run-name EXP27_P12_cage_eiie_smoke_s42
```

通过标准：

```text
logs/run_manifest.json 或 new_model_sidecar_manifest.json 存在。
metrics/daily_returns.csv 存在。
metrics/daily_weights.csv 存在。
metrics/gate_actions.csv 存在。
metrics/cage_eiie_candidate_weights.csv 存在。
net_return / nav / weights 全 finite。
unavailable_asset_weight_abs_max = 0.0。
rho_distribution 非空。
```

### 14.2 P12.2 CAGE pilot

```bash
.venv/bin/python -m src.experiments.run_experiment \
  --config configs/paper/p12_cage_eiie_pilot.yaml \
  --seed 42 \
  --run-name EXP28_P12_cage_eiie_pilot_s42
```

最低配置：

```text
n_trials_per_model = 5
seed = 42
promotion_metric = validation_return_cost_risk_utility
test_used_for_model_selection = false
```

### 14.3 P12.3 CAGE ablation pilot

```bash
.venv/bin/python -m src.experiments.run_experiment \
  --config configs/paper/p12_cage_eiie_ablation.yaml \
  --seed 42 \
  --run-name EXP29_P12_cage_eiie_ablation_s42
```

必须包含：

```text
cage_eiie_fixed_rho_25
cage_eiie_fixed_rho_50
cage_eiie_multilevel_gate
cage_eiie_no_cvar
cage_eiie_distributional
eiie_native
```

### 14.4 P12.4 CAGE formal seed-runner

只有 P12.2 通过 promotion gate 后才执行。

```bash
.venv/bin/python -m src.experiments.paper_seed_runner \
  --config configs/paper/p12_cage_eiie_formal_seed_runner.yaml \
  --seeds 42,123,2024,3407,9999 \
  --run-name-prefix EXP30_P12_formal_cage_eiie \
  --aggregate-output-dir results/paper_tables/p12_cage_eiie_formal
```

配置硬要求：

```text
hpo.n_trials_per_model = 50
hpo.selection_split = validation
hpo.final_report_split = test
rankability.diagnostic_status = formal
rankability.rankable_in_unified_table = true
protocol.model_extension_id = core13_v2_p12_p13_20260524
```

### 14.5 P13.1 GT-RCPO-lite smoke

```bash
.venv/bin/python -m src.experiments.run_experiment \
  --config configs/paper/p13_gt_rcpo_lite_smoke.yaml \
  --seed 42 \
  --run-name EXP31_P13_gt_rcpo_lite_smoke_s42
```

### 14.6 P13.2 GT-RCPO-lite pilot

```bash
.venv/bin/python -m src.experiments.run_experiment \
  --config configs/paper/p13_gt_rcpo_lite_pilot.yaml \
  --seed 42 \
  --run-name EXP32_P13_gt_rcpo_lite_pilot_s42
```

### 14.7 P13.3 GT-RCPO-lite formal seed-runner

只在 P13 promotion gate 通过后执行。

```bash
.venv/bin/python -m src.experiments.paper_seed_runner \
  --config configs/paper/p13_gt_rcpo_lite_formal_seed_runner.yaml \
  --seeds 42,123,2024,3407,9999 \
  --run-name-prefix EXP33_P13_formal_gt_rcpo_lite \
  --aggregate-output-dir results/paper_tables/p13_gt_rcpo_lite_formal
```

配置硬要求同 P12 formal；若 `n_trials_per_model < 50`，不得进入 P14 同预算主表。

---

## 15. Formal comparison 设计

### 15.1 P14 final comparison 候选模型

如果 P12 通过，P14 formal HPO comparison 至少包含通过 promotion gate 的 CAGE 变体，以及已有 formal HPO 强基准：

```text
cage_eiie_distributional, if promoted
cage_eiie_multilevel_gate, if promoted
cage_eiie_frozen_gate, if promoted
eiie_native
pgportfolio_eiie_native
cnn_ppo_native
full_dqn_gated_multitask_cnn_ppo
ppo_dqn_hierarchical_reimplementation
```

未晋级 CAGE 变体只能进入 ablation / diagnostic，不进入 P14 formal 主排名。

裸 `equal_weight` / `risk_parity` 属于 deterministic baseline；只有在加入正式 P1 baseline run dirs 且 manifest 满足 formal filter 时，才作为 deterministic reference 进入 P14。否则 P14 中的相关行应使用已有 P9 trainable child 名称，例如：

```text
hybrid_dqn_optimizer_equal_weight
hybrid_dqn_optimizer_risk_parity
```

如果 P13 也通过，额外加入：

```text
graph_transformer_risk_constrained_actor_critic_lite
```

### 15.2 聚合命令模板

```bash
.venv/bin/python -m src.experiments.paper_aggregate \
  --run-dir results/EXP30_P12_formal_cage_eiie_s42 \
  --run-dir results/EXP30_P12_formal_cage_eiie_s123 \
  --run-dir results/EXP30_P12_formal_cage_eiie_s2024 \
  --run-dir results/EXP30_P12_formal_cage_eiie_s3407 \
  --run-dir results/EXP30_P12_formal_cage_eiie_s9999 \
  --run-dir results/EXP05_P7_formal_hpo_main_native_s42 \
  --run-dir results/EXP05_P7_formal_hpo_main_native_s123 \
  --run-dir results/EXP05_P7_formal_hpo_main_native_s2024 \
  --run-dir results/EXP05_P7_formal_hpo_main_native_s3407 \
  --run-dir results/EXP05_P7_formal_hpo_main_native_s9999 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s42 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s123 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s2024 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s3407 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s9999 \
  --paper-group-id p14_new_model_final \
  --output-dir results/paper_tables/p14_new_model_final \
  --protocol-id core13_v2_full_reset_20260522 \
  --data-cutoff-date 2026-05-20 \
  --require-formal-manifest \
  --require-availability-mask-contract \
  --benchmark-model <promoted_primary_cage_model> \
  --benchmark-model eiie_native \
  --benchmark-model ppo_dqn_hierarchical_reimplementation \
  --benchmark-model full_dqn_gated_multitask_cnn_ppo
```

`<promoted_primary_cage_model>` 必须来自 promotion gate 结果，例如：

```text
cage_eiie_frozen_gate
cage_eiie_multilevel_gate
cage_eiie_distributional
```

如果没有任何 CAGE 变体晋级，则不得执行 P14 formal；只能输出 P12 diagnostic / negative-result appendix。

若 P13 formal 也通过，则追加：

```bash
  --run-dir results/EXP33_P13_formal_gt_rcpo_lite_s42 \
  --run-dir results/EXP33_P13_formal_gt_rcpo_lite_s123 \
  --run-dir results/EXP33_P13_formal_gt_rcpo_lite_s2024 \
  --run-dir results/EXP33_P13_formal_gt_rcpo_lite_s3407 \
  --run-dir results/EXP33_P13_formal_gt_rcpo_lite_s9999 \
  --benchmark-model graph_transformer_risk_constrained_actor_critic_lite
```

### 15.3 聚合过滤

所有进入 P14 的 run 必须满足：

```text
run_manifest.protocol_id == core13_v2_full_reset_20260522
data_governance.return_source == adj_nav
data_governance.valuation_source == adj_nav
data_governance.execution_price_source == ohlcv
execution_model.valuation_execution_split == true
rankability.rankable_in_unified_table == true
diagnostic_status == formal
test_used_for_model_selection == false
availability_mask_contract_passed == true
```

不满足者只能进入 diagnostic comparison。

P14 聚合必须优先读取 HPO per-model final outputs：

```text
metrics/hpo_model_final_comparison.csv
metrics/hpo_model_final_daily_returns.csv
metrics/hpo_model_final_daily_turnover.csv
metrics/hpo_model_final_daily_costs.csv
```

不得把同一 run 中的 best payload、baseline_comparison、main_comparison 和 hpo_model_final_comparison 重复纳入同一 paper_model_id。聚合器必须输出 dedup report；若 dedup report 缺失，P14 只能标为 aggregation_pending diagnostic。

---

## 16. 输出目录与产物

### 16.1 P12 smoke 输出

```text
results/EXP27_P12_cage_eiie_smoke_s42/
  logs/run_manifest.json
  logs/new_model_sidecar_manifest.json
  metrics/daily_returns.csv
  metrics/daily_weights.csv
  metrics/gate_actions.csv
  metrics/cage_eiie_candidate_weights.csv
  metrics/turnover_cost_breakdown.csv
  metrics/risk_metrics.csv
```

### 16.2 P12 pilot 输出

```text
results/EXP28_P12_cage_eiie_pilot_s42/
  logs/run_manifest.json
  logs/hpo_search_space_manifest.csv
  logs/hpo_trials.csv
  logs/validation_selection_report.csv
  metrics/hpo_model_final_comparison.csv
  metrics/hpo_model_final_daily_returns.csv
  metrics/gate_action_summary.csv
```

### 16.3 P12 formal 聚合输出

```text
results/paper_tables/p12_cage_eiie_formal/
  paper_main_comparison.csv
  paper_seed_summary.csv
  paper_paired_statistics.csv
  paper_cost_risk_decomposition.csv, pending aggregator extension
  paper_gate_action_summary.csv, pending aggregator extension
  source_run_dirs.txt
  diagnostic_status.json
```

### 16.4 P14 final 输出

```text
results/paper_tables/p14_new_model_final/
  paper_main_comparison.csv
  paper_seed_summary.csv
  paper_paired_statistics.csv
  paper_cost_risk_decomposition.csv, pending aggregator extension
  paper_new_model_ablation.csv, pending aggregator extension
  paper_pareto_frontier.csv, pending aggregator extension
  source_run_dirs.txt
  diagnostic_status.json
```

说明：当前 `paper_aggregate` 核心产物是 `paper_main_comparison.csv`、`paper_seed_summary.csv`、`paper_paired_statistics.csv` 和诊断表。上述 cost-risk、gate-action、Pareto 产物必须在 P12.0 implementation gate 中实现；未实现前不能作为 P12/P14 formal 完成条件。

---

## 17. Paired statistics 设计

P14 至少做以下配对检验：

```text
cage_eiie_distributional vs eiie_native
cage_eiie_distributional vs pgportfolio_eiie_native
cage_eiie_distributional vs cnn_ppo_native
cage_eiie_distributional vs full_dqn_gated_multitask_cnn_ppo
cage_eiie_distributional vs ppo_dqn_hierarchical_reimplementation
```

如果 P13 晋级：

```text
graph_transformer_risk_constrained_actor_critic_lite vs cage_eiie_distributional
graph_transformer_risk_constrained_actor_critic_lite vs eiie_native
graph_transformer_risk_constrained_actor_critic_lite vs ppo_dqn_hierarchical_reimplementation
```

建议检验对象：

```text
daily net_return
seed-level cumulative_return
seed-level turnover
seed-level transaction_cost
seed-level max_drawdown
seed-level CVaR_5
```

建议统计方法：

```text
paired t-test, if normality acceptable
Wilcoxon signed-rank test
bootstrap confidence interval
multiple-testing correction, at least Holm-Bonferroni
```

---

## 18. 成功、失败和论文主张

### 18.1 强成功

CAGE-EIIE formal 结果满足：

```text
5-seed mean cumulative_return >= eiie_native
且 turnover / transaction_cost <= eiie_native
且 max_drawdown / CVaR_5 不劣于 eiie_native
```

可写论文主张：

```text
CAGE-EIIE improves cumulative return while reducing execution frictions relative to EIIE under the Core-13 v2 formal protocol.
```

中文：

```text
CAGE-EIIE 在 Core-13 v2 正式协议下，相比 EIIE 同时改善累计收益和执行摩擦。
```

### 18.2 中等成功

CAGE-EIIE 满足：

```text
cumulative_return 略低于 eiie_native，但差距不超过 1%-2%，
turnover / cost 明显下降，
risk-adjusted metrics 改善。
```

可写论文主张：

```text
CAGE-EIIE improves the return-cost-risk trade-off of EIIE through distributional execution control.
```

中文：

```text
CAGE-EIIE 通过分布式风险感知的执行控制，改善了 EIIE 的收益-成本-风险权衡。
```

### 18.3 弱成功

CAGE-EIIE 只超过旧主模型，但不能超过 EIIE 或 P9 hierarchy。

可写：

```text
CAGE-EIIE validates the benefit of replacing PPO allocation with EIIE candidates, but does not dominate the strongest EIIE baseline.
```

不能写：

```text
CAGE-EIIE is the best-performing model.
```

### 18.4 失败

如果 CAGE-EIIE 累计收益、风险和成本均不占优，应停止 formal 扩展。

论文处理：

```text
只作为 negative result / appendix diagnostic。
保留原论文主张：严格实验协议 + 旧主模型成本控制 + EIIE 收益优势。
```

---

## 19. Implementation checklist

### 19.1 Model registry

需要新增：

```text
cage_eiie_frozen_gate
cage_eiie_fixed_rho_25
cage_eiie_fixed_rho_50
cage_eiie_multilevel_gate
cage_eiie_no_cvar
cage_eiie_distributional
cage_eiie_joint_light
graph_transformer_risk_constrained_actor_critic_lite
```

### 19.2 Environment / execution core

必须确认：

```text
1. final target weights 可以由 model 输出。
2. model observation 使用 decision_time_current_weights；pre_execution_drifted_weights 只允许 execution core 内部计算和事后日志使用。
3. estimated_turnover 与实际 turnover 口径一致。
4. transaction cost 和 market impact 只在 execution core 中扣除一次。
5. adj_nav valuation 与 OHLCV execution split 不被破坏。
6. availability mask 在 candidate weights 和 final weights 上都生效。
```

### 19.3 Logging

必须新增：

```text
metrics/gate_actions.csv
metrics/gate_action_summary.csv
metrics/cage_eiie_candidate_weights.csv
metrics/cage_final_weights.csv
metrics/turnover_cost_breakdown.csv
metrics/risk_metrics.csv
logs/validation_selection_report.csv
logs/new_model_sidecar_manifest.json
```

多档 gate 日志字段：

```text
date
decision_date
execution_date
model_name
seed
gate_action_index
gate_action_binary
rebalance_intensity
rho
rebalance_values
candidate_turnover_estimate
candidate_cost_estimate
realized_turnover
realized_cost
scheduler_allowed_rebalance
forced_hold_reason
execution_weight_mode
```

说明：`gate_action_binary` 只表示 hold / execute；五档动作必须以 `gate_action_index` 和 `rebalance_intensity/rho` 为准。

### 19.4 Aggregator

必须支持：

```text
rho_distribution aggregation
hold_ratio aggregation
return_per_turnover
return_per_cost
CVaR_5 / CVaR_10
max_drawdown non-NaN check
formal / diagnostic filtering
source_run_dirs.txt
```

### 19.5 Audit

必须新增：

```text
scripts/validate_cage_eiie_artifacts.py
scripts/validate_new_model_formal_readiness.py
```

校验内容：

```text
candidate weights finite
final weights finite
rho action valid
unavailable asset weight = 0
daily_returns finite
nav finite
no duplicate cost deduction
sidecar manifest complete
validation selection does not use test
```

---

## 20. 新增 run ledger 字段

P12/P13 必须扩展 ledger：

```text
phase
run_name
model_name
model_family
config_path
run_dir
seed
hpo_trial_count
selection_split
selection_metric
test_used_for_model_selection
protocol_id
model_extension_id
post_hoc_development_disclosure
data_mode
rankable_in_unified_table
diagnostic_status
availability_mask_contract_passed
valuation_execution_split
search_space_manifest_path
sidecar_manifest_path
artifact_paths
promotion_gate_status
blocking_reason
```

Formal readiness audit 也必须扩展。当前最终门禁若只覆盖 `main_hpo_5seed`、`main_hpo_plus_p9`、`p9_related_work_hpo`，则 P12/P14 完成后仍不能自动证明新 formal 已纳入最终审计。P12.0 必须增加或更新：

```text
src/experiments/formal_readiness.py
results/full_reproduction/core13_v2_full_reset_20260522/formal_readiness_audit.csv
```

使其覆盖：

```text
results/paper_tables/p12_cage_eiie_formal
results/paper_tables/p13_gt_rcpo_lite_formal, if promoted
results/paper_tables/p14_new_model_final
```

---

## 21. 推荐执行顺序

```text
Step 0: 修复或确认 risk metrics 聚合，确保 Sharpe/MDD/CVaR 不再系统性 NaN。
Step 1: 实现 CAGE-EIIE frozen gate。
Step 2: 跑 P12.1 smoke。
Step 3: 跑 P12.2 validation pilot。
Step 4: 跑 P12.3 ablation pilot。
Step 5: 判断 CAGE 是否晋级 formal。
Step 6: 同步实现 GT-RCPO-lite smoke，但不阻塞 P12。
Step 7: 跑 P13.1/P13.2 scout。
Step 8: 只有 P13 明显胜出时才进入 formal。
Step 9: 跑 P12 formal 5-seed HPO。
Step 10: 跑 P14 final aggregation。
Step 11: 生成新论文表、图、artifact bundle 和 final audit。
```

---

## 22. 最小可执行版本

如果工程资源有限，只做下面这个最小闭环：

```text
1. cage_eiie_frozen_gate
2. cage_eiie_multilevel_gate
3. cage_eiie_distributional
4. eiie_native baseline
5. full_dqn_gated_multitask_cnn_ppo baseline
6. ppo_dqn_hierarchical_reimplementation baseline
```

最小实验：

```text
seed = 42
n_trials_per_model = 5
selection_split = validation
metrics = cumulative_return, turnover, cost, max_drawdown, CVaR_5
```

如果最小版本不能超过或接近 EIIE，就不要扩大 P12 formal。

---

## 23. 论文表格规划

### Table A2: New-model formal comparison

字段：

```text
model
mean_cumulative_return
std_cumulative_return
mean_annualized_return
mean_sharpe
mean_sortino
mean_max_drawdown
mean_CVaR_5
mean_turnover
mean_transaction_cost
mean_return_per_cost
rank_by_cumulative_return
rank_by_utility
```

### Table B2: CAGE ablation

模型：

```text
eiie_native
cage_eiie_fixed_rho_25
cage_eiie_fixed_rho_50
cage_eiie_binary_gate
cage_eiie_multilevel_gate
cage_eiie_no_cvar
cage_eiie_distributional
```

### Table C2: Gate action analysis

字段：

```text
model
hold_ratio
rebalance_25_ratio
rebalance_50_ratio
rebalance_75_ratio
rebalance_100_ratio
mean_rho
rho_std
mean_turnover_when_rebalance
mean_return_after_rebalance
```

### Figure F2: Return-cost Pareto frontier

横轴：

```text
transaction_cost_total 或 turnover_mean
```

纵轴：

```text
cumulative_return 或 annualized_return
```

点：

```text
models
```

### Figure F3: Drawdown and NAV path

内容：

```text
NAV curve
underwater curve
drawdown periods
```

### Figure F4: Gate intensity over time

内容：

```text
rho_t time series
market drawdown overlay
transaction cost spikes
```

---

## 24. 不建议做的路线

```text
1. 不建议只扩大旧主模型 HPO。
2. 不建议只把 PPO 换成 SAC 后声称新算法。
3. 不建议只加 Transformer encoder 后声称主创新。
4. 不建议用 P13 full GT-RCPO 直接替代 P12。
5. 不建议把 P12 pilot 的 test 结果作为 promotion gate。
6. 不建议用旧 P7/P9 checkpoint 直接作为 CAGE formal 的 EIIE expert，除非能证明其训练未使用 test 并且 lineage 完整。
```

---

## 25. 最终裁决

本文件的实验设计结论：

```text
CAGE-EIIE 是下一轮最适合推进的主线实验。
GT-RCPO-lite 可以并行做 scout，但不能拖慢 CAGE-EIIE。
只有通过 validation-only promotion gate 的模型，才允许进入 5-seed equal-budget formal HPO。
```

推荐执行优先级：

```text
1. cage_eiie_frozen_gate
2. cage_eiie_multilevel_gate
3. cage_eiie_distributional
4. cage_eiie_joint_light
5. graph_transformer_risk_constrained_actor_critic_lite
```

论文最终主张必须由 formal 结果决定，而不是由模型名称或结构复杂度决定。
