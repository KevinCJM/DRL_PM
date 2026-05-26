# P16 Risk-Aware Graph-Transformer Constrained Actor-Critic 实验设计

生成日期：2026-05-25  
适用项目：`/Users/chenjunming/Desktop/DRL_PM`  
继承基础协议：`core13_v2_full_reset_20260522`  
建议扩展编号：`core13_v2_p16_ra_gt_rcpo_20260525`  
文档状态：**target-after-implementation / code + experiment runbook**  

---

## 0. 执行结论

本实验不是继续放大已经失败的 P13 scout，而是新增一个完整模型扩展：

```text
P16: Risk-Aware Graph-Transformer Constrained Actor-Critic
```

当前项目中已有 `graph_transformer_risk_constrained_actor_critic_lite`，但它只是 platform-adapted lite 版本：

```text
1. 使用 decision-visible rolling correlation / momentum / volatility 生成候选权重。
2. 使用规则化 rho scorer 控制调仓强度。
3. fit() 只写 training_result，不是真正 graph-transformer actor-critic 梯度训练。
4. P13 validation pilot 未通过 promotion gate。
```

因此 P16 必须作为新模型完整实现与新实验，不能把已有 P13 lite 结果包装成完整 RA-GT-RCPO。

当前实验目标：

```text
验证 graph/transformer representation + constrained actor-critic
是否能在 Core-13 v2 上改善 return-risk-cost trade-off。
```

---

## 1. 与现有实验的关系

P16 继承 Core-13 v2 的数据、估值、执行、成本、约束和评估协议，不重置数据协议：

```text
protocol_id = core13_v2_full_reset_20260522
asset_universe_id = core13_v2
data_cutoff_date = 2026-05-20
data_mode = availability_mask
valuation_source = adj_nav
return_source = adj_nav
reward_return_source = adj_nav
metrics_return_source = adj_nav
execution_price_source = ohlcv
valuation_execution_split = true
reward_valuation_split = true
execution_protocol = platform_backtest_engine
```

但 P16 改变模型族和 search space，因此必须作为 model-extension addendum 单独登记：

```text
model_extension_id = core13_v2_p16_ra_gt_rcpo_20260525
post_hoc_development_disclosure = true
test_used_for_model_selection = false
```

P16 不得复用：

```text
旧 17 资产池结果
旧 Core-13 非 reset 结果
P13 lite smoke/pilot checkpoint
P12 CAGE checkpoint
任何 test-ranking 反推的超参数
```

P16 固定新增 artifact group：

```text
results/paper_tables/p16_validation_references
results/paper_tables/p16_promotion_gate
results/paper_tables/p16_ra_gt_rcpo_final
```

这些目录必须独立于 `main_hpo_5seed`、`main_hpo_plus_p9`、`p12_p13_promotion_gate`、`p14_new_model_final`，不得覆盖既有论文表格。

---

## 2. 当前代码与已完成实验

### 2.1 已有代码

当前已有 lite 实现：

```text
src/baselines/gt_rcpo_lite.py
```

证据：

`{ "kind": "file", "ref": "src/baselines/gt_rcpo_lite.py" }`

当前 lite 版本特征：

```text
strategy_name = graph_transformer_risk_constrained_actor_critic_lite
fit_required = true
requires_daily_diagnostics = true
candidate = momentum / volatility - correlation_penalty
rho = choose_rho(expected_return, turnover, cost, CVaR_loss, drawdown)
training_algorithm = graph_transformer_risk_constrained_actor_critic_lite
gradient_updates = 0
```

这说明它不是完整 neural graph-transformer actor-critic，而是一个可运行、可诊断的平台适配近似模型。

### 2.2 已有配置

当前已有 P13 lite 配置：

```text
configs/paper/p13_gt_rcpo_lite_smoke.yaml
configs/paper/p13_gt_rcpo_lite_pilot.yaml
configs/paper/p13_gt_rcpo_lite_formal_seed_runner.yaml
configs/paper/p13_gt_rcpo_lite_formal_comparison.yaml
```

证据：

`{ "kind": "file", "ref": "configs/paper/p13_gt_rcpo_lite_pilot.yaml" }`  
`{ "kind": "file", "ref": "configs/paper/p13_gt_rcpo_lite_formal_seed_runner.yaml" }`

这些配置可作为 P16 配置模板，但不能直接作为完整 P16 formal 配置。

### 2.3 已有实验结果

P13 lite smoke：

```text
run = results/EXP31_P13_gt_rcpo_lite_smoke_s42
model = graph_transformer_risk_constrained_actor_critic_lite
cumulative_return = 0.326710
average_turnover = 0.001111
total_transaction_cost = 0.001056
rankable_in_unified_table = false
```

P13 lite validation pilot：

```text
run = results/EXP32_P13_gt_rcpo_lite_pilot_s42
best_trial_number = 0
best_value = 0.806863
test cumulative_return = 0.326710
max_drawdown_loss = 0.071318
CVaR_loss_5 = 0.013405
```

Promotion gate：

```text
P13 cumulative_return_validation = 0.094365
P13 validation utility = 0.062539
best promoted CAGE validation utility = 0.073823
P13 gate result = failed
blocking_reason = P13 validation promotion conditions not met
```

证据：

`{ "kind": "file", "ref": "results/EXP31_P13_gt_rcpo_lite_smoke_s42/metrics/baseline_comparison.csv" }`  
`{ "kind": "file", "ref": "results/EXP32_P13_gt_rcpo_lite_pilot_s42/metrics/hpo_model_final_comparison.csv" }`  
`{ "kind": "file", "ref": "results/EXP32_P13_gt_rcpo_lite_pilot_s42/metrics/hpo_model_final_risk_metrics.csv" }`  
`{ "kind": "file", "ref": "results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv" }`

结论：

```text
P13 lite 未证明 graph/transformer/risk-constrained 方向在当前实现下有效。
P16 若继续做，必须升级为真正可训练模型，而不是调 lite scorer。
```

---

## 3. 研究问题

### RQ1. Graph/Transformer 表征是否优于现有 EIIE / CNN-PPO 表征？

对照：

```text
cage_eiie_joint_light
eiie_native
pgportfolio_eiie_native
cnn_ppo_native
full_dqn_gated_multitask_cnn_ppo
```

### RQ2. Risk-aware constrained actor-critic 是否改善风险指标？

重点指标：

```text
max_drawdown_loss
CVaR_loss_5
volatility
Sharpe
Sortino
Calmar
```

### RQ3. 复杂模型是否值得替代 CAGE-EIIE 的低频执行 overlay？

判据：

```text
如果 P16 收益不高于 CAGE，但 MDD/CVaR 显著更低，可作为 risk-controlled alternative。
如果收益、Sharpe、MDD、CVaR、成本均不占优，则只进入 appendix diagnostic / future work。
```

---

## 4. P16 模型定义

### 4.1 正式模型名

代码名：

```text
risk_aware_graph_transformer_constrained_actor_critic
```

论文名：

```text
Risk-Aware Graph-Transformer Constrained Actor-Critic
```

建议别名：

```text
ra_gt_rcpo
```

### 4.2 输入分层

Actor observation 只能使用 decision-date 可见信息，且不得包含 actor 输出后才能计算的 candidate-dependent 字段：

```text
market_image_t
decision_time_current_weights_t
availability_mask_t
rolling_return_window_t
rolling_volatility_t
rolling_correlation_matrix_t
liquidity_features_t
current_drawdown_t
recent_net_return_distribution_features_t
```

Gate / critic / constraint heads 可以在 actor 生成 `candidate_weights_t` 后使用以下派生字段：

```text
candidate_weights_t
candidate_delta_weights_t = candidate_weights_t - decision_time_current_weights_t
estimated_turnover(candidate, decision_time_current_weights)
estimated_cost(candidate, decision_time_current_weights)
```

该分层是硬约束：`estimated_turnover(candidate,current)` 和 `estimated_cost(candidate,current)` 不得进入 actor encoder / actor logits 输入，否则会产生循环依赖。

禁止进入 observation / actor / critic：

```text
future returns
test-window statistics
execution-date-only fields
post-hoc regime label
pre_execution_drifted_weights
realized_cost
realized_turnover
```

`pre_execution_drifted_weights` 只允许在 execution core 内部计算，或作为事后日志。

### 4.3 Graph Encoder

图节点：

```text
Core-13 assets
```

边权：

```text
rolling correlation / partial correlation / sector prior
```

首版只允许：

```text
decision_visible_rolling_correlation
```

不得使用全样本相关矩阵。

### 4.4 Temporal Transformer Encoder

输入：

```text
[window, asset, feature]
```

输出：

```text
asset_embedding_i
portfolio_context_embedding
```

### 4.5 Actor

Actor 输出候选权重：

```text
candidate_weights = masked_softmax(actor_logits, availability_mask)
```

执行动作仍走平台执行核心：

```text
PortfolioAction(
  target_weights = candidate_weights,
  rebalance_action = 1 if scheduler_allowed and rho > 0 else 0,
  rebalance_intensity = rho
)
```

禁止模型端先做：

```text
(1-rho) * current + rho * candidate
```

否则会与 execution core 的 partial rebalance 插值重复。

### 4.6 Critic

至少包含：

```text
V_return(s)
V_cost(s)
V_drawdown(s)
V_cvar_loss(s)
```

可选：

```text
distributional_quantile_critic
```

CVaR 统一采用 loss 口径：

```text
CVaR_loss_5 >= 0
penalty_cvar = max(0, CVaR_loss_5 - cvar_loss_budget)
```

### 4.7 Constrained Objective

基础 reward：

```text
net_log_return_after_cost
```

约束：

```text
E[turnover_t] <= average_turnover_per_step_budget
E[transaction_cost_t] <= average_cost_per_step_budget
CVaR_loss_5 <= cvar_loss_budget
max_drawdown_loss <= drawdown_budget
```

报告指标另行记录：

```text
total_turnover = sum_t turnover_t
total_transaction_cost = sum_t transaction_cost_t
```

HPO search 和 constrained training 使用 per-step budget；promotion / 论文表格可同时展示 `average_cost_per_step` 与 `total_transaction_cost`，不得把两者混用。

训练目标：

```text
maximize:
  E[net_log_return_after_cost]
  - lambda_turnover * turnover_violation
  - lambda_cost * cost_violation
  - lambda_cvar * cvar_violation
  - lambda_drawdown * drawdown_violation
```

首版建议用 PPO-style clipped actor-critic + Lagrangian multipliers，不直接上完整 CPO 二阶优化。

---

## 5. 代码实现计划

### 5.1 新增文件

```text
src/baselines/risk_aware_gt_rcpo.py
src/models/risk_aware_graph_transformer.py
src/agents/constrained_actor_critic_agent.py
tests/test_p16_ra_gt_rcpo.py
configs/paper/p16_ra_gt_rcpo_smoke.yaml
configs/paper/p16_ra_gt_rcpo_pilot.yaml
configs/paper/p16_ra_gt_rcpo_ablation.yaml
configs/paper/p16_ra_gt_rcpo_formal_seed_runner.yaml
configs/paper/p16_ra_gt_rcpo_formal_comparison.yaml
```

### 5.2 修改文件

```text
src/baselines/__init__.py
src/experiments/registry.py
src/experiments/run_experiment.py
src/experiments/pipeline.py
src/experiments/paper_aggregate.py
src/experiments/formal_readiness.py
src/config.py
```

`src/experiments/pipeline.py` 是 P16 必改项。现有新模型 artifact / sidecar 逻辑不得继续硬编码 P12/P13 的 `MODEL_EXTENSION_ID` 或 `cage_*` 文件名；P16 必须按 `model_extension_id = core13_v2_p16_ra_gt_rcpo_20260525` 写入独立 sidecar、manifest 和 aggregate metadata。

### 5.3 注册名

正式模型：

```text
risk_aware_graph_transformer_constrained_actor_critic
```

消融模型：

```text
ra_gt_rcpo_no_graph
ra_gt_rcpo_no_transformer
ra_gt_rcpo_no_cvar_constraint
ra_gt_rcpo_no_cost_constraint
ra_gt_rcpo_no_turnover_constraint
ra_gt_rcpo_mlp_actor_critic
```

### 5.4 HPO whitelist

`hpo.trainable_models` 显式列出的 P16 模型必须能进入 equal-budget HPO。正式配置不得依赖 `hpo.native_only` 黑白名单推断。

### 5.5 输出 sidecar

P16 必须新增或复用以下诊断产物：

```text
ra_gt_rcpo_daily_diagnostics.csv
ra_gt_rcpo_constraint_multipliers.csv
ra_gt_rcpo_graph_diagnostics.csv
ra_gt_rcpo_actor_critic_training_history.csv
ra_gt_rcpo_risk_decomposition.csv
```

最低字段：

```text
date
model_name
seed
fold_id
rho
rebalance_intensity
scheduler_allowed_rebalance
estimated_turnover
realized_turnover
estimated_cost
realized_cost
CVaR_loss_5
max_drawdown_loss
lambda_turnover
lambda_cost
lambda_cvar
lambda_drawdown
graph_feature_mode
constraint_violation_count
```

### 5.6 Formal readiness 接入

`src/experiments/formal_readiness.py` 必须新增 P16 审计路径：

```text
p16_validation_references:
  required_for = promotion_gate
  status = required_before_p16_2_gate

p16_promotion_gate:
  required_for = p16_formal
  files:
    - promotion_gate_report.csv
    - validation_reference_comparison.csv

p16_ra_gt_rcpo_final:
  required_for = paper_final_if_p16_promoted
  files:
    - paper_main_comparison.csv
    - paper_seed_summary.csv
    - paper_paired_statistics.csv
    - paper_aggregate_manifest.json
```

P16 readiness 口径：

```text
P16.0 / P16.1: 不依赖 P14 完成
P16.2 promotion: 依赖 p16_validation_references
P16.4 formal: 依赖 P16 promotion gate passed
P16.5 combined aggregation: 依赖 P14/P12 formal outputs 和 P16 formal outputs
```

---

## 6. 实验阶段

### P16.0 代码 readiness

目标：

```text
确认模型可注册、配置可加载、输出 schema 可聚合。
```

检查项：

```text
pytest tests/test_p16_ra_gt_rcpo.py
python - <<'PY'
from src.config import ConfigLoader
from src.experiments.registry import ExperimentRegistry, DEEP_BASELINE_CLASSES
from src.experiments.run_experiment import NATIVE_HPO_MODEL_NAMES

model_name = "risk_aware_graph_transformer_constrained_actor_critic"
cfg = ConfigLoader.load("configs/paper/p16_ra_gt_rcpo_smoke.yaml")
assert model_name in DEEP_BASELINE_CLASSES
assert model_name in NATIVE_HPO_MODEL_NAMES
exp = ExperimentRegistry().create_experiment(cfg)
print(type(exp).__name__)
PY
```

说明：当前 `src.experiments.run_experiment` 不支持 `--dry-run`。若未来实现 dry-run，可把上面的 registry smoke test 替换为 CLI dry-run；实现前不得在 runbook 中使用不可执行命令。

通过标准：

```text
ConfigLoader.load 成功
ExperimentRegistry 可实例化
DEEP_BASELINE_CLASSES 包含 risk_aware_graph_transformer_constrained_actor_critic
run_experiment.NATIVE_HPO_MODEL_NAMES 包含 risk_aware_graph_transformer_constrained_actor_critic
策略 fit_required=true
fit() 真实训练，gradient_updates > 0
daily_returns/daily_weights/daily_turnover/daily_costs 非空
sidecar diagnostics 非空
```

### P16.1 Smoke

配置：

```text
configs/paper/p16_ra_gt_rcpo_smoke.yaml
```

预算：

```text
seed = 42
n_trials = 1
max_train_steps = 32
max_validation_steps = 32
max_gradient_updates_per_epoch = 8
```

用途：

```text
只验证可运行，不进入主表。
```

输出：

```text
results/EXP34_P16_ra_gt_rcpo_smoke_s42
```

rankability：

```text
rankable_in_unified_table = false
diagnostic_status = diagnostic
```

### P16.2 Validation Pilot

配置：

```text
configs/paper/p16_ra_gt_rcpo_pilot.yaml
```

预算：

```text
seed = 42
n_trials_per_model = 5
selection_split = validation
final_report_split = validation
test_outputs = forbidden
```

候选：

```text
risk_aware_graph_transformer_constrained_actor_critic
ra_gt_rcpo_no_graph
ra_gt_rcpo_no_transformer
ra_gt_rcpo_no_cvar_constraint
ra_gt_rcpo_no_cost_constraint
ra_gt_rcpo_no_turnover_constraint
ra_gt_rcpo_mlp_actor_critic
```

用途：

```text
只用于 promotion gate，不作为正式结果。
不得生成 test split final report，不得写 test daily outputs。
```

### P16.2a Validation Reference Bundle

P16 promotion gate 不得直接引用既有 final test 表。P16 pilot 前必须生成同 split、同 seed、同 validation-only 口径的 reference bundle：

```text
results/paper_tables/p16_validation_references/
  validation_reference_comparison.csv
  validation_reference_daily_returns.csv
  validation_reference_risk_metrics.csv
  validation_selection_report.csv
```

最小模型覆盖：

```text
eiie_native
ppo_native
cage_eiie_joint_light
full_dqn_gated_multitask_cnn_ppo
cnn_ppo_native
pgportfolio_eiie_native
risk_parity
```

最小字段：

```text
paper_model_id
model_name
seed
split = validation
cumulative_return
Sharpe
max_drawdown_loss
CVaR_loss_5
average_turnover
average_cost_per_step
total_transaction_cost
validation_utility
source_run_dir
rankable_reference = true
```

`cage_eiie_joint_light_validation_utility`、`ppo_native_validation_average_turnover` 等 promotion gate 输入只能来自该目录，不能从 P12/P14 final test 表反推。

### P16.3 Promotion Gate

P16 必须同时满足：

```text
failed_trial_rate <= 20%
finite_artifact_rate == 1.0
validation_cumulative_return >= eiie_native_validation - 0.01
validation_CVaR_loss_5 <= eiie_native_validation_CVaR_loss_5
validation_max_drawdown_loss <= eiie_native_validation_max_drawdown_loss
validation_average_turnover <= ppo_native_validation_average_turnover
validation_average_cost_per_step <= configured_average_cost_per_step_budget + cost_budget_tolerance
```

默认：

```text
cost_budget_tolerance = 1.0e-6
```

若目标是替代 CAGE 主模型，还必须满足至少一个：

```text
validation_utility >= cage_eiie_joint_light_validation_utility
或
validation_cumulative_return >= cage_eiie_joint_light_validation_cumulative_return - 0.01
且 CVaR/MDD/turnover 至少两个指标优于 CAGE
```

未通过：

```text
只进入 appendix diagnostic / future work
不得进入 P16 formal main table
```

### P16.4 Formal HPO

只有通过 P16.3 的模型进入 formal。

预算：

```text
seeds = [42, 123, 2024, 3407, 9999]
n_trials_per_model = 50
selection_split = validation
final_report_split = test
```

输出：

```text
results/EXP35_P16_formal_ra_gt_rcpo_s42
results/EXP35_P16_formal_ra_gt_rcpo_s123
results/EXP35_P16_formal_ra_gt_rcpo_s2024
results/EXP35_P16_formal_ra_gt_rcpo_s3407
results/EXP35_P16_formal_ra_gt_rcpo_s9999
```

必须生成：

```text
hpo_search_space_manifest.csv
hpo_trials.csv
hpo_model_final_comparison.csv
hpo_model_final_daily_returns.csv
hpo_model_final_daily_weights.csv
hpo_model_final_daily_turnover.csv
hpo_model_final_daily_costs.csv
hpo_model_final_risk_metrics.csv
ra_gt_rcpo_* sidecar diagnostics
run_manifest.json 或 formal sidecar manifest
```

### P16.4a P1 Fixed Deterministic Formal Export

Table P16-A 若包含 `risk_parity / buy_and_hold / equal_weight`，必须先生成 P1 fixed deterministic formal export。现有 `EXP04_P1_pilot_baseline_main_native_fixed_s42` 是 diagnostic / rankable=false，不能直接进入 P16-A。

目标输出：

```text
results/EXP36_P1_fixed_deterministic_formal_export/
  metrics/baseline_comparison.csv
  daily_returns.csv 或 metrics/hpo_model_final_daily_returns.csv 等价正式导出
  daily_weights.csv
  daily_turnover.csv
  daily_costs.csv
  logs/run_manifest.json
```

manifest / sidecar 必须写明：

```text
diagnostic_status = formal
rankable_in_unified_table = true
baseline_family = traditional
deterministic_baseline = true
n_independent_seeds = 1
protocol_id = core13_v2_full_reset_20260522
data_cutoff_date = 2026-05-20
data_mode = availability_mask
valuation_execution_split = true
availability_mask_contract_passed = true
test_used_for_model_selection = false
```

P1 deterministic formal export 不属于 HPO per-model final output；它是 P16.5 聚合器的单独合法输入类别。

### P16.5 Combined Aggregation

聚合范围：

```text
P16 formal run dirs
P14 new-model final run dirs
P7 main native formal run dirs
P9 related-work formal run dirs
P1 fixed traditional baseline run dirs
```

如果 Table P16-A 展示 `risk_parity / buy_and_hold / equal_weight`，P1 fixed traditional baseline 就是 mandatory input，不能写成 optional。聚合时必须标记：

```text
baseline_family = traditional
deterministic_baseline = true
n_independent_seeds = 1
rankable_in_unified_table = true
```

确定性 baseline 可用于 paired daily comparison，但不得解释为 5-seed HPO 随机模型。

输出目录：

```text
results/paper_tables/p16_ra_gt_rcpo_final
```

必须输出：

```text
paper_main_comparison.csv
paper_seed_summary.csv
paper_paired_statistics.csv
paper_diagnostic_comparison.csv
paper_aggregate_manifest.json
```

聚合规则：

```text
1. Trainable model rows 只读 HPO per-model final outputs。
2. Deterministic traditional rows 只读 P1 fixed deterministic formal export。
3. 不重复读取 parent best payload。
4. diagnostic / proxy / external original 不进入主表。
5. source_run + paper_model_id + seed 去重。
6. rankable_in_unified_table=false 的行只进 diagnostic。
```

目标聚合命令必须带 formal 过滤参数：

```text
python -m src.experiments.paper_aggregate \
  --run-dir <P16 formal seed run dirs...> \
  --run-dir <P14/P12 formal run dirs...> \
  --run-dir <P7/P9 formal run dirs...> \
  --run-dir <P1 fixed deterministic baseline run dir> \
  --output-dir results/paper_tables/p16_ra_gt_rcpo_final \
  --paper-group-id p16_ra_gt_rcpo_final \
  --benchmark-model <promoted_p16_primary_model> \
  --protocol-id core13_v2_full_reset_20260522 \
  --data-cutoff-date 2026-05-20 \
  --require-formal-manifest \
  --require-availability-mask-contract
```

若当前 CLI 缺少上述参数，P16.5 属于 target-after-implementation，不能执行降级聚合冒充 formal。

---

## 7. HPO Search Space

首版 search space：

```yaml
hpo:
  n_trials_per_model: 50
  trainable_models:
    - risk_aware_graph_transformer_constrained_actor_critic
  search_space:
    ra_gt_rcpo.learning_rate:
      type: float
      low: 1.0e-5
      high: 5.0e-4
      log: true
    ra_gt_rcpo.lambda_turnover:
      type: float
      low: 0.1
      high: 5.0
    ra_gt_rcpo.lambda_cost:
      type: float
      low: 1.0
      high: 30.0
    ra_gt_rcpo.average_cost_per_step_budget:
      type: float
      low: 0.00005
      high: 0.00100
    ra_gt_rcpo.lambda_cvar:
      type: float
      low: 0.0
      high: 2.0
    ra_gt_rcpo.lambda_drawdown:
      type: float
      low: 0.0
      high: 2.0
    ra_gt_rcpo.cvar_loss_budget:
      type: float
      low: 0.005
      high: 0.04
    ra_gt_rcpo.drawdown_budget:
      type: float
      low: 0.05
      high: 0.15
    ra_gt_rcpo.graph_edge_threshold:
      type: float
      low: 0.05
      high: 0.40
    ra_gt_rcpo.transformer_layers:
      type: int
      low: 1
      high: 3
    ra_gt_rcpo.attention_heads:
      type: categorical
      choices: [2, 4]
```

Search-space manifest 必须使用现有平台 schema：

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

---

## 8. 指标

主指标：

```text
cumulative_return
Sharpe
max_drawdown_loss
CVaR_loss_5
average_turnover
total_transaction_cost
```

辅助指标：

```text
Sortino
Calmar
volatility
return_per_turnover
return_per_cost
constraint_violation_rate
finite_artifact_rate
failed_trial_rate
```

口径：

```text
failed_trial_rate = failed_trials / total_trials
finite_artifact_rate = finite_daily_output_trials / completed_trials
pruned_trials 单独统计，不计入 failed_trials，除非 pruning 原因是非有限输出或训练异常。
```

论文主张不得只用累计收益。P16 的合理成功标准是 Pareto improvement：

```text
收益不显著弱于 CAGE/EIIE，
同时 MDD/CVaR/turnover/cost 至少两个指标明显改善。
```

---

## 9. 表格设计

### Table P16-A: Formal model comparison

行：

```text
risk_aware_graph_transformer_constrained_actor_critic
cage_eiie_joint_light
cage_eiie_distributional
eiie_native
pgportfolio_eiie_native
cnn_ppo_native
full_dqn_gated_multitask_cnn_ppo
ppo_dqn_hierarchical_reimplementation
risk_parity
buy_and_hold
equal_weight
```

`risk_parity / buy_and_hold / equal_weight` 是 deterministic traditional baselines；表格必须保留 `baseline_family`、`deterministic_baseline`、`n_independent_seeds` 或等价元数据，避免被误读为 5-seed HPO 模型。

列：

```text
cumulative_return
Sharpe
max_drawdown_loss
CVaR_loss_5
average_turnover
total_transaction_cost
n_seeds
rankable_in_unified_table
```

### Table P16-B: Architecture ablation

默认口径：P16-B 是 validation pilot diagnostic ablation table，不进入 formal main ranking。

若 P16-B 要进入正式附录表，所有 ablation variants 必须按 P16 formal 预算重跑：

```text
50 trials/model × 5 seeds
same validation-only selection
same formal manifest
same availability-mask contract
```

行：

```text
risk_aware_graph_transformer_constrained_actor_critic
ra_gt_rcpo_no_graph
ra_gt_rcpo_no_transformer
ra_gt_rcpo_no_cvar_constraint
ra_gt_rcpo_no_cost_constraint
ra_gt_rcpo_no_turnover_constraint
ra_gt_rcpo_mlp_actor_critic
```

### Table P16-C: Constraint diagnostics

列：

```text
lambda_turnover_mean
lambda_cost_mean
lambda_cvar_mean
lambda_drawdown_mean
constraint_violation_rate
avg_turnover
avg_cost
CVaR_loss_5
max_drawdown_loss
```

---

## 10. 图设计

建议图：

```text
1. NAV curve: P16 vs CAGE vs EIIE vs Risk Parity
2. Drawdown curve
3. Turnover / cost curve
4. CVaR rolling tail-risk curve
5. Graph attention / correlation heatmap
6. Constraint multiplier trajectory
```

图源必须来自 formal artifacts，不得从 notebook 手工拼图。

Graph heatmap 只能使用：

```text
decision-window rolling correlation
或 learned attention weights from formal diagnostics
```

禁止使用 full-sample correlation、test-window statistics 或 post-hoc regime labels 生成图。

---

## 11. 论文写作口径

若 P16 超过 CAGE：

```text
The proposed risk-aware graph-transformer constrained actor-critic improves the formal Core-13 return-risk-cost frontier.
```

若 P16 收益低于 CAGE，但风险更好：

```text
RA-GT-RCPO provides a more conservative risk-controlled alternative, reducing drawdown and tail risk at the cost of lower cumulative return.
```

若 P16 未通过 gate：

```text
The graph-transformer constrained actor-critic extension did not pass the validation-only promotion gate under Core-13 v2. This negative result suggests that, for a small ETF/LOF universe, representation complexity alone is insufficient without stronger allocation alpha.
```

必须披露：

```text
P16 is a post-hoc model extension motivated by completed Core-13 v2 results.
The final test split was not used for HPO selection.
```

---

## 12. Go / No-Go

### 可以启动 P16.0 的条件

```text
当前 Core-13 v2 readiness audit = go
P16 目标配置文件已存在
P16 registry smoke test 可执行
P16 不依赖 P14/P12 formal outputs 启动 code readiness
```

P14/P12 formal outputs 只在 P16.5 combined aggregation 前强制要求。

### 可以启动 P16.1 smoke 的条件

```text
risk_aware_graph_transformer_constrained_actor_critic 注册成功
配置可加载
fit() 真实训练，gradient_updates > 0
sidecar diagnostics 可写
```

### 可以启动 P16.4 formal 的条件

```text
P16.2 validation pilot 通过 promotion gate
hpo_search_space_manifest.csv 完整
run_manifest / formal sidecar 完整
availability_mask contract 通过
test_used_for_model_selection = false
```

### 不得进入主表的情况

```text
只跑 smoke
只跑 pilot
gradient_updates = 0
rankable_in_unified_table = false
缺 hpo_search_space_manifest
缺 formal manifest
未通过 validation promotion gate
```

---

## 13. 最小执行顺序

当前阶段建议：

```text
1. 完成 P16 详细设计。
2. 实现 risk_aware_graph_transformer_constrained_actor_critic。
3. 跑 P16.0 单元测试和配置加载。
4. 跑 P16.1 smoke。
5. 跑 P16.2 validation pilot。
6. 仅当通过 promotion gate，进入 P16.4 formal。
7. 聚合 P16 + P14。
8. 再决定是否改写论文主结果。
```

不建议：

```text
直接跑 P16 formal
把 P13 lite formal config 直接升级为主表
把 P13 lite 已失败 pilot 写成完整 RA-GT-RCPO 实验
```
