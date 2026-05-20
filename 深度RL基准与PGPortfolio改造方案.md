# 深度 RL Baseline 与 PGPortfolio 改造方案

生成日期：2026-05-14

## 1. 结论

两个分析师的核心判断属实：

- 当前平台的 `ppo_baseline`、`cnn_ppo_baseline`、`bernoulli_gated_ppo`、`dqn_only`、`eiie` 仍是 `supervised_execution_aligned_proxy`，不能称为原生 PPO / DQN / EIIE 强化学习基准。
- PGPortfolio 值得加入 baseline，但不应直接复制或改写其 GPL-3.0 源码进入 `src/`。
- 正确改造路径应分两条线：
  - 平台内新增 PyTorch 原生 RL baselines，统一使用当前 ETF/LOF 数据、`PortfolioRebalanceEnv`、`CostModel`、`ConstraintManager`、`BacktestEngine`。
  - PGPortfolio 作为 external original baseline；只有白名单内本地克隆允许通过独立环境/子进程运行，白名单外只能导入用户手动运行后的结果。

验收口径：

- 平台内原生 baseline 必须输出 `rl_training=true`、`platform_native_rl_training=true`，且训练日志能证明其使用 rollout / replay / policy gradient，而不是一次性监督式未来收益拟合。
- PGPortfolio external baseline 必须输出 `rl_training=true`、`platform_native_rl_training=false`、`external_original_implementation=true`、`license="GPL-3.0"`，并在 manifest 中记录外部 repo 路径、commit、命令、依赖环境、数据导出口径。
- 当前 proxy baselines 应保留但重命名或显式标注为 neural proxy baselines，不应混入 native RL 表格。

## 2. 当前代码事实

当前 deep baseline 注册点：

- `{ "kind": "path", "ref": "src/experiments/registry.py:139" }`
- `{ "kind": "path", "ref": "src/experiments/registry.py:140" }`
- `{ "kind": "path", "ref": "src/experiments/registry.py:141" }`
- `{ "kind": "path", "ref": "src/experiments/registry.py:142" }`
- `{ "kind": "path", "ref": "src/experiments/registry.py:143" }`
- `{ "kind": "path", "ref": "src/experiments/registry.py:144" }`

当前 proxy 训练元数据：

- `{ "kind": "path", "ref": "src/baselines/deep_training.py:173" }`
- `{ "kind": "path", "ref": "src/baselines/deep_training.py:174" }`
- `{ "kind": "path", "ref": "tests/test_baseline_output_schema.py:215" }`
- `{ "kind": "path", "ref": "tests/test_baseline_output_schema.py:216" }`

当前 EIIE proxy 训练方式：

- `{ "kind": "path", "ref": "src/baselines/eiie.py:45" }`
- `{ "kind": "path", "ref": "src/baselines/eiie.py:105" }`

当前 DQN-only 只实现 `hold` 和 `equal_weight` 模板：

- `{ "kind": "path", "ref": "src/baselines/dqn_only.py:25" }`
- `{ "kind": "path", "ref": "src/baselines/dqn_only.py:40" }`

当前 baseline comparison 仍按 strategy 回测，不负责原生 RL 训练：

- `{ "kind": "path", "ref": "src/experiments/pipeline.py:383" }`
- `{ "kind": "path", "ref": "src/experiments/pipeline.py:397" }`

已有可复用原生训练组件：

- `{ "kind": "path", "ref": "src/envs/portfolio_rebalance_env.py:24" }`
- `{ "kind": "path", "ref": "src/agents/ppo_agent.py:19" }`
- `{ "kind": "path", "ref": "src/agents/dqn_agent.py:16" }`
- `{ "kind": "path", "ref": "src/buffers/rollout_buffer.py" }`
- `{ "kind": "path", "ref": "src/buffers/replay_buffer.py" }`
- `{ "kind": "path", "ref": "src/buffers/prioritized_replay_buffer.py" }`

## 3. PGPortfolio 外部约束

外部来源：

- GitHub: https://github.com/ZhengyaoJiang/PGPortfolio
- Paper: https://arxiv.org/pdf/1706.10059.pdf

已核实事实：

- 仓库说明为 PGPortfolio / Policy Gradient Portfolio，是论文 `A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem` 的源码实现。
- README 说明其方法是 portfolio-specific policy optimization，使用 immediate reward optimization 并纳入 transaction cost regularization。
- GitHub 页面显示 license 为 GPL-3.0。
- README 依赖说明包含 Python 2.7+/3.5+、TensorFlow >= 1.0、tflearn，与当前 PyTorch/Gymnasium 栈不匹配。

结论：

- 不直接 vendoring PGPortfolio 源码到本仓库。
- 不从 PGPortfolio 复制代码片段到 `src/`。
- 仅允许在白名单内调用用户本地克隆的 PGPortfolio 作为 external subprocess baseline；白名单外只能导入用户手动运行后的结果。
- 平台内的 `pgportfolio_eiie_native` 应为 clean-room PyTorch 复现，只复刻算法语义和论文口径。

## 4. Baseline 命名与分层

### 4.1 保留但重命名当前 proxy baselines

当前实现继续保留，用于诊断和快速 smoke：

- `ppo_proxy`
- `cnn_ppo_proxy`
- `bernoulli_gated_ppo_proxy`
- `dqn_template_proxy`
- `eiie_proxy`

兼容策略：

- 旧名称 `ppo_baseline`、`cnn_ppo_baseline`、`bernoulli_gated_ppo`、`dqn_only`、`eiie` 暂时保留 alias。
- `run_manifest.json`、`baseline_comparison.csv` 必须写入：
  - `baseline_training_family="neural_proxy"`
  - `training_algorithm="supervised_execution_aligned_proxy"`
  - `rl_training=false`
  - `platform_native_rl_training=false`
  - `execution_path_proxy=true`

### 4.2 新增平台内 native RL baselines

新增名称：

- `ppo_native`
- `cnn_ppo_native`
- `bernoulli_gated_ppo_native`
- `dqn_template_native`
- `eiie_native`
- `pgportfolio_eiie_native`

共同要求：

- 统一从 `build_pipeline_artifacts()` 获取 dataset、split、market image。
- 统一训练使用 train split。
- 统一 validation 用 deterministic policy 评估和 checkpoint 选择。
- 统一 test 必须用 `BacktestEngine` 输出 daily CSV；native runner 不允许自行拼接 `daily_returns.csv`、`daily_weights.csv`、`daily_turnover.csv`、`daily_rebalance.csv`、`daily_costs.csv`。
- 统一使用当前 `next_open` / `next_close` 执行、成本、约束、可用性 mask、调仓频率。

### 4.3 新增 PGPortfolio original external baseline

新增名称：

- `pgportfolio_original_external`

定位：

- 作者原版实现 baseline。
- 独立 Python 环境和独立源码目录。
- 本平台只负责数据导出、白名单内命令调用、结果导入、schema 对齐；白名单外 repo 不由平台启动进程。

Manifest 必须标记：

- `external_original_implementation=true`
- `external_repo="https://github.com/ZhengyaoJiang/PGPortfolio"`
- `external_license="GPL-3.0"`
- `external_dependency_stack="tensorflow1/tflearn"`
- `source_code_vendored=false`

## 5. 代码改造方案

### 5.1 配置层

修改文件：

- `{ "kind": "path", "ref": "src/config.py" }`
- `{ "kind": "path", "ref": "configs/baselines.yaml" }`
- `{ "kind": "path", "ref": "configs/experiments/baseline_comparison.yaml" }`

新增配置建议：

```yaml
baselines:
  proxy:
    enabled: true
    models:
      - ppo_proxy
      - cnn_ppo_proxy
      - bernoulli_gated_ppo_proxy
      - dqn_template_proxy
      - eiie_proxy
  native_rl:
    enabled: true
    models:
      - ppo_native
      - cnn_ppo_native
      - bernoulli_gated_ppo_native
      - dqn_template_native
      - eiie_native
      - pgportfolio_eiie_native
    train_epochs: 10
    validation_interval: 1
    checkpoint: true
  external_pgportfolio:
    enabled: false
    repo_path: null
    python_executable: null
    config_template_path: null
    timeout_seconds: 86400
    docker_image: null
```

校验规则：

- `external_pgportfolio.enabled=true` 时：
  - `repo_path` 必须存在。
  - 默认只允许 `repo_path` 位于项目白名单内，推荐路径为 `external/PGPortfolio`。
  - 若 `repo_path` 位于白名单外，平台代码不得主动执行；child run 状态必须为 `skipped_out_of_scope`，并在 manifest 中记录 `{ "kind": "out_of_scope", "ref": "external_pgportfolio_repo" }`。
  - 如需使用白名单外 PGPortfolio，用户应在外部手动运行，再通过 import 脚本导入结果；平台不得跨当前子树执行外部 repo 命令。
  - `python_executable` 必须存在或由 `docker_image` 提供。
  - 默认 CI 不启用 external baseline。

### 5.2 Registry 层

修改文件：

- `{ "kind": "path", "ref": "src/experiments/registry.py" }`

拆分当前 registry：

```python
PROXY_BASELINE_CLASSES = {
    "ppo_proxy": PPOBaselineStrategy,
    "cnn_ppo_proxy": CNNPPOBaselineStrategy,
    "bernoulli_gated_ppo_proxy": BernoulliGatedPPOStrategy,
    "dqn_template_proxy": DQNOnlyStrategy,
    "eiie_proxy": EIIEStrategy,
}

NATIVE_RL_BASELINE_RUNNERS = {
    "ppo_native": NativePPOBaselineRunner,
    "cnn_ppo_native": NativeCNNPPOBaselineRunner,
    "bernoulli_gated_ppo_native": NativeBernoulliGatedPPOBaselineRunner,
    "dqn_template_native": NativeDQNTemplateBaselineRunner,
    "eiie_native": NativeEIIEBaselineRunner,
    "pgportfolio_eiie_native": PGPortfolioEIIENativeRunner,
}

EXTERNAL_BASELINE_RUNNERS = {
    "pgportfolio_original_external": ExternalPGPortfolioRunner,
}
```

兼容 alias：

```python
LEGACY_BASELINE_ALIASES = {
    "ppo_baseline": "ppo_proxy",
    "cnn_ppo_baseline": "cnn_ppo_proxy",
    "bernoulli_gated_ppo": "bernoulli_gated_ppo_proxy",
    "dqn_only": "dqn_template_proxy",
    "eiie": "eiie_proxy",
}
```

### 5.3 Pipeline 层

修改文件：

- `{ "kind": "path", "ref": "src/experiments/pipeline.py" }`

新增统一 runner 协议：

```python
class TrainingResult(TypedDict):
    status: str
    training_algorithm: str
    rl_training: bool
    platform_native_rl_training: bool
    training_summary: dict[str, Any]
    training_history: pd.DataFrame
    checkpoint_best_path: str | None
    checkpoint_last_path: str | None

class BaselineRunResult(TypedDict):
    status: str
    model_name: str
    baseline_family: str
    training_algorithm: str
    rl_training: bool
    platform_native_rl_training: bool
    proxy_training: bool
    external_original_implementation: bool
    daily_returns: pd.DataFrame
    daily_weights: pd.DataFrame
    daily_turnover: pd.DataFrame
    daily_rebalance: pd.DataFrame
    daily_costs: pd.DataFrame
    metrics: dict[str, float]
    model_returns: dict[str, pd.DataFrame]
    benchmark_returns: dict[str, pd.DataFrame]
    training_summary: dict[str, Any]
    training_history: pd.DataFrame
    training_history_flat: pd.DataFrame
    checkpoint_best_path: str | None
    checkpoint_last_path: str | None
    evaluated_checkpoint_path: str | None

class NativeBaselineRunner(Protocol):
    def fit(self, train_env: PortfolioRebalanceEnv, validation_env: PortfolioRebalanceEnv) -> TrainingResult: ...
    def evaluate_with_backtest(self, artifacts: Mapping[str, Any], segment: str, deterministic: bool = True) -> BaselineRunResult: ...
    def save_checkpoint(self, path: str | Path) -> str: ...
    def load_checkpoint(self, path: str | Path) -> None: ...
```

`TrainingResult.training_history` 最小列要求：

```text
epoch
step
env_steps
gradient_updates
train_reward
validation_metric
loss
status
```

约束：

- `training_history` 必须是一行一事件或一行一 epoch 的扁平 `pd.DataFrame`，不能塞嵌套 dict。
- native baseline 若 `status="completed"`，`training_history` 必须非空，且至少包含上述列。
- `validation_metric` 在非验证 step 可为空；completed run 至少必须有一行 finite `validation_metric`，否则不能选择 best checkpoint，run 状态应为 `failed_no_finite_validation_metric`。
- `checkpoint_best_path` 必须对应 best validation row；若 best validation row 缺 checkpoint，run 状态应为 `failed_missing_best_checkpoint`。
- 额外明细可加列，例如 `policy_loss`、`value_loss`、`entropy`、`epsilon`、`replay_size`，但不得替代最小列。

`run_strategy_comparison()` 改造为：

- proxy strategy：沿用当前 `BaseStrategy + BacktestEngine`。
- native runner：调用 `runner.fit(train_env, validation_env)`，再调用 `runner.evaluate_with_backtest(artifacts, segment="test", deterministic=True)`；该方法内部必须通过 `BacktestEngine` 生成统一 daily outputs。
- external runner：调用外部导出/导入流程；只有白名单内 `repo_path.resolve()` 才允许子进程执行，白名单外只能导入用户手动运行结果。

`evaluate_with_backtest()` 强制流程：

1. 从 `TrainingResult.checkpoint_best_path` 加载 best validation checkpoint。
2. 构造只读 deterministic strategy adapter。
3. 调用 `BacktestEngine.run()` 生成 test daily outputs。
4. 在 `BaselineRunResult` 中记录 `evaluated_checkpoint_path`。

禁止用训练结束后的 last weights 直接跑 test，除非 `checkpoint_best_path` 缺失且该 child run 状态明确为 `failed_missing_best_checkpoint`。

禁止行为：

- 不允许 native baseline 在未训练时直接 `eval()` 回测。
- 不允许 `platform_native_rl_training=true` 但 `training_history` 为空。
- 不允许 `status="completed"` 但无 daily outputs。
- 不允许 native runner 绕过 `BacktestEngine` 自行构造冻结日度 CSV。

### 5.4 原生 PPO / CNN-PPO baseline

新增文件：

- `{ "kind": "path", "ref": "src/baselines/native_ppo.py" }`

实现方式：

- `NativePPOBaselineRunner` 复用：
  - `PortfolioRebalanceEnv`
  - `PPOAgent`
  - `PPOActor`
  - `PPOCritic`
  - `EncoderFactory`
- `ppo_native` 使用 MLP/flatten encoder。
- `cnn_ppo_native` 使用 CNN encoder。
- 关闭 DQN gate：
  - `gate_network=None`
  - scheduler 允许调仓日提交 `rebalance_action=1`
  - scheduler 不允许调仓日必须 hold，提交 `rebalance_action=0` 或保持当前权重
  - `rebalance_intensity=1.0`
  - 不得绕过统一 `PortfolioRebalanceEnv` / `PortfolioExecutionCore` / `BacktestEngine`

训练语义：

- 从 train env collect rollout。
- 使用 GAE。
- 使用 clipped PPO objective。
- validation deterministic。
- best checkpoint 由 validation metric 选择。

输出：

- `training_algorithm="ppo_clipped_gae"`
- `rl_training=true`
- `platform_native_rl_training=true`
- `gate_training=false`

关键测试：

- `tests/test_native_baselines.py::test_ppo_native_collects_rollout_and_updates_parameters`
- `tests/test_native_baselines.py::test_cnn_ppo_native_outputs_nonempty_daily_csv`
- `tests/test_baseline_output_schema.py::test_native_ppo_manifest_marks_platform_native_rl`

### 5.5 Bernoulli-Gated CNN-PPO native baseline

新增文件：

- `{ "kind": "path", "ref": "src/baselines/native_bernoulli_gated_ppo.py" }`

实现方式：

- PPO actor 输出 candidate weights。
- Bernoulli gate 输出 `p_rebalance`。
- action：
  - sampled gate = 0：保持当前权重。
  - sampled gate = 1：执行 PPO candidate weights。
- rollout item 必须保存：
  - `gate_log_prob`
  - `gate_entropy`
  - `p_rebalance`
  - `rebalance_action`
  - `candidate_log_prob`
  - realized reward/cost/turnover

loss：

- actor PPO clipped loss。
- gate on-policy policy-gradient loss。
- entropy regularization。
- value loss。

归因规则：

- `gate=1` 时，candidate weights 实际进入环境执行，candidate log_prob 才进入 PPO actor policy loss。
- `gate=0` 时，PPO candidate 未执行，不能把 hold 收益归因给 PPO candidate；该步 candidate log_prob 不进入 PPO actor policy loss，或 actor loss weight 必须为 0。
- `gate=0` 时 critic / encoder 可继续通过 value loss、auxiliary loss 或共享表示损失更新。
- gate policy loss 使用该步实际执行动作的 advantage，不能用未执行 candidate 的收益替代。

禁止：

- 不用 DQN replay 训练 Bernoulli gate。
- 不用 MSE 拟合 `[hold_reward, rebalance_reward]`。

输出：

- `training_algorithm="bernoulli_gated_ppo_on_policy"`
- `rl_training=true`
- `platform_native_rl_training=true`
- `gate_training="on_policy_bernoulli"`

关键测试：

- gate=hold 时 env 收到 `rebalance=0`。
- gate log_prob 参与 loss 且参数梯度非空。
- `p_rebalance`、gate entropy、rebalance frequency 落盘。

### 5.6 DQN-only native template selector

新增文件：

- `{ "kind": "path", "ref": "src/baselines/native_dqn_template.py" }`

动作空间：

```text
0: equal_weight
1: minimum_variance
2: maximum_sharpe
3: risk_parity
4: inverse_volatility
5: defensive / money-market-heavy allocation
6: momentum_top_k
```

说明：

- 默认 `dqn_template_native` 严格按原始需求使用 7 个模板动作，不包含 `hold`。
- 非调仓日由 `RebalanceScheduler` 统一 hold，不把 hold 作为 DQN 模板动作。
- 若后续需要 hold 作为可学习动作，必须新增独立模型名 `dqn_template_with_hold_native`，并显式标注 `action_dim=8`，不能复用 `dqn_template_native` 名称。

模板实现：

- 复用已有传统策略或风险估计函数，避免重复实现优化器。
- 所有模板输出必须经过 `ConstraintManager.project()`。
- 不可用资产权重必须为 0。
- 所有模板只能使用 decision date 可见数据，不能读 execution-only 字段或未来窗口。

模板失败规则：

- `minimum_variance`、`maximum_sharpe`、`risk_parity` 等模板若因窗口不足、协方差奇异、优化器失败而无法生成合法权重，应返回 `template_status="invalid"`。
- 训练交互阶段默认 mask invalid 动作，DQN epsilon-greedy / argmax 只能在 valid template 集合中选动作。
- invalid 动作不得发送给 env 执行，也不得生成真实 next_state transition。
- invalid penalty 只作为 counterfactual / auxiliary Q target，用于压低 invalid template Q 值；默认固定使用：
  - `dqn_template.invalid_action_penalty=1.0`
  - `dqn_template.invalid_action_auxiliary_target_kept=true`
  - `dqn_template.invalid_action_auxiliary_reward = realized_reward - invalid_action_penalty`
  - auxiliary record 记录 `invalid_action=true`、`fallback_used=false`、`bootstrap_mask=0`、`next_state_source="none"`
  - invalid auxiliary target 只做标量监督 target，不参与真实 DQN transition bootstrap，不得复用 selected valid action 的 next_state 继续 bootstrap。
  - 若配置覆盖 penalty，必须写入 run manifest 和 training summary。
- 若某一步所有非 fallback 模板均 invalid，训练交互阶段执行 `equal_weight` fallback，并记录 `fallback_used=true`、`fallback_reason="all_templates_invalid"`；该 transition 的 next_state 来自实际执行的 equal_weight action。
- 评估/回测时若被选中模板 invalid，允许 fallback 到 `equal_weight`，但必须在 `action_info`、`run_manifest.json` 或 `baseline_training_summary.csv` 中记录 `fallback_template` 与 `fallback_reason`；不得向冻结日度 CSV 追加新列。
- fallback 后仍必须经过 `ConstraintManager.project()`。
- 默认评估 fallback 策略固定为 `dqn_template.fallback_policy="equal_weight"`；不允许静默退化为 hold 或当前持仓。

Q 网络设计：

- 若 Q 值依赖每个模板的 target weights / turnover / estimated cost，不能只给网络一个 candidate weights 后一次性输出所有模板 Q。
- 推荐方案 A：对每个模板先生成 action feature：`[state_latent, template_id_embedding, template_weights, estimated_turnover, estimated_cost, validity_mask]`，批量通过共享 Q 网络得到 `Q(template)`。
- 推荐方案 B：使用 state-only template Q network，但模板成本、约束和 invalid mask 必须在 action selection / target 计算中显式修正。
- 禁止把所有非 equal-weight 模板退化成当前持仓或同一个 candidate。

训练语义：

- 使用 `DQNAgent`。
- replay buffer / PER。
- target network。
- Double DQN 可配置。
- epsilon-greedy。
- n-step return。
- warmup 后更新。

输出：

- `training_algorithm="double_dqn_template_selector"`
- `rl_training=true`
- `platform_native_rl_training=true`
- `action_space="portfolio_templates"`

关键测试：

- 7 个模板均可产生合法 simplex 权重。
- replay 非空后 DQN update 改变参数。
- `double_dqn=false` 时走 vanilla DQN target。
- `per_enabled=true` 时 priority 被更新。

### 5.7 EIIE native baseline

新增文件：

- `{ "kind": "path", "ref": "src/baselines/native_eiie.py" }`

模型：

- shared evaluator：每个资产共享 CNN/MLP evaluator。
- 输入：`market_image[:, :, asset] + previous_weight_asset`。
- 输出：每资产 score。
- mask：不可用资产 score = `-inf`。
- softmax 得到 target weights。

训练语义：

- 使用当前平台 execution-aligned return。
- 使用可微近似成本作为训练目标。
- 最终评估仍走 `BacktestEngine` 扣真实 realized cost。
- 必须维护 PVM，即每个训练样本使用上一期组合权重。
- 训练优化项和环境 reward 对齐项必须区分：
  - 平台默认 `next_open` 下，完整环境 net return 包含旧持仓从 decision close 到 execution open 的 pre-execution return、执行后的 holding return、以及实际成本。
  - pre-execution return 对当前 action 通常是常数项，训练优化时可不进入 actor 梯度，但 training summary 必须声明 `pre_execution_return_in_actor_loss=false`。
  - action 相关优化项应使用 execution-aligned `holding_simple_return` / price relative 与可微近似成本。
  - validation/test 绩效必须以 `BacktestEngine` 的 realized net return 为准。
- 成本近似只能使用 decision date 可见字段，如 current weights、candidate weights、amount/adv20/volatility at decision 或 `CostEstimator` 的 decision-state 输入；禁止使用 execution-only 成本字段作为训练特征。

loss：

```text
price_relative = 1 + holding_simple_return_t
pre_trade_weights = drift(previous_weights, pre_execution_return_t)  # actor loss 可视作 stop-gradient 或常数路径
action_cost = differentiable_cost(pre_trade_weights, w_t, decision_visible_cost_inputs)
portfolio_growth = sum(w_t * (price_relative - 1)) - action_cost
loss = -mean(log(max(1 + portfolio_growth, eps)))
```

防误读约束：

- `pre_execution_return_t` 只用于把上一期权重漂移到 execution 前权重，或作为 realized reward target 的 stop-gradient 项。
- `pre_execution_return_t` 不得进入 `market_image`、observation、feature matrix、gate input、actor input 或任何可被模型直接读取的训练特征。
- 训练样本构造时，`pre_execution_return_t` 必须来自执行口径标签生成流程，不能作为特征列参与 fit/transform。

PVM 规则：

- `previous_weights` 必须来自训练路径内已经发生的上一期组合权重。
- `previous_weights` 应先按已可见 price relative 漂移为交易前权重，再计算调仓成本。
- PVM 更新只能使用当前训练路径已执行权重，不能读取 validation/test 或未来日期权重。
- `log(1 + portfolio_growth)` 必须设置数值下限 `eps`，避免 `portfolio_growth <= -1` 导致非 finite loss。

输出：

- `training_algorithm="eiie_policy_gradient_pvm"`
- `rl_training=true`
- `platform_native_rl_training=true`
- `portfolio_vector_memory=true`

关键测试：

- previous weights 改变时同一 market image 输出/成本目标不同。
- unavailable asset 权重为 0。
- 训练一步后 evaluator 参数变化。

### 5.8 PGPortfolio EIIE native clean-room 复现

新增文件：

- `{ "kind": "path", "ref": "src/baselines/pgportfolio_eiie.py" }`

定位：

- 平台内 PyTorch 复现 PGPortfolio 算法语义。
- 不复制 PGPortfolio 代码。
- 以论文/公开算法描述为依据。

实现内容：

- EIIE shared evaluator。
- PVM。
- OSBL mini-batch sampler。
- immediate reward policy gradient。
- transaction cost regularization。

与 `eiie_native` 的区别：

- `eiie_native` 是更直接的 EIIE policy-gradient baseline。
- `pgportfolio_eiie_native` 必须显式实现 OSBL/PVM 采样与训练记录，作为论文对齐 baseline。

输出：

- `training_algorithm="pgportfolio_eiie_osbl"`
- `rl_training=true`
- `platform_native_rl_training=true`
- `portfolio_vector_memory=true`
- `online_stochastic_batch_learning=true`
- `clean_room_reimplementation=true`
- `source_code_vendored=false`

关键测试：

- OSBL sampler 只从 train split 采样。
- PVM 在 episode/reset 后按日期更新。
- immediate reward loss 可回传到 shared evaluator。
- manifest 不包含 GPL 源码路径为内部文件。

### 5.9 PGPortfolio original external baseline

新增文件：

- `{ "kind": "path", "ref": "src/baselines/external_pgportfolio.py" }`
- `{ "kind": "path", "ref": "src/experiments/external_baselines.py" }`

新增脚本：

- `{ "kind": "path", "ref": "scripts/export_pgportfolio_dataset.py" }`
- `{ "kind": "path", "ref": "scripts/import_pgportfolio_results.py" }`

外部目录要求：

- PGPortfolio repo 不放进当前 `src/`。
- 推荐用户自行 clone 到白名单允许目录，默认推荐 `external/PGPortfolio`。
- 白名单外 repo 不执行，child run 写 `skipped_out_of_scope`，并在 manifest 标记为 `out_of_scope`。
- 白名单外结果只能由用户外部手动运行后，通过 import 脚本导入；平台代码不跨子树启动外部进程。
- 本平台只在 `results/<run>/external_pgportfolio/` 写导出数据、配置、stdout/stderr、导入后的 CSV。

流程：

1. 从当前 `MarketDatasetBundle` 导出 PGPortfolio 输入数据。
2. 生成 PGPortfolio 配置。
3. 仅当 `repo_path.resolve()` 位于白名单目录内时，才允许 subprocess 调用外部环境。
4. 收集外部 run 输出。
5. 转换为平台统一 daily outputs。
6. 用当前平台 metrics/statistics 重新计算最终指标。

白名单外流程：

- 不执行 subprocess。
- child run 写 `skipped_out_of_scope`。
- 仅允许导入用户已经在外部手动跑出的结果文件。

关键设计：

- 外部原版训练/回测可能无法完全复用当前 `CostModel`、`ConstraintManager`、`next_open` 执行。
- 转换后的 daily outputs 是 external result import，不等价于平台 `BacktestEngine` 同口径回测，只能进入 external reference 表。
- 因此结果表必须区分：
  - `evaluation_protocol="pgportfolio_original_external"`
  - `cost_model_shared=false` 或仅标记转换后统一评估范围。
  - `constraint_protocol_shared=false`
  - `rankable_in_unified_table=false`
- 若要公平同口径比较，应优先使用 `pgportfolio_eiie_native`。

外部导入最小 schema：

```text
date
nav
net_return
weights 或 per-asset weight columns
```

导入规则：

- 若外部结果缺少成本拆分，必须标记 `cost_model_shared=false`，numeric daily cost 字段写 `NaN` 或保持缺省；`cost_availability="not_available"` 只写入 manifest / comparison / training summary，不得把字符串写入日度数值 schema。
- 若外部结果缺少逐资产权重，导入状态为 `failed_missing_weights`，不能进入 comparison。
- date 必须能映射到当前 test split；无法对齐时状态为 `failed_date_alignment`。
- 权重列必须能一一映射到当前 test asset universe：
  - 缺少任一 test asset 权重列时，状态为 `failed_missing_asset_weights`。
  - 出现额外资产列时，默认状态为 `failed_extra_asset_weights`；若显式允许忽略额外列，必须在 manifest 记录 `ignored_external_weight_columns`。
  - `cash` / `cash_weight` 列默认不允许；若外部结果含现金列，状态为 `failed_cash_weight_column`，除非实验协议显式声明统一平台也允许现金资产。
  - 每日权重和容差默认 `abs(sum(weights) - 1.0) <= 1e-6`；超出容差状态为 `failed_weight_sum_tolerance`。
  - 缺失值、负权重或不可用资产非零权重不得自动归一化；必须失败并记录原因。

失败规则：

- 外部依赖缺失：status = `skipped_external_dependency_missing`，不能写 success。
- 外部进程非 0 退出：status = `failed`。
- 结果 CSV 空：status = `failed`。

关键测试：

- mock subprocess 成功路径，确认导出/导入 schema。
- mock subprocess 失败路径，确认父 run failed 或 child skipped。
- `external_pgportfolio.enabled=false` 时不影响默认测试。

## 6. 输出 schema 改造

新增字段只写入以下产物：

- `metrics/baseline_comparison.csv`
- `metrics/main_comparison.csv`
- `logs/run_manifest.json`
- 新增 `logs/baseline_training_summary.csv`
- 新增 `logs/baseline_training_history.csv`

禁止污染冻结日度 CSV schema：

- 不向 `daily_returns.csv`、`daily_weights.csv`、`daily_turnover.csv`、`daily_rebalance.csv`、`daily_costs.csv` 添加训练元数据列。
- 日度 CSV 只保留既有日度字段；训练口径、许可证、外部 runner、checkpoint 路径写入 comparison / manifest / training summary。

所有 baseline comparison / training summary row 新增字段：

```text
model_name
baseline_family
training_algorithm
rl_training
platform_native_rl_training
proxy_training
external_original_implementation
source_code_vendored
license
data_protocol
execution_protocol
evaluation_protocol
cost_protocol
cost_model_shared
cost_availability
constraint_protocol
constraint_protocol_shared
rankable_in_unified_table
train_status
checkpoint_best_path
checkpoint_last_path
evaluated_checkpoint_path
training_steps
validation_metric
training_summary
training_history_flat_path
```

推荐取值：

| baseline | baseline_family | training_algorithm | rl_training | platform_native_rl_training | proxy_training | external_original_implementation | rankable_in_unified_table |
| --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight / traditional_* | traditional | deterministic_strategy / optimizer_strategy | false | false | false | false | true |
| ppo_proxy | neural_proxy | supervised_execution_aligned_proxy | false | false | true | false | false |
| ppo_native | native_rl | ppo_clipped_gae | true | true | false | false | true |
| cnn_ppo_native | native_rl | ppo_clipped_gae | true | true | false | false | true |
| bernoulli_gated_ppo_native | native_rl | bernoulli_gated_ppo_on_policy | true | true | false | false | true |
| dqn_template_native | native_rl | double_dqn_template_selector | true | true | false | false | true |
| eiie_native | native_rl | eiie_policy_gradient_pvm | true | true | false | false | true |
| pgportfolio_eiie_native | native_rl | pgportfolio_eiie_osbl | true | true | false | false | true |
| pgportfolio_original_external | external_original | pgportfolio_original | true | false | false | true | false |

## 7. 实验公平性协议

同一张论文表内只能混合比较满足同一协议的 baseline。

推荐分表：

1. Unified-platform fair comparison：
   - main model
   - traditional baselines
   - native RL baselines
   - pgportfolio_eiie_native
   - 全部使用当前平台数据、成本、约束、回测。

2. Historical/original implementation reference：
   - pgportfolio_original_external
   - 标注为 external reference，不与统一平台结果直接排名。

3. Neural proxy diagnostics：
   - 当前 proxy baselines
   - 仅用于工程 smoke、消融参考、快速 sanity check。

HPO 要求：

- native RL baselines 与 main model 同 budget，budget 单位必须固定并写入 manifest：
  - `env_steps`
  - `gradient_updates`
  - `hpo_trials`
  - `random_seeds`
  - `validation_metric`
  - `selection_rule`
- 主表公平协议：每个 trainable model 必须使用完全相同的 `env_steps`、`gradient_updates`、`hpo_trials`、`random_seeds`、`validation_metric`、`selection_rule`。
- 非同 budget 结果只能进入 robustness / appendix / engineering diagnostics，不能进入主排名或统一统计检验。
- selection rule：按 validation primary metric 选择 best trial，test 只跑 best trial；median/worst validation trial 可作为附表，不参与主排名。
- proxy baseline 不参与 native RL HPO 排名，除非单独作为 proxy experiment。
- external PGPortfolio 可单独给固定 budget，并记录 wall time 与 trial 数。

## 8. 实施顺序

### 阶段 1：命名和 schema 去误导

目标：

- 当前 deep baselines 改名为 proxy。
- manifest 和 comparison CSV 写清 training family。

改动：

- `src/experiments/registry.py`
- `src/experiments/pipeline.py`
- `src/baselines/deep_training.py`
- `tests/test_baseline_output_schema.py`

验收：

- 旧名称 alias 可用。
- proxy row 中 `rl_training=false`、`platform_native_rl_training=false`、`proxy_training=true`。
- native 表不包含 proxy baseline。

### 阶段 2：PPO / CNN-PPO native

目标：

- 最小原生 PPO baseline 闭环。

改动：

- `src/baselines/native_ppo.py`
- `src/experiments/pipeline.py`
- `tests/test_native_baselines.py`

验收：

- rollout 非空。
- PPO loss 更新参数。
- test daily outputs 由 `BacktestEngine` 生成且非空。
- checkpoint 存在。

### 阶段 3：DQN-only native

目标：

- 真正模板 selector。

改动：

- `src/baselines/native_dqn_template.py`
- `src/baselines/template_portfolios.py`
- `tests/test_native_dqn_template.py`

验收：

- 7 个模板动作可用。
- replay / target / Double DQN 生效。
- 模板 invalid/fallback 规则可测试，不能静默退化为 hold 或 equal weight。

### 阶段 4：Bernoulli-Gated native

目标：

- on-policy gate 训练。

改动：

- `src/baselines/native_bernoulli_gated_ppo.py`
- `src/buffers/rollout_buffer.py`
- `tests/test_native_bernoulli_gate.py`

验收：

- gate log_prob 进入 loss。
- gate entropy 落盘。
- hold action 传 `rebalance=0`。

### 阶段 5：EIIE / PGPortfolio native

目标：

- PyTorch clean-room EIIE + PVM + OSBL。

改动：

- `src/baselines/native_eiie.py`
- `src/baselines/pgportfolio_eiie.py`
- `tests/test_pgportfolio_eiie.py`

验收：

- PVM 状态随日期推进。
- OSBL 只采 train。
- immediate reward loss 可训练。
- final test 必须使用 `BacktestEngine`。

### 阶段 6：PGPortfolio external original

目标：

- 可选作者原版外部 baseline。

改动：

- `src/baselines/external_pgportfolio.py`
- `src/experiments/external_baselines.py`
- `scripts/export_pgportfolio_dataset.py`
- `scripts/import_pgportfolio_results.py`
- `tests/test_external_pgportfolio.py`

验收：

- external disabled 默认不跑。
- mock subprocess 成功后导入统一 outputs，且 manifest 标记 `platform_native_rl_training=false`。
- mock subprocess 失败不伪成功。
- manifest 记录 GPL/external/source_code_vendored=false。

## 9. 最低测试清单

必须新增：

- `test_proxy_baselines_are_not_marked_platform_native_rl`
- `test_evaluated_checkpoint_path_persisted_in_summary`
- `test_native_ppo_updates_parameters_and_writes_daily_outputs`
- `test_native_cnn_ppo_uses_cnn_encoder`
- `test_bernoulli_gate_on_policy_log_prob_has_gradient`
- `test_dqn_template_native_uses_replay_target_double_dqn`
- `test_dqn_template_all_actions_generate_valid_weights`
- `test_dqn_template_invalid_action_records_penalty_or_fallback`
- `test_dqn_invalid_auxiliary_target_has_no_env_transition_bootstrap`
- `test_eiie_native_uses_pvm_previous_weights`
- `test_pgportfolio_eiie_osbl_samples_train_only`
- `test_external_import_weights_align_test_universe`
- `test_external_pgportfolio_disabled_by_default`
- `test_external_pgportfolio_subprocess_failure_fails_child_run`
- `test_external_pgportfolio_is_not_rankable_in_unified_table`
- `test_baseline_comparison_separates_proxy_native_external_rows`
- `test_hpo_equal_budget_excludes_proxy_when_native_only`

建议回归：

```bash
.venv/bin/python -m pytest \
  tests/test_config.py \
  tests/test_baselines.py \
  tests/test_baseline_output_schema.py \
  tests/test_native_baselines.py \
  tests/test_experiments.py \
  tests/test_outputs.py
```

长跑回归：

```bash
.venv/bin/python -m pytest
```

## 10. 风险与约束

许可证风险：

- PGPortfolio 是 GPL-3.0；不要复制源码进当前仓库。
- external runner 只记录路径、命令、结果，不分发 GPL 源码。

工程风险：

- PGPortfolio 原版依赖 TensorFlow 1.x/tflearn，默认 CI 不应安装。
- external result 可能与统一平台执行口径不同，必须单独标注。

公平性风险：

- proxy、native、external 三类 baseline 不能混成一个无说明排行榜。
- 论文主表应以 unified-platform fair comparison 为准。

数据泄漏风险：

- native baselines 的 scaler/PCA/feature selector 仍只能 train fit。
- OSBL sampler 不能采 validation/test。
- PVM 初始化和更新不能读取未来组合权重。

## 11. 最终验收标准

该问题关闭条件：

- `baseline_comparison` 能同时跑 traditional、proxy、native RL、external disabled。
- 至少以下 native baselines 可训练，并通过 `BacktestEngine` 生成非空 daily outputs：
  - `ppo_native`
  - `cnn_ppo_native`
  - `dqn_template_native`
  - `eiie_native`
  - `pgportfolio_eiie_native`
- `pgportfolio_original_external` 在 mock 环境下通过；真实外部环境可选。
- `baseline_comparison.csv` 能清楚区分 `neural_proxy`、`native_rl`、`external_original`。
- `main_comparison.csv` 和 `statistics_summary.csv` 必须排除 `rankable_in_unified_table=false` 的 external/proxy 结果，避免进入主排名或 paired significance test。
- HPO fair protocol 不把 proxy baseline 伪装成 native trainable model。
- 文档和论文结果表不再把当前 proxy baselines 称为原生 PPO/DQN/EIIE。

## 12. 本轮评审建议采纳情况

已采纳 A/B 的全部有效建议：

- DQN-only 默认动作空间改为原需求 7 模板，不含 hold；可学习 hold 需独立命名 `dqn_template_with_hold_native`。
- test daily outputs 强制由 `BacktestEngine` 生成，native runner 不允许自行拼接冻结日度 CSV。
- `BaselineRunResult` 和 runner 协议补充 `fit`、`evaluate_with_backtest`、checkpoint、training summary/history、paired returns 字段。
- `native_rl_training` 拆为 `rl_training` 与 `platform_native_rl_training`，external PGPortfolio 不再标为平台内 native。
- EIIE / PGPortfolio loss 明确区分训练优化项与完整环境 net return，禁止使用 execution-only 成本字段。
- DQN template Q 网络补充 per-template action feature 方案、state-only 备选方案、invalid/fallback 规则。
- 训练元数据仅写入 comparison / manifest / training summary，不污染冻结日度 CSV。
- external PGPortfolio 默认只允许白名单内 repo，白名单外标记 `out_of_scope` 且默认不执行。
- PPO native 明确受 `RebalanceScheduler` 约束，不绕过统一执行核心。
- Bernoulli-Gated PPO 明确 gate=hold 时不把 hold 收益归因给 PPO candidate。
- external PGPortfolio schema 补齐 `evaluation_protocol`、`execution_protocol`、`cost_model_shared`、`constraint_protocol_shared`、`rankable_in_unified_table`。
- HPO budget 明确定义为 `env_steps`、`gradient_updates`、`hpo_trials`、`random_seeds`、`validation_metric`、`selection_rule`。

本轮新增收紧项：

- `TrainingResult.training_history` 固定最小扁平列，避免嵌套 dict 破坏 plotting。
- native test 必须 load best validation checkpoint，再经 strategy adapter 调 `BacktestEngine.run()`。
- DQN invalid action 默认使用 `invalid_action_penalty=1.0`，训练交互阶段 mask invalid action，penalty 只作为 auxiliary Q target；评估 fallback 固定为 `equal_weight`。
- `pre_execution_return_t` 只可作为权重漂移/stop-gradient target，不得进入任何模型输入。
- external PGPortfolio 白名单外只允许 `skipped_out_of_scope` 和离线导入，不允许平台主动执行。
- external daily outputs 明确为 imported external results，不等价于平台同口径回测。
- HPO 主表要求完全同 budget；非同 budget 只能进 appendix/diagnostics。
- `main_comparison.csv` 与 `statistics_summary.csv` 必须排除不可排名的 external/proxy rows。

本轮二次收紧项：

- `BaselineRunResult` 补充 `evaluated_checkpoint_path` 字段。
- `training_history.validation_metric` 允许非验证 step 为空，但 completed run 至少要有一行 finite validation metric。
- DQN invalid action 在训练交互阶段必须被 mask；invalid penalty 只作为 auxiliary Q target，不能生成虚假 env transition。
- external subprocess 仅允许白名单内 `repo_path.resolve()`；白名单外只允许 skipped 和离线 import。
- external 成本缺失时 numeric daily cost 写 `NaN` 或缺省，`cost_availability=not_available` 只写元数据。
- baseline comparison 明确 traditional baseline 元数据行：非 RL、非 proxy、可进入统一主表。

本轮三次收紧项：

- `evaluated_checkpoint_path` 同步加入落盘字段清单，避免只存在于内存结果。
- DQN invalid auxiliary target 明确 `bootstrap_mask=0`，不参与真实 transition bootstrap。
- 顶部 PGPortfolio external 摘要改为仅白名单内可 subprocess，白名单外只能 import 用户手动结果。
- external weights 导入必须严格对齐 test asset universe，缺资产、额外资产、现金列、权重和越界均有失败状态。
- 测试清单补充 checkpoint path 落盘、invalid auxiliary no-bootstrap、external weights 对齐三项。
