"""Traditional baseline strategy contracts."""

from src.baselines.base_strategy import BaseStrategy, TraditionalStrategyBase
from src.baselines.buy_and_hold import BuyAndHoldStrategy
from src.baselines.cnn_ppo_baseline import CNNPPOBaselineStrategy
from src.baselines.equal_weight import EqualWeightStrategy
from src.baselines.external_pgportfolio import PGPORTFOLIO_EXTERNAL_MODEL_NAME, external_pgportfolio_summary
from src.baselines.fixed_ratio import FixedRatioStrategy
from src.baselines.hrp import HRPStrategy
from src.baselines.hybrid_dqn_optimizer_reimplementation import (
    HybridDQNOptimizerEqualWeightStrategy,
    HybridDQNOptimizerMarkowitzMeanVarianceStrategy,
    HybridDQNOptimizerMinimumVarianceStrategy,
    HybridDQNOptimizerRiskParityStrategy,
    HybridDQNOptimizerSharpeMaximizationStrategy,
)
from src.baselines.inverse_volatility import InverseVolatilityStrategy
from src.baselines.markowitz import (
    MarkowitzMaxSharpeStrategy,
    MarkowitzMeanVarianceStrategy,
    MarkowitzMinVarianceStrategy,
    MarkowitzStrategy,
)
from src.baselines.minimum_drawdown import MinimumDrawdownStrategy
from src.baselines.momentum import MomentumStrategy
from src.baselines.native_bernoulli_gated_ppo import NativeBernoulliGatedPPOBaselineStrategy
from src.baselines.native_dqn_template import NativeDQNTemplateStrategy
from src.baselines.native_eiie import NativeEIIEStrategy
from src.baselines.native_ppo import NativeCNNPPOBaselineStrategy, NativePPOBaselineStrategy
from src.baselines.pgportfolio_eiie import PGPortfolioEIIEStrategy
from src.baselines.ppo_dqn_hierarchical_reimplementation import PPODQNHierarchicalReimplementationStrategy
from src.baselines.ppo_baseline import PPOBaselineStrategy
from src.baselines.risk_evaluation import RiskEvaluationStrategy
from src.baselines.risk_parity import RiskParityStrategy

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "CNNPPOBaselineStrategy",
    "EqualWeightStrategy",
    "PGPORTFOLIO_EXTERNAL_MODEL_NAME",
    "FixedRatioStrategy",
    "HRPStrategy",
    "HybridDQNOptimizerEqualWeightStrategy",
    "HybridDQNOptimizerMarkowitzMeanVarianceStrategy",
    "HybridDQNOptimizerMinimumVarianceStrategy",
    "HybridDQNOptimizerRiskParityStrategy",
    "HybridDQNOptimizerSharpeMaximizationStrategy",
    "InverseVolatilityStrategy",
    "MarkowitzMaxSharpeStrategy",
    "MarkowitzMeanVarianceStrategy",
    "MarkowitzMinVarianceStrategy",
    "MarkowitzStrategy",
    "MinimumDrawdownStrategy",
    "MomentumStrategy",
    "NativeBernoulliGatedPPOBaselineStrategy",
    "NativeDQNTemplateStrategy",
    "NativeEIIEStrategy",
    "NativeCNNPPOBaselineStrategy",
    "NativePPOBaselineStrategy",
    "PGPortfolioEIIEStrategy",
    "PPODQNHierarchicalReimplementationStrategy",
    "PPOBaselineStrategy",
    "external_pgportfolio_summary",
    "RiskEvaluationStrategy",
    "RiskParityStrategy",
    "TraditionalStrategyBase",
]
