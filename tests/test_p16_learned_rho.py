from copy import deepcopy

import torch

from src.agents.constrained_actor_critic_agent import ConstrainedActorCriticAgent, agent_config_from_mapping
from src.baselines.deep_training import DeepBaselineTrainingBatch
from src.config import DEFAULT_CONFIG
from src.models.risk_aware_graph_transformer import RA_GT_RCPO_MODEL_NAME, build_risk_aware_graph_transformer


def test_rho_policy_head_shape():
    config = _config()
    model = build_risk_aware_graph_transformer(config, model_name=RA_GT_RCPO_MODEL_NAME)
    output = model(
        torch.zeros((2, 1, 3, 2), dtype=torch.float32),
        torch.full((2, 2), 0.5, dtype=torch.float32),
        torch.ones((2, 2), dtype=torch.bool),
    )

    assert output.rho_logits.shape == (2, 5)
    assert output.rho_probs.shape == (2, 5)
    assert output.rho.shape == (2,)


def test_rho_policy_participates_in_loss():
    config = _config()
    model = build_risk_aware_graph_transformer(config, model_name=RA_GT_RCPO_MODEL_NAME)
    agent = ConstrainedActorCriticAgent(
        model,
        config=agent_config_from_mapping(config, section=config["ra_gt_rcpo"]),
        device=torch.device("cpu"),
    )
    before = [param.detach().clone() for param in agent.model.rho_head.parameters()]

    history, stats = agent.train_offline(_training_batch())

    after = [param.detach() for param in agent.model.rho_head.parameters()]
    assert int(stats["gradient_updates"]) > 0
    assert not history.empty
    assert any(not torch.allclose(left, right) for left, right in zip(before, after, strict=True))


def test_high_entropy_eval_uses_expected_rho_instead_of_zero_argmax():
    config = _config()
    model = build_risk_aware_graph_transformer(config, model_name=RA_GT_RCPO_MODEL_NAME)
    with torch.no_grad():
        for parameter in model.rho_head.parameters():
            parameter.zero_()
    agent = ConstrainedActorCriticAgent(
        model,
        config=agent_config_from_mapping(config, section=config["ra_gt_rcpo"]),
        device=torch.device("cpu"),
    )

    action = agent.select_action(
        torch.zeros((1, 3, 2), dtype=torch.float32).numpy(),
        torch.full((2,), 0.5, dtype=torch.float32).numpy(),
        torch.ones((2,), dtype=torch.bool).numpy(),
    )

    assert action["rho_eval_used_expected"] is True
    assert action["rho_action_index"] == 2
    assert action["raw_rho"] == 0.5


def _config():
    config = deepcopy(DEFAULT_CONFIG)
    config["n_assets"] = 2
    config["n_features"] = 1
    config["window_size"] = 3
    config["ra_gt_rcpo"]["model_dim"] = 16
    config["ra_gt_rcpo"]["attention_heads"] = 2
    config["ra_gt_rcpo"]["batch_size"] = 2
    config["ra_gt_rcpo"]["rho_policy"] = "straight_through_gumbel_softmax_v1"
    config["cost_model"]["market_impact_enabled"] = False
    return config


def _training_batch():
    market_image = torch.tensor(
        [
            [[[0.00, 0.00], [0.01, -0.01], [0.02, 0.00]]],
            [[[0.00, 0.00], [-0.01, 0.01], [0.00, 0.02]]],
            [[[0.01, 0.00], [0.02, -0.02], [0.01, 0.01]]],
            [[[0.00, 0.01], [0.00, 0.02], [-0.01, 0.01]]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones((4, 2), dtype=torch.bool)
    current = torch.full((4, 2), 0.5, dtype=torch.float32)
    equal = current.clone()
    future = torch.tensor([[0.01, -0.005], [-0.002, 0.01], [0.015, -0.01], [0.0, 0.012]], dtype=torch.float32)
    return DeepBaselineTrainingBatch(market_image, mask, current, equal, future)
