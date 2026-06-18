import torch
from torch import nn

from jepa import JEPA
from module import RecurrentARPredictor


def test_jepa_predict_return_all_projects_each_depth():
    predictor = RecurrentARPredictor(
        num_frames=3,
        base_depth=1,
        refine_depth=1,
        max_depth=4,
        heads=2,
        mlp_dim=64,
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )
    model = JEPA(
        encoder=nn.Identity(),
        predictor=predictor,
        action_encoder=nn.Identity(),
        pred_proj=nn.Linear(16, 12),
    )

    emb = torch.randn(2, 3, 16)
    act_emb = torch.randn(2, 3, 16)
    out = model.predict(emb, act_emb, return_all=True, predictor_depth=3)

    assert out["pred"].shape == (2, 3, 12)
    assert out["preds"].shape == (3, 2, 3, 12)
    assert out["residuals"].shape == (3, 2, 3)
    assert out["continue_logits"].shape == (3, 2, 3)
    assert out["depth_used"].shape == (2, 3)


def test_jepa_predict_tensor_path_stays_compatible():
    predictor = RecurrentARPredictor(
        num_frames=3,
        base_depth=1,
        refine_depth=1,
        max_depth=4,
        heads=2,
        mlp_dim=64,
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )
    model = JEPA(
        encoder=nn.Identity(),
        predictor=predictor,
        action_encoder=nn.Identity(),
        pred_proj=nn.Linear(16, 12),
    )

    emb = torch.randn(2, 3, 16)
    act_emb = torch.randn(2, 3, 16)
    pred = model.predict(emb, act_emb)

    assert pred.shape == (2, 3, 12)


def test_jepa_predict_residual_halting_return_all():
    predictor = RecurrentARPredictor(
        num_frames=3,
        base_depth=1,
        refine_depth=1,
        max_depth=4,
        heads=2,
        mlp_dim=64,
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )
    model = JEPA(
        encoder=nn.Identity(),
        predictor=predictor,
        action_encoder=nn.Identity(),
        pred_proj=nn.Linear(16, 12),
    )

    emb = torch.randn(2, 3, 16)
    act_emb = torch.randn(2, 3, 16)
    out = model.predict(
        emb,
        act_emb,
        return_all=True,
        predictor_depth=4,
        halt_mode="residual",
        halt_eps=1e9,
        min_depth=2,
    )

    assert out["pred"].shape == (2, 3, 12)
    assert out["preds"].shape == (4, 2, 3, 12)
    assert torch.equal(out["depth_used"], torch.full((2, 3), 2))
    assert torch.allclose(out["pred"], out["preds"][1])


def test_jepa_predict_learned_halting_return_all():
    predictor = RecurrentARPredictor(
        num_frames=3,
        base_depth=1,
        refine_depth=1,
        max_depth=4,
        heads=2,
        mlp_dim=64,
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )
    torch.nn.init.zeros_(predictor.continue_head.weight)
    torch.nn.init.constant_(predictor.continue_head.bias, -10.0)
    model = JEPA(
        encoder=nn.Identity(),
        predictor=predictor,
        action_encoder=nn.Identity(),
        pred_proj=nn.Linear(16, 12),
    )

    emb = torch.randn(2, 3, 16)
    act_emb = torch.randn(2, 3, 16)
    out = model.predict(
        emb,
        act_emb,
        return_all=True,
        predictor_depth=4,
        halt_mode="learned",
        halt_threshold=0.5,
        min_depth=2,
    )

    assert out["pred"].shape == (2, 3, 12)
    assert torch.equal(out["depth_used"], torch.full((2, 3), 2))
    assert torch.allclose(out["pred"], out["preds"][1])


def test_jepa_depth_stats_accumulate_without_rollout_storage():
    predictor = RecurrentARPredictor(
        num_frames=3,
        base_depth=1,
        refine_depth=1,
        max_depth=4,
        heads=2,
        mlp_dim=64,
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )
    model = JEPA(
        encoder=nn.Identity(),
        predictor=predictor,
        action_encoder=nn.Identity(),
        pred_proj=nn.Linear(16, 12),
    )

    model._record_depth_stats(torch.tensor([[1, 2, 2], [4, 4, 4]]), max_depth=4)
    model._record_depth_stats(torch.tensor([[2, 3]]), max_depth=4)
    stats = model.get_depth_stats()

    assert stats["mean_depth"] == 22 / 8
    assert stats["num_predictions"] == 8
    assert stats["min_depth"] == 1
    assert stats["max_depth"] == 4
    assert stats["depth_histogram"] == {"1": 1, "2": 3, "3": 1, "4": 3}
