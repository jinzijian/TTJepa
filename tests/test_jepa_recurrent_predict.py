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
