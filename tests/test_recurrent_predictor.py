import torch

from module import RecurrentARPredictor


def make_predictor(max_depth=4):
    return RecurrentARPredictor(
        num_frames=3,
        base_depth=1,
        refine_depth=1,
        max_depth=max_depth,
        heads=2,
        mlp_dim=64,
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        dim_head=8,
        dropout=0.0,
        emb_dropout=0.0,
    )


def test_recurrent_predictor_shape():
    model = make_predictor(max_depth=4)
    x = torch.randn(4, 3, 16)
    c = torch.randn(4, 3, 16)

    y = model(x, c)
    assert y.shape == (4, 3, 16)


def test_recurrent_predictor_return_all_shape():
    model = make_predictor(max_depth=4)
    x = torch.randn(4, 3, 16)
    c = torch.randn(4, 3, 16)

    out = model(x, c, max_depth=3, return_all=True)

    assert out["pred"].shape == (4, 3, 16)
    assert out["preds"].shape == (3, 4, 3, 16)
    assert out["residuals"].shape == (3, 4, 3)
    assert out["depth_used"].shape == (4, 3)
    assert torch.equal(out["depth_used"], torch.full((4, 3), 3))
