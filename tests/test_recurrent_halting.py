import torch

from recurrent_halting import build_continue_targets


def test_oracle_depth_targets_continue_until_first_near_best_depth():
    target = torch.zeros(1, 1, 1)
    preds_all = torch.tensor(
        [
            [[[3.0]]],
            [[[2.0]]],
            [[[0.1]]],
            [[[0.0]]],
        ]
    )

    continue_target, halt_depth = build_continue_targets(
        preds_all,
        target,
        mode="oracle_depth",
        oracle_abs_tolerance=0.02,
    )

    assert continue_target[:, 0, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert halt_depth.item() == 3


def test_oracle_depth_targets_respect_min_depth():
    target = torch.zeros(1, 1, 1)
    preds_all = torch.tensor(
        [
            [[[0.0]]],
            [[[1.0]]],
            [[[2.0]]],
        ]
    )

    continue_target, halt_depth = build_continue_targets(
        preds_all,
        target,
        mode="oracle_depth",
        min_depth=2,
    )

    assert continue_target[:, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert halt_depth.item() == 2


def test_whitened_oracle_depth_weights_low_variance_latent_directions():
    target = torch.tensor(
        [
            [[100.0, 0.0]],
            [[-100.0, 2.0]],
        ]
    )
    preds_all = torch.stack(
        [
            target + torch.tensor([[[1.0, 1.0]]]),
            target + torch.tensor([[[10.0, 0.0]]]),
        ],
        dim=0,
    )

    raw_continue, raw_halt_depth = build_continue_targets(
        preds_all,
        target,
        mode="oracle_depth",
    )
    whitened_continue, whitened_halt_depth = build_continue_targets(
        preds_all,
        target,
        mode="oracle_depth_whitened",
    )

    assert raw_continue[:, 0, 0].tolist() == [0.0, 0.0]
    assert raw_halt_depth[0, 0].item() == 1
    assert whitened_continue[:, 0, 0].tolist() == [1.0, 0.0]
    assert whitened_halt_depth[0, 0].item() == 2


def test_probe_weighted_oracle_depth_prioritizes_task_latent_directions():
    target = torch.zeros(1, 1, 2)
    preds_all = torch.tensor(
        [
            [[[0.0, 1.0]]],
            [[[1.5, 0.0]]],
        ]
    )

    raw_continue, raw_halt_depth = build_continue_targets(
        preds_all,
        target,
        mode="oracle_depth",
    )
    weighted_continue, weighted_halt_depth = build_continue_targets(
        preds_all,
        target,
        mode="oracle_depth_probe_weighted",
        feature_weights=torch.tensor([0.1, 10.0]),
    )

    assert raw_continue[:, 0, 0].tolist() == [0.0, 0.0]
    assert raw_halt_depth.item() == 1
    assert weighted_continue[:, 0, 0].tolist() == [1.0, 0.0]
    assert weighted_halt_depth.item() == 2


def test_improvement_targets_match_next_step_error_reduction():
    target = torch.zeros(1, 1, 1)
    preds_all = torch.tensor(
        [
            [[[3.0]]],
            [[[1.0]]],
            [[[2.0]]],
        ]
    )

    continue_target, halt_depth = build_continue_targets(
        preds_all,
        target,
        mode="improvement",
        min_improvement=0.0,
    )

    assert continue_target[:, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert halt_depth.item() == 2
