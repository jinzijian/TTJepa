import torch


def _depth_values_like(pred_err):
    return torch.arange(
        1,
        pred_err.size(0) + 1,
        device=pred_err.device,
        dtype=torch.long,
    ).view(-1, *([1] * (pred_err.ndim - 1)))


def _normalize_feature_weights(feature_weights, feature_dim, device, dtype):
    weights = feature_weights.detach().to(device=device, dtype=dtype).flatten()
    if weights.numel() != feature_dim:
        raise ValueError(
            "feature_weights must have one value per prediction feature dimension"
        )
    weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights = weights.clamp_min(0.0)
    return weights / weights.mean().clamp_min(1e-8)


def _prediction_error(
    preds_all,
    target,
    *,
    whiten=False,
    whiten_eps=1e-6,
    feature_weights=None,
):
    delta = preds_all.detach() - target.detach().unsqueeze(0)
    if target.ndim < 2:
        raise ValueError("halt targets require a feature dimension")
    feature_dim = target.size(-1)
    if whiten:
        flat_target = target.detach().reshape(-1, feature_dim)
        # Batch whitening keeps the oracle from being dominated by high-variance
        # latent directions without requiring precomputed dataset statistics.
        scale = flat_target.std(dim=0, unbiased=False).clamp_min(float(whiten_eps))
        view_shape = [1] * (delta.ndim - 1) + [feature_dim]
        delta = delta / scale.view(*view_shape)

    squared_error = delta.pow(2)
    if feature_weights is not None:
        weights = _normalize_feature_weights(
            feature_weights,
            feature_dim,
            squared_error.device,
            squared_error.dtype,
        )
        view_shape = [1] * (squared_error.ndim - 1) + [feature_dim]
        squared_error = squared_error * weights.view(*view_shape)
    return squared_error.mean(dim=-1)


def build_continue_targets(
    preds_all,
    target,
    *,
    mode="improvement",
    residuals=None,
    min_improvement=0.0,
    min_relative_improvement=0.0,
    error_threshold=None,
    residual_threshold=None,
    oracle_relative_tolerance=0.0,
    oracle_abs_tolerance=0.0,
    min_depth=1,
    whiten_eps=1e-6,
    feature_weights=None,
):
    """Build per-depth continue labels for recurrent predictor halting.

    The returned target has shape (K, B, T). A value of 1 means "run one more
    refinement step after this depth"; 0 means "this depth is good enough".
    """
    use_whitened_error = mode == "oracle_depth_whitened"
    use_probe_weighted_error = mode == "oracle_depth_probe_weighted"
    if use_probe_weighted_error and feature_weights is None:
        raise ValueError(
            "feature_weights must be set when mode='oracle_depth_probe_weighted'"
        )
    label_mode = (
        "oracle_depth"
        if use_whitened_error or use_probe_weighted_error
        else mode
    )
    pred_err = _prediction_error(
        preds_all,
        target,
        whiten=use_whitened_error,
        whiten_eps=whiten_eps,
        feature_weights=feature_weights,
    )
    K = pred_err.size(0)
    if not 1 <= int(min_depth) <= K:
        raise ValueError("min_depth must satisfy 1 <= min_depth <= number of depths")

    continue_target = torch.zeros_like(pred_err)
    depth_values = _depth_values_like(pred_err)

    if label_mode == "improvement":
        if K > 1:
            improvement = pred_err[:-1] - pred_err[1:]
            continue_target[:-1] = (improvement > float(min_improvement)).float()
    elif label_mode == "relative_improvement":
        if K > 1:
            improvement = pred_err[:-1] - pred_err[1:]
            relative = improvement / pred_err[:-1].clamp_min(1e-8)
            continue_target[:-1] = (
                relative > float(min_relative_improvement)
            ).float()
    elif label_mode == "error_threshold":
        if error_threshold is None:
            raise ValueError(
                "error_threshold must be set when mode='error_threshold'"
            )
        continue_target = (pred_err > float(error_threshold)).float()
    elif label_mode == "residual_threshold":
        if residual_threshold is None:
            raise ValueError(
                "residual_threshold must be set when mode='residual_threshold'"
            )
        if residuals is None:
            raise ValueError("residuals must be set when mode='residual_threshold'")
        continue_target = (residuals.detach() > float(residual_threshold)).float()
    elif label_mode == "oracle_depth":
        allowed = depth_values >= int(min_depth)
        inf = torch.full_like(pred_err, float("inf"))
        allowed_pred_err = torch.where(allowed.expand_as(pred_err), pred_err, inf)
        best_err = allowed_pred_err.min(dim=0).values
        threshold = (
            best_err * (1.0 + float(oracle_relative_tolerance))
            + float(oracle_abs_tolerance)
        )
        can_halt = pred_err <= threshold.unsqueeze(0)
        can_halt = can_halt & allowed
        sentinel = torch.full_like(depth_values.expand_as(pred_err), K + 1)
        first_good_depth = torch.where(
            can_halt,
            depth_values.expand_as(pred_err),
            sentinel,
        ).min(dim=0).values.clamp(max=K)
        continue_target = (depth_values < first_good_depth.unsqueeze(0)).float()
    else:
        raise ValueError(f"Unsupported recurrent.halt_label_mode={mode}")

    if min_depth > 1 and K > 1:
        continue_target = torch.where(
            depth_values < int(min_depth),
            torch.ones_like(continue_target),
            continue_target,
        )
    continue_target[-1] = 0.0
    halt_depth = (continue_target > 0.5).sum(dim=0) + 1
    return continue_target, halt_depth
