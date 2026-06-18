"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.reset_depth_stats()

    def reset_depth_stats(self):
        self._depth_stats_sum = None
        self._depth_stats_count = None
        self._depth_stats_hist = None

    def _ensure_depth_stats(self, device, max_depth):
        if self._depth_stats_sum is None:
            self._depth_stats_sum = torch.zeros((), device=device, dtype=torch.float64)
            self._depth_stats_count = torch.zeros((), device=device, dtype=torch.long)
            self._depth_stats_hist = torch.zeros(
                max_depth + 1, device=device, dtype=torch.long
            )
        elif self._depth_stats_hist.numel() <= max_depth:
            hist = torch.zeros(max_depth + 1, device=device, dtype=torch.long)
            hist[: self._depth_stats_hist.numel()] = self._depth_stats_hist
            self._depth_stats_hist = hist

    def _record_depth_stats(self, depth_used, max_depth=None):
        with torch.no_grad():
            depth_used = depth_used.detach().reshape(-1).to(torch.long)
            if depth_used.numel() == 0:
                return
            max_depth = int(
                max_depth
                or getattr(self.predictor, "max_depth", None)
                or depth_used.max().item()
            )
            self._ensure_depth_stats(depth_used.device, max_depth)
            self._depth_stats_sum.add_(depth_used.to(torch.float64).sum())
            self._depth_stats_count.add_(depth_used.numel())
            hist = torch.bincount(depth_used, minlength=self._depth_stats_hist.numel())
            self._depth_stats_hist[: hist.numel()].add_(hist)

    def get_depth_stats(self):
        if self._depth_stats_count is None:
            return {}

        count = int(self._depth_stats_count.item())
        if count == 0:
            return {}

        hist = self._depth_stats_hist.detach().cpu()
        nonzero = torch.nonzero(hist, as_tuple=False).flatten().tolist()
        nonzero = [int(depth) for depth in nonzero if depth > 0]
        depth_histogram = {str(depth): int(hist[depth].item()) for depth in nonzero}
        depth_fraction = {
            str(depth): float(hist[depth].item() / count) for depth in nonzero
        }

        return {
            "mean_depth": float(self._depth_stats_sum.item() / count),
            "num_predictions": count,
            "min_depth": min(nonzero) if nonzero else None,
            "max_depth": max(nonzero) if nonzero else None,
            "depth_histogram": depth_histogram,
            "depth_fraction": depth_fraction,
        }

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(
        self,
        emb,
        act_emb,
        return_all=False,
        predictor_depth=None,
        halt_mode="none",
        halt_eps=None,
        halt_threshold=0.5,
        min_depth=1,
    ):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        predictor_kwargs = {}
        if (
            return_all
            or predictor_depth is not None
            or halt_mode != "none"
            or halt_eps is not None
            or halt_threshold != 0.5
            or min_depth != 1
        ):
            predictor_kwargs = {
                "max_depth": predictor_depth,
                "return_all": return_all,
                "halt_mode": halt_mode,
                "halt_eps": halt_eps,
                "halt_threshold": halt_threshold,
                "min_depth": min_depth,
            }

        try:
            out = self.predictor(emb, act_emb, **predictor_kwargs)
        except TypeError as err:
            if predictor_kwargs:
                raise TypeError(
                    "The configured predictor does not support recurrent "
                    "prediction options."
                ) from err
            raise

        if isinstance(out, dict):
            out = dict(out)
            preds = out["preds"]
            K, B, T, _ = preds.shape
            preds = self.pred_proj(rearrange(preds, "k b t d -> (k b t) d"))
            out["preds"] = rearrange(preds, "(k b t) d -> k b t d", k=K, b=B, t=T)
            if "depth_used" in out:
                depth_idx = out["depth_used"].clamp(1, K) - 1
                preds_by_token = rearrange(out["preds"], "k b t d -> b t k d")
                gather_idx = depth_idx[:, :, None, None].expand(
                    B, T, 1, preds_by_token.size(-1)
                )
                out["pred"] = preds_by_token.gather(2, gather_idx).squeeze(2)
            else:
                out["pred"] = out["preds"][-1]
            return out

        preds = self.pred_proj(rearrange(out, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(
        self,
        info,
        action_sequence,
        history_size: int = 3,
        predictor_depth=None,
        halt_mode="none",
        halt_eps=None,
        halt_threshold=0.5,
        min_depth=1,
        return_depth_stats=False,
    ):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        depth_used_rollout = []
        residuals_rollout = []
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred = self.predict(
                emb_trunc,
                act_trunc,
                return_all=return_depth_stats,
                predictor_depth=predictor_depth,
                halt_mode=halt_mode,
                halt_eps=halt_eps,
                halt_threshold=halt_threshold,
                min_depth=min_depth,
            )
            if return_depth_stats:
                pred_emb = pred["pred"][:, -1:]  # (BS, 1, D)
                depth_used_rollout.append(pred["depth_used"][:, -1:])
                residuals_rollout.append(pred["residuals"][:, :, -1])
            else:
                pred_emb = pred[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred = self.predict(
            emb_trunc,
            act_trunc,
            return_all=return_depth_stats,
            predictor_depth=predictor_depth,
            halt_mode=halt_mode,
            halt_eps=halt_eps,
            halt_threshold=halt_threshold,
            min_depth=min_depth,
        )
        if return_depth_stats:
            pred_emb = pred["pred"][:, -1:]  # (BS, 1, D)
            depth_used_rollout.append(pred["depth_used"][:, -1:])
            residuals_rollout.append(pred["residuals"][:, :, -1])
        else:
            pred_emb = pred[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        if return_depth_stats:
            depth_used = torch.cat(depth_used_rollout, dim=1)
            residuals = torch.stack(residuals_rollout, dim=1)
            residuals = rearrange(residuals, "k t bs -> bs t k")
            info["depth_used"] = rearrange(depth_used, "(b s) t -> b s t", b=B, s=S)
            info["residuals"] = rearrange(residuals, "(b s) t k -> b s t k", b=B, s=S)

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor, **rollout_kwargs):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        default_rollout_kwargs = getattr(self, "rollout_kwargs", {})
        rollout_kwargs = {**default_rollout_kwargs, **rollout_kwargs}
        info_dict = self.rollout(info_dict, action_candidates, **rollout_kwargs)
        if "depth_used" in info_dict:
            self._record_depth_stats(
                info_dict["depth_used"],
                max_depth=rollout_kwargs.get("predictor_depth"),
            )

        cost = self.criterion(info_dict)
        
        return cost
