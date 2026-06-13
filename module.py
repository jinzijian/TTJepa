import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class RecurrentRefineCell(nn.Module):
    """Shared refinement cell for recurrent next-embedding prediction."""

    def __init__(
        self,
        *,
        hidden_dim,
        feedback_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.feedback_proj = nn.Linear(feedback_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ConditionalBlock(
                    hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, h, c, feedback):
        h = h + self.feedback_proj(feedback)
        for block in self.blocks:
            h = block(h, c)
        return h


class RecurrentARPredictor(nn.Module):
    """Autoregressive predictor with fixed-depth recurrent refinement.

    The default return value is intentionally compatible with ARPredictor:
    forward(x, c) returns a tensor of shape (B, T, D). Passing return_all=True
    exposes all recurrent depths for deep supervision and analysis.
    """

    def __init__(
        self,
        *,
        num_frames,
        base_depth,
        refine_depth,
        max_depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        residual_scale_init=0.1,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.output_dim = output_dim or input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.base_transformer = Transformer(
            input_dim,
            hidden_dim,
            hidden_dim,
            base_depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.anchor_proj = (
            nn.Linear(input_dim, self.output_dim)
            if input_dim != self.output_dim
            else nn.Identity()
        )
        self.init_head = nn.Linear(hidden_dim, self.output_dim)
        self.refine_cell = RecurrentRefineCell(
            hidden_dim=hidden_dim,
            feedback_dim=self.output_dim,
            depth=refine_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.delta_head = nn.Linear(hidden_dim, self.output_dim)
        self.gamma_head = nn.Linear(hidden_dim, self.output_dim)
        self.continue_head = nn.Linear(hidden_dim, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        nn.init.zeros_(self.delta_head.bias)
        nn.init.constant_(self.gamma_head.bias, -2.0)
        nn.init.zeros_(self.continue_head.bias)
        self.register_load_state_dict_pre_hook(self._load_continue_head_compat)

    def _load_continue_head_compat(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Allow fixed-depth checkpoints saved before learned halting to load."""
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        weight_key = f"{prefix}continue_head.weight"
        bias_key = f"{prefix}continue_head.bias"
        if weight_key not in state_dict:
            state_dict[weight_key] = self.continue_head.weight.detach().clone()
        if bias_key not in state_dict:
            state_dict[bias_key] = self.continue_head.bias.detach().clone()

    def forward(
        self,
        x,
        c,
        max_depth=None,
        return_all=False,
        halt_mode="none",
        halt_eps=None,
        halt_threshold=0.5,
        min_depth=1,
    ):
        """
        x: (B, T, D)
        c: (B, T, action_emb_dim)
        """
        if halt_mode not in {"none", "residual", "learned"}:
            raise NotImplementedError(
                "halt_mode must be 'none', 'residual', or 'learned'"
            )
        if halt_mode == "residual" and halt_eps is None:
            raise ValueError("halt_eps must be set when halt_mode='residual'")
        if not 0.0 <= float(halt_threshold) <= 1.0:
            raise ValueError("halt_threshold must satisfy 0 <= halt_threshold <= 1")

        K = int(max_depth or self.max_depth)
        if K < 1:
            raise ValueError("max_depth must be >= 1")
        if min_depth < 1 or min_depth > K:
            raise ValueError("min_depth must satisfy 1 <= min_depth <= max_depth")

        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)

        anchor = self.anchor_proj(x)
        h = self.base_transformer(x, c)
        c = self.cond_proj(c)
        z_hat = self.init_head(h)

        preds = []
        residuals = []
        continue_logits = []
        prev = z_hat
        selected_pred = None
        depth_used = torch.full(
            x.shape[:2],
            fill_value=K,
            device=x.device,
            dtype=torch.long,
        )
        active = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)

        for depth_idx in range(K):
            feedback = z_hat - anchor
            h = self.refine_cell(h, c, feedback)
            delta = self.delta_head(h)
            gamma = torch.sigmoid(self.gamma_head(h))
            continue_logit = self.continue_head(h).squeeze(-1)
            z_hat = z_hat + self.residual_scale * gamma * delta

            residual = (z_hat - prev).pow(2).mean(dim=-1)
            preds.append(z_hat)
            residuals.append(residual)
            continue_logits.append(continue_logit)
            prev = z_hat

            if halt_mode in {"residual", "learned"}:
                current_depth = depth_idx + 1
                if halt_mode == "residual":
                    can_halt = residual <= float(halt_eps)
                else:
                    continue_prob = torch.sigmoid(continue_logit)
                    can_halt = continue_prob <= float(halt_threshold)
                if current_depth < min_depth:
                    can_halt = torch.zeros_like(can_halt, dtype=torch.bool)
                newly_halted = active & can_halt

                if selected_pred is None:
                    selected_pred = torch.zeros_like(z_hat)
                selected_pred = torch.where(
                    newly_halted.unsqueeze(-1),
                    z_hat,
                    selected_pred,
                )
                depth_used = torch.where(
                    newly_halted,
                    torch.full_like(depth_used, current_depth),
                    depth_used,
                )
                active = active & ~newly_halted
                if not return_all and not active.any():
                    break

        if halt_mode in {"residual", "learned"}:
            if selected_pred is None:
                selected_pred = z_hat
            else:
                selected_pred = torch.where(active.unsqueeze(-1), z_hat, selected_pred)
            z_hat = selected_pred

        if return_all:
            return {
                "pred": z_hat,
                "preds": torch.stack(preds, dim=0),
                "residuals": torch.stack(residuals, dim=0),
                "continue_logits": torch.stack(continue_logits, dim=0),
                "depth_used": depth_used,
            }

        return z_hat
