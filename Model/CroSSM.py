import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------
# Optional: try to use real Mamba from mamba-ssm
#   pip install mamba-ssm
# If not installed, fallback to a simple Mamba-like linear-time block
# -----------------------------
_MAMBA_CLS = None
try:
    # some versions export Mamba directly
    from mamba_ssm import Mamba as _MAMBA_CLS  # type: ignore
except Exception:
    try:
        # other versions place it here
        from mamba_ssm.modules.mamba_simple import Mamba as _MAMBA_CLS  # type: ignore
    except Exception:
        _MAMBA_CLS = None


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class MHAOnly(nn.Module):
    """
    Wrap nn.MultiheadAttention so that forward returns ONLY the attention output tensor.
    """
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, attn_drop=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=attn_drop,
            batch_first=True
        )

    def forward(self, q, k, v):
        out, _ = self.mha(q, k, v, need_weights=True)
        return out


# ==========================================================
# (NEW) Simple fallback Mamba-like module (no external deps)
# - linear-time scan with per-channel decay (SSM-ish)
# - depthwise causal conv for local mixing
# - gating + in/out projections (Mamba-style)
# ==========================================================
class SimpleMambaLike(nn.Module):
    """
    A lightweight, dependency-free Mamba-like block:
      x -> in_proj -> u, gate
      u -> depthwise causal conv -> SiLU
      u -> (b,c) and scan: state = a*state + b_t ; y_t = c_t * state
      y -> gate -> out_proj
    This is NOT the official Mamba kernel, but matches the "SSM + gating + conv" spirit,
    and keeps O(N) time in principle.
    """
    def __init__(self, d_model: int, d_conv: int = 4, expand: int = 2, drop: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(d_model * expand)

        # in_proj: produce content u and gate
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        # depthwise causal conv over sequence length
        self.dwconv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,   # causal-ish (we'll slice)
            bias=True
        )

        # produce b and c per token (input-dependent)
        self.bc_proj = nn.Linear(self.d_inner, 2 * self.d_inner)

        # per-channel decay parameter (stable 0<a<1)
        self.log_decay = nn.Parameter(torch.zeros(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)  # (B,N,inner), (B,N,inner)

        # depthwise causal conv: (B,inner,N) -> conv -> (B,inner,N+pad) then slice to N
        u_conv = self.dwconv(u.transpose(1, 2))[:, :, :N].transpose(1, 2)
        u_conv = F.silu(u_conv)

        b, c = self.bc_proj(u_conv).chunk(2, dim=-1)  # (B,N,inner)

        # decay in (0,1): a = exp(-softplus(log_decay))
        a = torch.exp(-F.softplus(self.log_decay)).to(dtype=x.dtype, device=x.device)  # (inner,)

        # scan (simple loop). N is small in patch setting; for very long N use real mamba-ssm.
        state = torch.zeros(B, self.d_inner, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(N):
            state = state * a + b[:, t, :]
            ys.append(state * c[:, t, :])
        y = torch.stack(ys, dim=1)  # (B,N,inner)

        y = y * torch.sigmoid(gate)
        y = self.drop(self.out_proj(y))
        return y


class MambaOnly(nn.Module):
    """
    Use real Mamba if available; otherwise fallback to SimpleMambaLike.
    Forward returns ONLY tensor (B,N,D).
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, drop: float = 0.0):
        super().__init__()
        self.use_real = _MAMBA_CLS is not None
        if self.use_real:
            # Real Mamba (from mamba-ssm)
            # Common signature: Mamba(d_model, d_state=16, d_conv=4, expand=2)
            self.core = _MAMBA_CLS(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.drop = nn.Dropout(drop)
        else:
            # Fallback Mamba-like
            self.core = SimpleMambaLike(d_model, d_conv=d_conv, expand=expand, drop=drop)
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(x)
        return self.drop(y)


# ==========================================================
# (CHANGED) Replace Self-Attn Block with Mamba Block (方案一)
# ==========================================================
class MambaBlock(nn.Module):
    """
    Transformer-like block but uses Mamba instead of Self-Attn:
      LN -> Mamba -> residual
      LN -> FFN  -> residual
    """
    def __init__(self, dim, mlp_ratio=4.0, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = MambaOnly(dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand, drop=proj_drop)
        self.drop1 = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)

    def forward(self, x):
        # x: (B, N, D)
        x = x + self.drop1(self.mamba(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """LN(target) as Q, LN(source) as K,V -> CrossAttn -> FFN, 输出维度=target_dim"""
    def __init__(self, target_dim, source_dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(target_dim)
        self.norm_kv = nn.LayerNorm(source_dim)

        self.attn = MHAOnly(embed_dim=target_dim, num_heads=num_heads, kdim=source_dim, vdim=source_dim, attn_drop=attn_drop)
        self.drop1 = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(target_dim)
        self.mlp = MLP(target_dim, mlp_ratio=mlp_ratio, drop=proj_drop)

    def forward(self, x_tgt, x_src):
        # x_tgt: (B, N, Dt), x_src: (B, N, Ds)
        q = self.norm_q(x_tgt)
        kv = self.norm_kv(x_src)
        out = self.attn(q, kv, kv)
        x_tgt = x_tgt + self.drop1(out)
        x_tgt = x_tgt + self.mlp(self.norm2(x_tgt))
        return x_tgt


def _pick_heads(dim: int) -> int:
    """给定维度选择一个合理 head 数（必须整除 dim）"""
    for h in [8, 4, 2, 1]:
        if dim % h == 0:
            return h
    return 1


# --------- TwoStreamStage：每流内部(原self-attn) -> 改为 MambaBlock；跨模态仍用 Cross-Attn ---------
class TwoStreamStage(nn.Module):
    def __init__(self, dim_a, dim_b, depth=1, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
                 # mamba hyperparams (can tune)
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        heads_a = _pick_heads(dim_a)
        heads_b = _pick_heads(dim_b)

        # (CHANGED) self blocks now use MambaBlock
        self.self_a = nn.ModuleList([
            MambaBlock(dim_a, mlp_ratio=mlp_ratio,
                      mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand,
                      proj_drop=proj_drop)
            for _ in range(depth)
        ])
        self.self_b = nn.ModuleList([
            MambaBlock(dim_b, mlp_ratio=mlp_ratio,
                      mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand,
                      proj_drop=proj_drop)
            for _ in range(depth)
        ])

        # cross-attn unchanged
        self.cross_a_from_b = nn.ModuleList([
            CrossAttnBlock(dim_a, dim_b, heads_a, mlp_ratio, attn_drop, proj_drop) for _ in range(depth)
        ])
        self.cross_b_from_a = nn.ModuleList([
            CrossAttnBlock(dim_b, dim_a, heads_b, mlp_ratio, attn_drop, proj_drop) for _ in range(depth)
        ])

        # learnable fusion strength per block (init small => sigmoid ~ 0)
        self.alpha_ab = nn.ParameterList([nn.Parameter(torch.tensor(-4.0)) for _ in range(depth)])  # A<-B
        self.alpha_ba = nn.ParameterList([nn.Parameter(torch.tensor(-4.0)) for _ in range(depth)])  # B<-A

    def forward(self, a, b):
        for i, (sa, sb, cab, cba) in enumerate(zip(self.self_a, self.self_b, self.cross_a_from_b, self.cross_b_from_a)):
            a = sa(a)
            b = sb(b)

            # A <- B with gated update
            a0 = a
            a1 = cab(a, b)
            wa = torch.sigmoid(self.alpha_ab[i])   # scalar in (0,1)
            a = a0 + wa * (a1 - a0)

            # B <- A with gated update (use updated a)
            b0 = b
            b1 = cba(b, a)
            wb = torch.sigmoid(self.alpha_ba[i])
            b = b0 + wb * (b1 - b0)

        return a, b


# -----------------------------
# Main model
# -----------------------------
class CSSM(nn.Module):
    def __init__(self, input_channels, input_channels2, n_classes, patch_size):
        super().__init__()

        # LiDAR-guided HSI band gating
        hidden = max(8, input_channels2 * 4)
        self.hsi_band_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, input_channels, 1),
            nn.Sigmoid()
        )

        self.patch_size = patch_size
        self.N = patch_size * patch_size

        self.planes_a = [128, 64, 32]
        self.planes_b = [8, 16, 32]

        # token embedding: (B,C,H,W) -> (B,N,D)
        self.embed_a = nn.Conv2d(input_channels, self.planes_a[0], kernel_size=1, bias=True)
        self.embed_b = nn.Conv2d(input_channels2, self.planes_b[0], kernel_size=1, bias=True)

        # pos embeddings per stage
        self.pos_a1 = nn.Parameter(torch.zeros(1, self.N, self.planes_a[0]))
        self.pos_b1 = nn.Parameter(torch.zeros(1, self.N, self.planes_b[0]))

        self.pos_a2 = nn.Parameter(torch.zeros(1, self.N, self.planes_a[1]))
        self.pos_b2 = nn.Parameter(torch.zeros(1, self.N, self.planes_b[1]))

        self.pos_a3 = nn.Parameter(torch.zeros(1, self.N, self.planes_a[2]))
        self.pos_b3 = nn.Parameter(torch.zeros(1, self.N, self.planes_b[2]))

        # stages (self blocks are now Mamba inside TwoStreamStage)
        self.stage1 = TwoStreamStage(self.planes_a[0], self.planes_b[0], depth=1)
        self.proj_a12 = nn.Linear(self.planes_a[0], self.planes_a[1])
        self.proj_b12 = nn.Linear(self.planes_b[0], self.planes_b[1])

        self.stage2 = TwoStreamStage(self.planes_a[1], self.planes_b[1], depth=1)
        self.proj_a23 = nn.Linear(self.planes_a[1], self.planes_a[2])
        self.proj_b23 = nn.Linear(self.planes_b[1], self.planes_b[2])

        self.stage3 = TwoStreamStage(self.planes_a[2], self.planes_b[2], depth=1)

        # fusion head
        self.FusionLayer = nn.Sequential(
            nn.Conv2d(self.planes_a[2] * 2, self.planes_a[2], kernel_size=1),
            nn.BatchNorm2d(self.planes_a[2]),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes_a[2], n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_a1, std=0.02)
        nn.init.trunc_normal_(self.pos_b1, std=0.02)
        nn.init.trunc_normal_(self.pos_a2, std=0.02)
        nn.init.trunc_normal_(self.pos_b2, std=0.02)
        nn.init.trunc_normal_(self.pos_a3, std=0.02)
        nn.init.trunc_normal_(self.pos_b3, std=0.02)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def _to_tokens(self, x, embed_conv, pos):
        B, _, H, W = x.shape
        assert H == self.patch_size and W == self.patch_size, "输入 patch 大小需等于 patch_size"
        x = embed_conv(x)                              # (B,D,H,W)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B,N,D)
        x = x + pos
        return x

    def _tokens_to_map(self, t):
        B, N, D = t.shape
        H = W = self.patch_size
        x = t.transpose(1, 2).contiguous().view(B, D, H, W)
        return x

    def forward(self, x1, x2):
        # --- LiDAR-guided band gate ---
        gate = self.hsi_band_gate(x2)   # (B, C1, 1, 1)
        x1 = x1 * gate

        # ---- stage1 ----
        a = self._to_tokens(x1, self.embed_a, self.pos_a1)  # (B,N,128)
        b = self._to_tokens(x2, self.embed_b, self.pos_b1)  # (B,N,8)
        a, b = self.stage1(a, b)

        # ---- stage2 ----
        a = self.proj_a12(a) + self.pos_a2                  # (B,N,64)
        b = self.proj_b12(b) + self.pos_b2                  # (B,N,16)
        a, b = self.stage2(a, b)

        # ---- stage3 ----
        a = self.proj_a23(a) + self.pos_a3                  # (B,N,32)
        b = self.proj_b23(b) + self.pos_b3                  # (B,N,32)
        a, b = self.stage3(a, b)

        # back to feature maps
        ss_x1 = self._tokens_to_map(a)                      # (B,32,H,W)
        ss_x2 = self._tokens_to_map(b)                      # (B,32,H,W)

        x = self.FusionLayer(torch.cat([ss_x1, ss_x2], dim=1))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # test shape
    B = 2
    patch = 7
    img1 = torch.randn(B, 6, patch, patch)
    img2 = torch.randn(B, 6, patch, patch)

    net = CSSM(input_channels=6, input_channels2=6, n_classes=15, patch_size=patch)
    out = net(img1, img2)
    print(out.shape)  # (2, 15)

    # check which Mamba is used
    # (prints True if mamba-ssm installed and imported successfully)
    for m in net.modules():
        if isinstance(m, MambaOnly):
            print("Use real mamba-ssm:", m.use_real)
            break
