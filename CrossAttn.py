import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch import nn
import torch.nn.functional as F


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
    This avoids torchsummary crashing on (attn_out, None) when need_weights=False.
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


class SelfAttnBlock(nn.Module):
    """标准 Transformer Encoder Block：LN -> SelfAttn -> FFN"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # self.attn = nn.MultiheadAttention(
        #     embed_dim=dim, num_heads=num_heads,
        #     dropout=attn_drop, batch_first=True
        # )
        
        self.attn = MHAOnly(embed_dim=dim, num_heads=num_heads, attn_drop=attn_drop)


        self.drop1 = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)

    def forward(self, x):
        # x: (B, N, D)
        x_norm = self.norm1(x)
        attn_out= self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop1(attn_out)

        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """LN(target) as Q, LN(source) as K,V -> CrossAttn -> FFN, 输出维度=target_dim"""
    def __init__(self, target_dim, source_dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(target_dim)
        self.norm_kv = nn.LayerNorm(source_dim)

        # PyTorch MHA 支持 kdim/vdim 不同（非常适合跨模态不同维度）
        # self.attn = nn.MultiheadAttention(
        #     embed_dim=target_dim, num_heads=num_heads,
        #     kdim=source_dim, vdim=source_dim,
        #     dropout=attn_drop, batch_first=True
        # )
        self.attn = MHAOnly(embed_dim=target_dim, num_heads=num_heads,kdim=source_dim,vdim=source_dim, attn_drop=attn_drop)

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


# --------- 两流一阶段：Self-Attn(各自) + 双向 Cross-Attn ---------

class TwoStreamStage(nn.Module):
    """
    一个 stage 内：
      1) A self-attn
      2) B self-attn
      3) A <- cross-attn from B
      4) B <- cross-attn from A
    """
    def __init__(self, dim_a, dim_b, depth=1, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        heads_a = _pick_heads(dim_a)
        heads_b = _pick_heads(dim_b)

        self.self_a = nn.ModuleList([
            SelfAttnBlock(dim_a, heads_a, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.self_b = nn.ModuleList([
            SelfAttnBlock(dim_b, heads_b, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])

        # 双向 cross-attn：A<-B, B<-A
        self.cross_a_from_b = nn.ModuleList([
            CrossAttnBlock(dim_a, dim_b, heads_a, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.cross_b_from_a = nn.ModuleList([
            CrossAttnBlock(dim_b, dim_a, heads_b, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])

    def forward(self, a, b):
        # a: (B,N,Da), b: (B,N,Db)
        for sa, sb, cab, cba in zip(self.self_a, self.self_b, self.cross_a_from_b, self.cross_b_from_a):
            a = sa(a)
            b = sb(b)
            a = cab(a, b)  # A <- B
            b = cba(b, a)  # B <- A（注意这里用更新后的 a）
        return a, b


# The fucking main model class #

class JViT(nn.Module):
    def __init__(self, input_channels, input_channels2, n_classes, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.N = patch_size * patch_size

        #### 可能是原来的好的点，原来的处理维度
        self.planes_a = [128, 64, 32]
        self.planes_b = [8, 16, 32]

        # 1) token embedding：把 (B,C,H,W) -> (B,N,D)
        #    这里用 1x1 conv 做 per-pixel embedding，然后展平为 token
        self.embed_a = nn.Conv2d(input_channels, self.planes_a[0], kernel_size=1, bias=True)
        self.embed_b = nn.Conv2d(input_channels2, self.planes_b[0], kernel_size=1, bias=True)

        # 位置编码：每个 stage 各自一份（维度不同）
        self.pos_a1 = nn.Parameter(torch.zeros(1, self.N, self.planes_a[0]))
        self.pos_b1 = nn.Parameter(torch.zeros(1, self.N, self.planes_b[0]))

        self.pos_a2 = nn.Parameter(torch.zeros(1, self.N, self.planes_a[1]))
        self.pos_b2 = nn.Parameter(torch.zeros(1, self.N, self.planes_b[1]))
        
        self.pos_a3 = nn.Parameter(torch.zeros(1, self.N, self.planes_a[2]))
        self.pos_b3 = nn.Parameter(torch.zeros(1, self.N, self.planes_b[2]))

        # 2) 三个 stage：每个 stage 内都做一次“各自 self + 双向 cross”
        self.stage1 = TwoStreamStage(self.planes_a[0], self.planes_b[0], depth=1)
        self.proj_a12 = nn.Linear(self.planes_a[0], self.planes_a[1])
        self.proj_b12 = nn.Linear(self.planes_b[0], self.planes_b[1])

        self.stage2 = TwoStreamStage(self.planes_a[1], self.planes_b[1], depth=1)
        self.proj_a23 = nn.Linear(self.planes_a[1], self.planes_a[2])
        self.proj_b23 = nn.Linear(self.planes_b[1], self.planes_b[2])

        self.stage3 = TwoStreamStage(self.planes_a[2], self.planes_b[2], depth=1)

        # 3) 融合头：Merge + 分类
        self.FusionLayer = nn.Sequential(
            nn.Conv2d(self.planes_a[2] * 2, self.planes_a[2], kernel_size=1),
            nn.BatchNorm2d(self.planes_a[2]),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes_a[2], n_classes)

        self._init_weights()

    def _init_weights(self):
        # 轻量初始化
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


    #### 输入到 token，输出回 feature map 的辅助函数 ####   
    def _to_tokens(self, x, embed_conv, pos):
        # x: (B,C,H,W) -> conv -> (B,D,H,W) -> (B,N,D) + pos
        B, _, H, W = x.shape
        assert H == self.patch_size and W == self.patch_size, "输入 patch 大小需等于 patch_size"
        x = embed_conv(x)                              # (B,D,H,W)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B,N,D)
        x = x + pos
        return x

    def _tokens_to_map(self, t):
        # t: (B,N,D) -> (B,D,H,W)
        B, N, D = t.shape
        H = W = self.patch_size
        x = t.transpose(1, 2).contiguous().view(B, D, H, W)
        return x

    def forward(self, x1, x2):
        """
        the same ,but can be changed to 2 stages or 4 stages.
        原 forward 的“3段特征提取 + 融合分类”，但每段改成 Transformer stage：
          stage1: token embed -> self/self + cross/cross
          stage2: 线性投影到新维度 -> self/self + cross/cross
          stage3: 同上
        """
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

        # 转回 feature map, 准备进入跟原来一样的fusion layer
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

    net = JViT(input_channels=6, input_channels2=6, n_classes=15, patch_size=patch)
    out = net(img1, img2)
    print(out.shape)  # (2, 15)
