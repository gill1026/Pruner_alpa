"""ReconPruner and reconstruction decoder (reimplementation after FastDriveVLA, arXiv:2507.23318).

From the paper: learnable query Q, one decoder layer as PrunerLayer with full attention,
S = Scorer(V* ⊙ Q*), training split M = 1[S > 0] with a straight-through estimator,
V_fore = M̃ ⊙ V, V_back = (1 − M̃) ⊙ V, 6-layer reconstruction decoder.
Our choices (not in the paper): PrunerLayer init from Alpamayo LLM layer 0, the Pruner sees the
real prompt prefix (camera/frame text + all 24 images) with the LLM's own MRoPE position ids,
and the decoder is a per-image ViT-style decoder with a linear patch head.
"""
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding


class ReconPruner(nn.Module):
    def __init__(self, parts, init="llm_layer0"):
        super().__init__()
        cfg = Qwen3VLTextConfig(**parts["text_config"])
        cfg._attn_implementation = "sdpa"  # unset when the layer is built outside a full model
        d = cfg.hidden_size
        self.layer = Qwen3VLTextDecoderLayer(cfg, 0)
        self.rotary = Qwen3VLTextRotaryEmbedding(cfg)
        if init == "llm_layer0":
            self.layer.load_state_dict({k: v.float() for k, v in parts["layer0_state"].items()}, strict=True)
        elif init != "random":
            raise ValueError(init)
        self.query = nn.Parameter(torch.randn(1, 1, d) * parts["text_embeds"].float().std())
        self.scorer = nn.Linear(d, 1)
        # fixed prompt prefix: text embeddings, position ids, where the image tokens go
        self.register_buffer("text_embeds", parts["text_embeds"].float(), persistent=False)
        self.register_buffer("img_idx", parts["img_idx"], persistent=False)
        pos = parts["pos_ids"]
        q_pos = pos.max(dim=1, keepdim=True).values + 1  # query sits right after the prefix
        self.register_buffer("pos_ids", torch.cat([pos, q_pos], dim=1), persistent=False)

    def forward(self, v):
        """v: (B, N, D) visual tokens of one window, N = n_images * tok_per_img. Returns scores (B, N)."""
        b = v.shape[0]
        x = self.text_embeds.unsqueeze(0).expand(b, -1, -1).to(v.dtype).clone()
        x[:, self.img_idx] = v
        x = torch.cat([x, self.query.expand(b, -1, -1).to(v.dtype)], dim=1)
        pos = self.pos_ids.unsqueeze(1).expand(-1, b, -1)
        cos_sin = self.rotary(x, pos)
        # an explicit all-zero mask: with attention_mask=None the HF layer falls back to causal attention
        mask = torch.zeros(1, 1, 1, x.shape[1], device=x.device, dtype=x.dtype)
        h = self.layer(x, position_embeddings=cos_sin, attention_mask=mask)
        if isinstance(h, tuple):
            h = h[0]
        return self.scorer(h[:, self.img_idx] * h[:, -1:]).squeeze(-1)


def ste_mask(s):
    """Binary mask in the forward pass, gradient of S in the backward pass: M̃ = S + sg(M − S)."""
    m = (s > 0).to(s.dtype)
    return s + (m - s).detach(), m


class ReconDecoder(nn.Module):
    """Tokens of one image (dropped ones zeroed) -> RGB image. Linear patch head, MAE style."""

    def __init__(self, in_dim, grid_hw, width=768, depth=6, heads=12, patch=32):
        super().__init__()
        self.grid_hw, self.patch = grid_hw, patch
        self.in_proj = nn.Linear(in_dim, width)
        # positions must be learnable here: zeroed tokens are otherwise indistinguishable
        self.pos = nn.Parameter(torch.zeros(1, grid_hw[0] * grid_hw[1], width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, patch * patch * 3)

    def forward(self, tokens):
        """tokens: (N, T, in_dim) -> (N, 3, H, W) in pixel range [0, 1] (not clamped)."""
        x = self.head(self.norm(self.blocks(self.in_proj(tokens) + self.pos)))
        n, (gh, gw), p = x.shape[0], self.grid_hw, self.patch
        x = x.reshape(n, gh, gw, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(n, 3, gh * p, gw * p)


class QwenReconDecoder(nn.Module):
    """Paper-style decoder: `depth` Qwen3-VL decoder layers at the LLM width + a feedforward head.

    The paper uses six Qwen2.5-VL-3B decoder layers; position handling and the head are not
    specified, so we use 2D MRoPE (row, col) plus a learnable absolute embedding and a linear patch head.
    """

    def __init__(self, parts, grid_hw, depth=6, patch=32, init="llm_layers"):
        super().__init__()
        cfg = Qwen3VLTextConfig(**parts["text_config"])
        cfg._attn_implementation = "sdpa"
        d = cfg.hidden_size
        self.grid_hw, self.patch = grid_hw, patch
        self.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(cfg, i) for i in range(depth)])
        self.pretrained = init == "llm_layers"
        if self.pretrained:  # ours: start from Alpamayo LLM layers 1..depth (layer 0 is the PrunerLayer init)
            for i, layer in enumerate(self.layers):
                layer.load_state_dict({k: v.float() for k, v in parts["llm_layers_state"][i + 1].items()}, strict=True)
        elif init != "random":
            raise ValueError(init)
        self.rotary = Qwen3VLTextRotaryEmbedding(cfg)
        self.pos = nn.Parameter(torch.zeros(1, grid_hw[0] * grid_hw[1], d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, patch * patch * 3)
        rows = torch.arange(grid_hw[0]).repeat_interleave(grid_hw[1])
        cols = torch.arange(grid_hw[1]).repeat(grid_hw[0])
        self.register_buffer("pos_ids", torch.stack([torch.zeros_like(rows), rows, cols]), persistent=False)

    def forward(self, tokens):
        n, t, _ = tokens.shape
        x = tokens + self.pos.to(tokens.dtype)
        cos_sin = self.rotary(x, self.pos_ids.unsqueeze(1).expand(-1, n, -1))
        mask = torch.zeros(1, 1, 1, t, device=x.device, dtype=x.dtype)  # full attention
        for layer in self.layers:
            x = layer(x, position_embeddings=cos_sin, attention_mask=mask)
            x = x[0] if isinstance(x, tuple) else x
        x = self.head(self.norm(x))
        (gh, gw), p = self.grid_hw, self.patch
        x = x.reshape(n, gh, gw, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(n, 3, gh * p, gw * p)


def build_decoder(cfg_model, parts):
    kind = cfg_model.get("decoder_type", "light")
    if kind == "qwen":
        return QwenReconDecoder(parts, parts["grid_hw"], cfg_model["decoder_depth"],
                                init=cfg_model.get("decoder_init", "llm_layers"))
    if kind == "light":
        return ReconDecoder(parts["text_config"]["hidden_size"], parts["grid_hw"], cfg_model["decoder_width"],
                            cfg_model["decoder_depth"], cfg_model["decoder_heads"])
    raise ValueError(kind)


def decoder_param_groups(decoder):
    """(pretrained params, newly initialized params): they get different learning rates."""
    pre = list(decoder.layers.parameters()) if getattr(decoder, "pretrained", False) else []
    ids = {id(p) for p in pre}
    return pre, [p for p in decoder.parameters() if id(p) not in ids]


@torch.no_grad()
def check_full_attention(pruner, n_tokens, dim, device):
    """Changing the last visual token must change the score of the first one (non-causal)."""
    v = torch.randn(1, n_tokens, dim, device=device)
    s0 = pruner(v)
    v[:, -1] += 10.0
    s1 = pruner(v)
    assert (s0[0, 0] - s1[0, 0]).abs() > 0, "PrunerLayer attention is causal"
