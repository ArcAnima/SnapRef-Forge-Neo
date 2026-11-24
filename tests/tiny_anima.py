"""Test helpers (not a test module): a tiny CPU model with the interfaces snapref_forge uses from Forge's Anima
(blocks with self_attn.compute_qkv / compute_attention / output_proj, cross_attn, mlp.layer1 / layer2,
prepare_embedded_sequence, t_embedder, pos_embedder), a minimal stand-in for Forge's model patcher, and synthetic
SnapRef adapter files: v2a-like (stream LoRAs only), v2s-like (+ a reference-stream "style" branch on every
reference module, as snapref_v2s_pre1) and v2t-like (+ a target-stream "style_tgt" branch on a module with a tgt
LoRA and on one without). The stream LoRAs and the cs / char mode rows are the same tensors in all three kinds."""
import json
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

DIM, HEADS, BLOCKS, CHANNELS, RANK, ALPHA = 8, 2, 2, 4, 2, 1.5
SELF = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj")
REF_ONLY = ("mlp.layer1", "mlp.layer2", "cross_attn.q_proj", "cross_attn.output_proj")
SHAPES = {name: (DIM, DIM) for name in SELF + ("cross_attn.q_proj", "cross_attn.output_proj")}   # (in, out)
SHAPES.update({"mlp.layer1": (DIM, 2 * DIM), "mlp.layer2": (2 * DIM, DIM)})
STYLE_TARGETS = SELF + REF_ONLY              # v2s: the style branch sits on every reference-stream module
TGT_TARGETS = ("self_attn.q_proj", "mlp.layer1")   # mlp.layer1 has no tgt stream LoRA


class TinyAttention(nn.Module):
    def __init__(self, cross=False):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.output_proj = (nn.Linear(DIM, DIM, bias=False) for _ in range(4))
        self.output_dropout = nn.Identity()
        self.cross = cross

    def compute_qkv(self, x, context=None, rope_emb=None, transformer_options=None):
        source = context if self.cross else x
        split = lambda t: t.view(t.shape[0], t.shape[1], HEADS, DIM // HEADS)   # noqa: E731
        return split(self.q_proj(x)), split(self.k_proj(source)), split(self.v_proj(source))

    def compute_attention(self, q, k, v, transformer_options=None):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        return self.output_dropout(self.output_proj(out.reshape(q.shape[0], q.shape[1], DIM)))

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        return self.compute_attention(*self.compute_qkv(x, context, rope_emb=rope_emb), transformer_options=transformer_options)


class TinyBlock(nn.Module):
    x_dim = DIM

    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = TinyAttention(), TinyAttention(cross=True)
        self.mlp = nn.Module()
        self.mlp.layer1, self.mlp.layer2 = nn.Linear(DIM, 2 * DIM, bias=False), nn.Linear(2 * DIM, DIM, bias=False)

    def forward(self, x, emb, context, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None,
                transformer_options={}):
        B, T, H, W, _ = x.shape
        h = x.reshape(B, T * H * W, DIM) + emb.reshape(emb.shape[0], 1, DIM)
        h = h + self.self_attn(h, None, rope_emb=rope_emb_L_1_1_D, transformer_options=transformer_options)
        ctx = context if context.ndim == 3 else context[None]
        h = h + self.cross_attn(h, ctx.expand(B, -1, -1), transformer_options=transformer_options)
        h = h + self.mlp.layer2(F.gelu(self.mlp.layer1(h)))
        return h.reshape(B, T, H, W, DIM)


class Anima(nn.Module):
    """Class name as Forge's model; patch size 1 so latents map to tokens one to one."""
    patch_spatial = patch_temporal = 1
    model_channels = DIM

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.x_embedder = nn.Linear(CHANNELS, DIM, bias=False)
        self.time = nn.Linear(1, DIM)
        self.final = nn.Linear(DIM, CHANNELS, bias=False)
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(BLOCKS))
        self.pos_embedder = SimpleNamespace(t_ntk_factor=1.0, h_ntk_factor=1.0, w_ntk_factor=1.0,
                                            dim_temporal_range=torch.arange(2) / 2, dim_spatial_range=torch.arange(2) / 2)
        self.t_embedder = [lambda t: self.time(t.float().reshape(t.shape[0], 1, 1)), lambda e: (e, None)]
        self.t_embedding_norm = nn.Identity()

    def prepare_embedded_sequence(self, x, fps=None, padding_mask=None):
        hidden = self.x_embedder(x.permute(0, 2, 3, 4, 1))
        rope = torch.zeros(hidden.shape[1] * hidden.shape[2] * hidden.shape[3], 1)   # unused by the tiny blocks
        return hidden, rope, None

    def final_layer(self, x, emb, adaln_lora_B_T_3D=None):
        return self.final(x)

    def unpatchify(self, x):
        return x.permute(0, 4, 1, 2, 3)

    def forward(self, x, timesteps, context, padding_mask=None, transformer_options=None, **kwargs):
        """As train/asym.target_forward without a reference."""
        hidden, rope, _ = self.prepare_embedded_sequence(x, padding_mask=padding_mask)
        emb, adaln = self.t_embedder[1](self.t_embedder[0](timesteps.reshape(x.shape[0], 1)).to(hidden.dtype))
        emb = self.t_embedding_norm(emb)
        for block in self.blocks:
            hidden = block(hidden, emb, context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0), adaln_lora_B_T_3D=adaln,
                           extra_per_block_pos_emb=None, transformer_options=transformer_options or {})
        return self.unpatchify(self.final_layer(hidden.to(context.dtype), emb, adaln_lora_B_T_3D=adaln))[:, :, :1]


class Patcher:
    """Just enough of Forge's ModelPatcher: object patches resolved by dotted name, transformer options."""

    def __init__(self, dm, patches=None, options=None):
        self.model = SimpleNamespace(diffusion_model=dm, computation_dtype=torch.float32)
        self.patches, self.options, self.preserved = dict(patches or {}), dict(options or {}), 0

    def clone(self):
        return Patcher(self.model.diffusion_model, self.patches, self.options)

    def _walk(self, name):
        obj = self.model
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    def get_model_object(self, name):
        return self.patches[name] if name in self.patches else self._walk(name)

    def add_object_patch(self, name, obj):
        self.patches[name] = obj

    def set_transformer_option(self, key, value):
        self.options[key] = value

    def add_extra_preserved_memory_during_sampling(self, n):
        self.preserved += n

    def apply(self):
        """Install the patches on the model (Forge does this when it loads the model for sampling)."""
        for name, obj in self.patches.items():
            parent, attr = name.rsplit(".", 1)
            setattr(self._walk(parent), attr, obj)
        return self.model.diffusion_model


@contextmanager
def forge_backend():
    """backend.utils.pad_to_patch_size for Session.capture (patch size 1: nothing to pad)."""
    backend, utils = ModuleType("backend"), ModuleType("backend.utils")
    utils.pad_to_patch_size = lambda x, patch_size: x
    backend.utils = utils
    with patch.dict(sys.modules, {"backend": backend, "backend.utils": utils}):
        yield


def _rand(gen, *shape, scale=0.5):
    return torch.randn(*shape, generator=gen) * scale


def quantize(w):
    """deploy/quantize_adapter_fp8.quantize: float8_e4m3fn with one float32 scale per row."""
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    return (w / scale).to(torch.float8_e4m3fn), scale


def write_adapter(path, kind="v2s", fp8=False, meta_key="anima_ref_kv"):
    """kind: v2a (stream LoRAs, modes cs / char), v2s (+ ref branch "style", mode style) or v2t (+ tgt branch
    "style_tgt"). Returns the path."""
    tensors = {}

    def matrix(key, value):
        if fp8:
            tensors[key], tensors[key + "_scale"] = quantize(value)
        else:
            tensors[key] = value.to(torch.bfloat16)

    def lora(root, shape, gen, down="lora_down", up="lora_up"):
        matrix(root + down, _rand(gen, RANK, shape[0]))
        matrix(root + up, _rand(gen, shape[1], RANK))
        tensors[root + "alpha"] = torch.tensor(ALPHA)

    gen = torch.Generator().manual_seed(0)   # v2a part: identical in every kind
    for b in range(BLOCKS):
        for target in SELF + REF_ONLY:
            lora(f"anima_ref_kv.ref.blocks.{b}.{target}.", SHAPES[target], gen)
        for target in SELF:
            lora(f"anima_ref_kv.tgt.blocks.{b}.{target}.", SHAPES[target], gen)
    rows = _rand(gen, 2, DIM, scale=0.2)
    targets = {"ref": {"rank": RANK, "alpha": ALPHA, "targets": list(SELF + REF_ONLY)},
               "tgt": {"rank": RANK, "alpha": ALPHA, "targets": list(SELF)}}
    meta = {"format": "anima_ref_kv/1", "ref_t": 10, "modes": ["cs", "char"], "lora": targets}
    if kind == "v2a":
        tensors["anima_ref_kv.mode_embed"] = rows
    else:
        gen = torch.Generator().manual_seed(1)
        tensors["anima_ref_kv.mode_embed"] = torch.cat([rows, _rand(gen, 1, DIM, scale=0.2)])
        meta["modes"] = ["cs", "char", "style"]
        spec = lambda stream, targets: {"stream": stream, "modes": ["style"], "rank": RANK, "alpha": ALPHA,   # noqa: E731
                                        "targets": list(targets), "combine": "add"}
        meta["branches"] = {"style": spec("ref", STYLE_TARGETS)}
        for b in range(BLOCKS):
            for target in STYLE_TARGETS:
                lora(f"anima_ref_kv.branch.style.blocks.{b}.{target}.", SHAPES[target], gen, "down", "up")
        if kind == "v2t":
            gen = torch.Generator().manual_seed(2)
            meta["branches"]["style_tgt"] = spec("tgt", TGT_TARGETS)
            for b in range(BLOCKS):
                for target in TGT_TARGETS:
                    lora(f"anima_ref_kv.branch.style_tgt.blocks.{b}.{target}.", SHAPES[target], gen, "down", "up")
    save_file(tensors, str(path), metadata={meta_key: json.dumps(meta)})
    return path


def inputs(batch=2, seed=3):
    """Target latent x [B, C, 1, 3, 4], timesteps, target context, a reference latent [1, C, 1, 2, 2] and its
    caption context."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, CHANNELS, 1, 3, 4, generator=gen), torch.rand(batch, generator=gen),
            torch.randn(batch, 3, DIM, generator=gen), torch.randn(1, CHANNELS, 1, 2, 2, generator=gen),
            torch.randn(1, 2, DIM, generator=gen))
