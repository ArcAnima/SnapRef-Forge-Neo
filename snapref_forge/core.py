"""SnapRef adapter format and inference math (Apache-2.0, ArcAnima/SnapRef)."""
from __future__ import annotations

import json
import math
import base64
import io

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file


def block_map(indices, count):
    if set(indices) == set(range(count)):
        return tuple(range(count))
    if count == 40 and set(indices) == set(range(28)):
        return tuple(i for i in range(40) if i not in {2,5,8,11,14,17,21,24,27,30,33,36})
    raise ValueError(f"Unsupported adapter/model block layout: {len(indices)} / {count}")


class Adapter:
    def __init__(self, path):
        with safe_open(str(path), framework='pt') as f:
            md = f.metadata() or {}
        self.meta = json.loads(md.get('snapref') or md.get('anima_ref_kv') or '{}')
        if self.meta.get('format') != 'anima_ref_kv/1':
            raise ValueError('Select SnapRef weights (anima_ref_kv/1), not an ordinary LoRA.')
        tensors = load_file(str(path), device='cpu')
        self.mode_embed = tensors.get('anima_ref_kv.mode_embed')
        self.modes = self.meta.get('modes', ['cs'])
        self.ref_t = int(self.meta.get('ref_t', 10))
        self.lora = {}
        for key in tensors:
            if not key.startswith('anima_ref_kv.') or not key.endswith('.lora_down'):
                continue
            root = key.removesuffix('lora_down')
            _, stream, name = root[:-1].split('.', 2)
            down, up = (self.dequant(tensors, root+n) for n in ['lora_down','lora_up'])
            if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
                raise ValueError(f'Invalid LoRA shape: {name}')
            scale = float(tensors[root+'alpha']) / down.shape[0]
            if not math.isfinite(scale): raise ValueError('Non-finite adapter scale')
            self.lora.setdefault(name,{})[stream] = down, up, scale
        if {s for streams in self.lora.values() for s in streams} != {'ref','tgt'}:
            raise ValueError('SnapRef requires ref and tgt LoRA streams.')
        self.indices = {int(n.split('.')[1]) for n in self.lora if n.startswith('blocks.')}

    @staticmethod
    def dequant(tensors, key):
        weight = tensors[key]
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and key+'_scale' not in tensors:
            raise ValueError(f'FP8 tensor is missing its row scales: {key}')
        if key+'_scale' in tensors:
            weight = weight.float() * tensors[key+'_scale']
        return weight.to(torch.bfloat16)


def select_references(images, weights, layout):
    if len(images) > 4 or len(images) != len(weights):
        raise ValueError('Use up to four reference images and one strength per image.')
    if layout not in ('Separate references', 'Side by side'):
        raise ValueError('Unknown reference layout')
    refs, strengths = [], []
    for image, weight in zip(images, weights):
        if not math.isfinite(float(weight)) or not 0 <= float(weight) <= 2:
            raise ValueError('Reference strengths must be finite numbers between 0 and 2.')
        if image is None or weight == 0: continue
        if isinstance(image,str):
            try:
                encoded = image.split(',',1)[1] if image.startswith('data:') else image
                with Image.open(io.BytesIO(base64.b64decode(encoded,validate=True))) as decoded:
                    image = decoded.copy()
            except Exception as exc:
                raise ValueError('Reference must be an image or a base64-encoded image.') from exc
        if not isinstance(image,Image.Image): raise ValueError('Invalid reference image')
        image = ImageOps.exif_transpose(image).convert('RGBA')
        bg = Image.new('RGBA', image.size, 'white')
        refs.append(Image.alpha_composite(bg,image).convert('RGB'))
        strengths.append(float(weight))
    if layout == 'Side by side' and len(refs) > 1:
        height = refs[0].height
        refs = [r.resize((max(1,round(r.width*height/r.height)),height),Image.Resampling.LANCZOS) for r in refs]
        canvas = Image.new('RGB',(sum(r.width for r in refs),height),'white')
        left = 0
        for ref in refs:
            canvas.paste(ref,(left,0)); left += ref.width
        refs, strengths = [canvas], [1.]
    return refs, strengths


def attention(q, k, v, refs, sizes, biases, rows, row_biases):
    return _rows_attention(q, k, v, refs, sizes, biases, rows, row_biases)


def _lse_attention(q, k, v):
    """Fused attention with its log-sum-exp. q, k, v [B, H, S, D] -> out [B, H, S, D] (float32), lse [B, H, S, 1]."""
    if not (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)):
        raise RuntimeError("no fused kernel")
    try:
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[:2]
    except (RuntimeError, AttributeError):
        out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)[:2]
    return out.float(), lse.float()[..., :q.shape[2], None]


def _rows_attention(q, k_t, v_t, kv_ref, sizes, biases, rows, row_biases=None):
    """One call for a batch where only some rows read the reference (cond reads it, uncond does not): plain
    attention over the target keys for every row, plus the reference groups merged by log-sum-exp into the rows
    that read them. Rows without the reference get exactly the plain attention. row_biases: optional extra log bias
    per selected row, added to every reference group of that row (Fidelity: uncond rows at log(1 - f))."""
    B, S, H, D = q.shape
    qh, kt, vt = q.transpose(1, 2), k_t.transpose(1, 2), v_t.transpose(1, 2)
    try:
        out_t, lse_t = _lse_attention(qh, kt, vt)
    except RuntimeError:
        return _masked_attention(q, k_t, v_t, kv_ref, sizes, biases, rows, row_biases)
    sel = torch.as_tensor(rows, device=q.device)
    k_r, v_r = kv_ref
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    qs = qh.index_select(0, sel)
    parts = [(0.0, out_t.index_select(0, sel), lse_t.index_select(0, sel))]
    for i, b in enumerate(biases):
        kr = k_r[:, starts[i]:starts[i] + sizes[i]].expand(len(rows), -1, -1, -1).to(q.dtype).transpose(1, 2)
        vr = v_r[:, starts[i]:starts[i] + sizes[i]].expand(len(rows), -1, -1, -1).to(q.dtype).transpose(1, 2)
        parts.append((b, *_lse_attention(qs, kr, vr)))
    if row_biases is not None:
        extra = torch.as_tensor(row_biases, device=q.device, dtype=torch.float32).view(-1, 1, 1, 1)
        parts = [parts[0]] + [(b + extra, o, lse) for b, o, lse in parts[1:]]   # target group stays at 0
    m = torch.stack([(lse + b).expand_as(parts[0][2]) for b, _, lse in parts]).amax(dim=0)
    num, den = 0.0, 0.0
    for b, o, lse in parts:
        w = torch.exp(lse + b - m)
        num, den = num + w * o, den + w
    out = out_t.clone()
    out.index_copy_(0, sel, num / den)
    return out.to(q.dtype).transpose(1, 2).reshape(B, S, H * D)


def _masked_attention(q, k_t, v_t, kv_ref, sizes, biases, rows=None, row_biases=None):
    """Reference implementation (and fallback) of _weighted_attention: SDPA with an additive key mask."""
    k_r, v_r = kv_ref
    B, S, H, D = q.shape
    k = torch.cat([k_r.expand(B, *k_r.shape[1:]).to(q.dtype), k_t], dim=1).transpose(1, 2)
    v = torch.cat([v_r.expand(B, *v_r.shape[1:]).to(q.dtype), v_t], dim=1).transpose(1, 2)
    n = k.shape[2]
    pad = (-n) % 16
    if pad:
        k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
    mask = torch.zeros(B if rows is not None else 1, 1, 1, n + pad, device=q.device, dtype=q.dtype)
    start = 0
    for size, bias in zip(sizes, biases):
        mask[..., start:start + size] = bias
        start += size
    if rows is not None:
        selected = torch.zeros(B, device=q.device, dtype=torch.bool)
        selected[rows] = True
        mask[~selected, ..., :start] = float("-inf")
        if row_biases is not None:
            mask[rows, ..., :start] += torch.as_tensor(row_biases, device=q.device, dtype=q.dtype).view(-1, 1, 1, 1)
    if pad:
        mask[..., n:] = float("-inf")
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=mask.expand(B, H, S, n + pad))
    return out.transpose(1, 2).reshape(B, S, H * D)


def _rope(pe, t, h, w):
    """VideoRopePosition3DEmb for explicit per-token positions (train/anima_env.rope_from_positions)."""
    def rot(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)
    dev = t.device
    emb = torch.cat([rot(t, 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(dev))),
                     rot(h, 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(dev))),
                     rot(w, 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(dev)))], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _correct(guided, pos, correction, rescale_phi=0.5):
    """APG (drop the update's component parallel to the cond prediction) or CFG rescale, in x0 space
    (as ElectronGirl/ComfyUI-Anima-Ref-Adapter, Apache-2.0)."""
    dims = tuple(range(1, pos.ndim))
    if correction == "apg":
        update = (guided - pos).double()
        unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
        return pos + (update - (update * unit).sum(dim=dims, keepdim=True) * unit).to(pos.dtype)
    if correction == "rescale":
        ratio = pos.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True)
        return rescale_phi * guided * ratio + (1 - rescale_phi) * guided
    return guided
