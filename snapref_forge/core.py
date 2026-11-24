"""SnapRef adapter format and inference math (Apache-2.0, ArcAnima/SnapRef)."""
from __future__ import annotations

import json
import math
import base64
import io
import os
import struct
from pathlib import Path

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


BRANCH_PREFIX = 'anima_ref_kv.branch.'
BRANCH_MODES = ('style',)   # modes that exist only through a branch; cs / char work with every adapter


class Adapter:
    """SnapRef weights: the ref / tgt stream LoRAs and, from v2s on, mode-gated branch LoRAs
    (anima_ref_kv.branch.<branch>.<module>.down / up / alpha, metadata branches.<branch> =
    {stream, modes, rank, alpha, targets, combine: "add"}; train/style_branch.py). A branch runs only in the modes it
    lists, right after its stream's LoRA on the same module (or on its own where that stream has no LoRA)."""

    def __init__(self, path):
        with safe_open(str(path), framework='pt') as f:
            md = f.metadata() or {}
        self.meta = json.loads(md.get('snapref') or md.get('anima_ref_kv') or '{}')
        if self.meta.get('format') != 'anima_ref_kv/1':
            raise ValueError('Select SnapRef weights (anima_ref_kv/1), not an ordinary LoRA.')
        self.name = Path(path).name
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
        self.branches = self._read_branches(tensors)
        modules = set(self.lora).union(*(b['modules'] for b in self.branches.values()))
        self.indices = {int(n.split('.')[1]) for n in modules if n.startswith('blocks.')}

    def _read_branches(self, tensors):
        """anima_ref_kv.branch.<branch>.<module>.down / up / alpha -> {branch: {'stream', 'modes', 'modules':
        {module: (down, up, alpha / rank)}}} in metadata order (as the ComfyUI node's _Adapter._read_branches).
        FP8 files carry a per-row float32 <key>_scale, as the stream LoRAs."""
        specs = self.meta.get('branches') or {}
        if not isinstance(specs, dict): raise ValueError('Invalid SnapRef branch metadata')
        found = {}
        for key in tensors:
            if not (key.startswith(BRANCH_PREFIX) and key.endswith('.down')):
                continue
            root = key.removesuffix('down')
            branch, name = root[len(BRANCH_PREFIX):-1].split('.', 1)
            if branch not in specs:
                raise ValueError(f'Branch {branch} has weights but no metadata (branches.{branch})')
            down, up = (self.dequant(tensors, root+n) for n in ['down','up'])
            if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
                raise ValueError(f'Invalid branch LoRA shape: {branch}.{name}')
            scale = float(tensors[root+'alpha']) / down.shape[0]
            if not math.isfinite(scale): raise ValueError('Non-finite adapter scale')
            found.setdefault(branch,{})[name] = down, up, scale
        out = {}
        for branch, spec in specs.items():
            stream, combine = spec.get('stream'), spec.get('combine', 'add')
            if stream not in ('ref','tgt') or combine != 'add':
                raise ValueError(f'Branch {branch} (stream {stream}, combine {combine}) is not supported '
                                 '(stream ref or tgt, combine add)')
            if branch in found:
                out[branch] = {'stream':stream, 'modes':list(spec.get('modes') or []), 'modules':found[branch]}
        return out

    def branches_for(self, mode):
        """The branches that run in this mode, in metadata order (style_branch.branches_for); () for cs / char."""
        return tuple(name for name, b in self.branches.items() if mode in b['modes'])

    def check_mode(self, mode, label=None):
        """cs / char: never an error here (as before). style: the adapter must list the mode, carry its mode_embed
        row and a branch for it; otherwise a clear error before any encoding (wording of the ComfyUI node)."""
        if mode not in BRANCH_MODES:
            return
        label = label or self.name
        if mode not in self.modes:
            raise ValueError(f"SnapRef: {label} has no '{mode}' mode (its modes: {', '.join(self.modes)}). "
                             "Style mode needs an adapter with a style branch, e.g. snapref_v2s_pre; "
                             "use mode cs or char with this adapter.")
        if self.mode_embed is None or self.mode_embed.shape[0] <= self.modes.index(mode) or not self.branches_for(mode):
            raise ValueError(f"SnapRef: {label} lists the '{mode}' mode but has no mode_embed row or branch weights "
                             "for it (anima_ref_kv.branch.*). Style mode needs an adapter such as snapref_v2s_pre.")

    @staticmethod
    def dequant(tensors, key):
        weight = tensors[key]
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and key+'_scale' not in tensors:
            raise ValueError(f'FP8 tensor is missing its row scales: {key}')
        if key+'_scale' in tensors:
            weight = weight.float() * tensors[key+'_scale']
        return weight.to(torch.bfloat16)


MAX_HEADER = 100 * 1024 * 1024   # bytes; real adapter headers are well under 1 MB


def read_meta(path):
    """The adapter's own metadata (__metadata__ 'snapref' or 'anima_ref_kv', JSON) from the safetensors header
    alone (8-byte little-endian length + JSON), never the tensors (as the ComfyUI node's adapter_info.read_meta)."""
    with open(path, 'rb') as handle:
        head = handle.read(8)
        if len(head) != 8: raise ValueError('not a safetensors file (no header)')
        (size,) = struct.unpack('<Q', head)
        if size > MAX_HEADER: raise ValueError(f'header too large ({size} bytes)')
        raw = handle.read(size)
    if len(raw) != size: raise ValueError('header cut short (file incomplete)')
    header = json.loads(raw)
    meta = (header.get('__metadata__') or {}) if isinstance(header, dict) else {}
    out = json.loads(meta.get('snapref') or meta.get('anima_ref_kv') or '{}')
    if not isinstance(out, dict): raise ValueError('adapter metadata is not an object')
    return out


def has_style(meta):
    """Style needs the mode in the adapter's list and a branch that runs in it (as adapter_info._summary)."""
    modes = [str(m) for m in meta.get('modes', ['cs'])]
    branches = meta.get('branches') or {}
    return 'style' in modes and isinstance(branches, dict) and any(
        isinstance(b, dict) and 'style' in (b.get('modes') or []) for b in branches.values())


MAX_REFERENCES = 8   # as the ComfyUI SnapRef node


def select_references(images, weights, layout):
    if len(images) > MAX_REFERENCES or len(images) != len(weights):
        raise ValueError(f'Use up to {MAX_REFERENCES} reference images and one strength per image.')
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
    """Reference implementation (and fallback) of _weighted_attention: SDPA with an additive key mask.

    rows: only these batch rows read the reference; row_biases: extra log bias per row in rows.
    Off CUDA the mask holds no -inf: MPS returns NaN (a black image) for -inf in an additive mask, although no
    query row is ever fully masked here. So off CUDA the keys are not padded, and rows that do not read the
    reference run as a separate plain call over the target keys. On CUDA nothing changes: keys are padded to a
    multiple of 16 (aligned mask rows for the memory-efficient kernel) and masked with -inf."""
    B, S, H, D = q.shape
    if rows is not None and not q.is_cuda:
        rows = list(rows)
        others = [r for r in range(B) if r not in rows]
        if others:
            out = q.new_empty(B, S, H * D)
            if rows:
                sel = torch.as_tensor(rows, device=q.device)
                out[sel] = _masked_attention(q[sel], k_t[sel], v_t[sel], kv_ref, sizes, biases,
                                             list(range(len(rows))), row_biases)
            o = torch.as_tensor(others, device=q.device)
            plain = F.scaled_dot_product_attention(q[o].transpose(1, 2), k_t[o].transpose(1, 2), v_t[o].transpose(1, 2))
            out[o] = plain.transpose(1, 2).reshape(len(others), S, H * D)
            return out
    k_r, v_r = kv_ref
    k = torch.cat([k_r.expand(B, *k_r.shape[1:]).to(q.dtype), k_t], dim=1).transpose(1, 2)
    v = torch.cat([v_r.expand(B, *v_r.shape[1:]).to(q.dtype), v_t], dim=1).transpose(1, 2)
    n = k.shape[2]
    pad = (-n) % 16 if q.is_cuda else 0
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


_FP64_ARITH: dict[str, bool] = {}   # device.type -> can this backend compute in float64


def _fp64_env_override():
    """SNAPREF_FP64=0 forces the float32 correction, =1 forces float64, unset probes the backend."""
    raw = os.environ.get("SNAPREF_FP64", "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    return None


def _fp64_arith_ok(device) -> bool:
    """True only if the backend can actually compute in float64, not merely hold a float64 dtype
    (ArcAnima/SnapRef PR #5 by mouyase). MPS has no float64; Intel XPU accepts the cast but its float64
    arithmetic fails ("Required aspect fp64 is not supported on the device"). Probed once per device type with
    an elementwise op and a reduction on an 8-element tensor; a backend that supports float64 keeps it."""
    override = _fp64_env_override()
    if override is not None:
        return override
    key = getattr(device, "type", None) or str(device)
    hit = _FP64_ARITH.get(key)
    if hit is not None:
        return hit
    try:
        probe = torch.ones(8, device=device, dtype=torch.float64)
        float((probe * 2.0 + 1.0).sum().item())
        ok = True
    except (RuntimeError, NotImplementedError, TypeError):
        ok = False
    _FP64_ARITH[key] = ok
    return ok


def _correction_dtype(t):
    """float64 where the backend can run it, float32 everywhere else (APG only projects the update onto the
    per-sample direction of the cond x0 prediction; float32 is enough there)."""
    return torch.float64 if _fp64_arith_ok(t.device) else torch.float32


def _correct(guided, pos, correction, rescale_phi=0.5):
    """APG (drop the update's component parallel to the cond prediction) or CFG rescale, in x0 space
    (as ElectronGirl/ComfyUI-Anima-Ref-Adapter, Apache-2.0)."""
    dims = tuple(range(1, pos.ndim))
    if correction == "apg":
        cdt = _correction_dtype(pos)
        update = (guided - pos).to(cdt)
        unit = F.normalize(pos.to(cdt).flatten(1), dim=1).view_as(pos)
        return pos + (update - (update * unit).sum(dim=dims, keepdim=True) * unit).to(pos.dtype)
    if correction == "rescale":
        ratio = pos.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True)
        return rescale_phi * guided * ratio + (1 - rescale_phi) * guided
    return guided
