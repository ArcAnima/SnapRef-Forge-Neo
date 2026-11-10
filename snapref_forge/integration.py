"""Forge-native model patches; no ComfyUI runtime dependency."""
from __future__ import annotations

import math
from functools import wraps
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .core import attention, block_map, _rope, _correct

KEY = 'snapref_forge'
OFF = 'snapref_forge_off'


def cleanup_after(original):
    @wraps(original)
    def guarded(p,*args,**kwargs):
        try:
            return original(p,*args,**kwargs)
        finally:
            cleanup = getattr(p,'_snapref_cleanup',None)
            if cleanup is not None:
                del p._snapref_cleanup
                cleanup(p)
    guarded._snapref_guard = True
    return guarded


@torch.inference_mode()
def encode_image(image, vae, megapixels):
    scale = min(1., (megapixels * 1024**2 / (image.width*image.height))**.5)
    size = tuple(max(16, int(side*scale)//16*16) for side in image.size)
    image = image.resize(size, Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.asarray(image,dtype=np.float32)/255.).unsqueeze(0)
    ref = vae.first_stage_model.process_in(vae.encode(pixels)).detach().cpu()
    return ref.unsqueeze(2) if ref.ndim == 4 else ref


class Session:
    def __init__(self, adapter, refs, context, mode, weights, fidelity=1., in_uncond=False):
        if mode not in adapter.modes: raise ValueError(f'Adapter does not support mode {mode}')
        if not math.isfinite(fidelity) or not 0 <= fidelity <= 1: raise ValueError('Fidelity must be between 0 and 1')
        self.adapter, self.refs, self.context, self.mode = adapter, refs, context.detach().cpu(), mode
        self.biases = [math.log(w) for w in weights]
        self.fidelity, self.in_uncond = fidelity, in_uncond
        self.cache = {}
        self.stream = self.record = self.kv = self.rows = self.row_biases = None

    def clear(self):
        self.cache.clear()
        self.stream = self.record = self.kv = self.rows = self.row_biases = None

    def linear(self, original, streams):
        def forward(x, *args, **kwargs):
            out = original(x, *args, **kwargs)
            if self.stream not in streams: return out
            down, up, scale = streams[self.stream]
            key = (id(down),x.device,x.dtype)
            if key not in self.cache: self.cache[key] = down.to(x), up.to(x)
            down, up = self.cache[key]
            return out + F.linear(F.linear(x,down),up)*scale
        return forward

    def self_attention(self, attn, original, index):
        def forward(x, context=None, rope_emb=None, transformer_options=None):
            options = transformer_options or {}
            if self.record is None and self.kv is None:
                return original(x,context,rope_emb=rope_emb,transformer_options=options)
            q,k,v = attn.compute_qkv(x,context,rope_emb=rope_emb)
            if self.record is not None:
                self.record[index] = k,v
                return attn.compute_attention(q,k,v,transformer_options=options)
            out = attention(q,k,v,self.kv[index],self.sizes,self.biases,self.rows,self.row_biases)
            return attn.output_dropout(attn.output_proj(out))
        return forward

    @torch.inference_mode()
    def capture(self, dm, ref, x):
        from backend.utils import pad_to_patch_size
        ref = pad_to_patch_size(ref.to(x),(dm.patch_temporal,dm.patch_spatial,dm.patch_spatial))
        hidden,_,_ = dm.prepare_embedded_sequence(ref,padding_mask=None)
        _,_,h,w,_ = hidden.shape
        h0,w0 = (round((target//dm.patch_spatial-source)/2) for target,source in zip(x.shape[-2:],(h,w)))
        hh,ww = torch.meshgrid(torch.arange(h,device=x.device)+h0,torch.arange(w,device=x.device)+w0,indexing='ij')
        rope = _rope(dm.pos_embedder,torch.full_like(hh,self.adapter.ref_t).flatten(),hh.flatten(),ww.flatten())
        emb,adaln = dm.t_embedder[1](dm.t_embedder[0](torch.zeros(1,1,device=x.device)).to(hidden.dtype))
        emb = dm.t_embedding_norm(emb)
        if self.adapter.mode_embed is not None:
            emb = emb + self.adapter.mode_embed[self.adapter.modes.index(self.mode)].to(emb)[None,None,:]
        self.record = [None]*len(dm.blocks)
        self.stream = 'ref'
        hidden = hidden.float() if hidden.dtype == torch.float16 else hidden
        for block in dm.blocks:
            hidden = block(hidden,emb,self.context.to(x),rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0),
                           adaln_lora_B_T_3D=adaln,extra_per_block_pos_emb=None,transformer_options={})
        kv = self.record
        self.record = None
        return kv

    def patch(self, unet):
        dm = unet.get_model_object('diffusion_model')
        mapping = block_map(self.adapter.indices,len(dm.blocks))
        m = unet.clone()
        # Resolve before mutating the patcher so unsupported checkpoints fail atomically.
        targets = []
        for name,streams in self.adapter.lora.items():
            if name.startswith('blocks.'):
                _,index,suffix = name.split('.',2)
                name = f'blocks.{mapping[int(index)]}.{suffix}'
            attr = f'diffusion_model.{name}.forward'
            targets.append((attr,self.linear(m.get_model_object(attr),streams)))
        for attr,fn in targets: m.add_object_patch(attr,fn)
        for index,block in enumerate(dm.blocks):
            attr = f'diffusion_model.blocks.{index}.self_attn.forward'
            m.add_object_patch(attr,self.self_attention(block.self_attn,m.get_model_object(attr),index))
        original = m.get_model_object('diffusion_model.forward')
        token = object()

        def forward(x,timesteps,context,padding_mask=None,**kwargs):
            options = kwargs.get('transformer_options') or {}
            if options.get(KEY) is not token:
                return original(x,timesteps,context,padding_mask=padding_mask,**kwargs)
            try:
                flags = options.get('cond_or_uncond') or [0]
                count = x.shape[0]//len(flags)
                positive = [r for i,f in enumerate(flags) if f == 0 for r in range(i*count,(i+1)*count)]
                negative = [r for r in range(x.shape[0]) if r not in positive]
                self.rows = list(range(x.shape[0])) if self.in_uncond or self.fidelity < 1 else positive
                if options.get(OFF): self.rows = []
                if self.rows:
                    key = ('kv',tuple(x.shape[-2:]),x.device,x.dtype)
                    if key not in self.cache:
                        for old in list(self.cache):
                            if old[0] == 'kv': del self.cache[old]
                        records = [self.capture(dm,ref,x) for ref in self.refs]
                        sizes = [r[0][0].shape[1] for r in records]
                        if len(records) == 1:
                            combined = records.pop()
                        else:
                            combined = []
                            for _ in dm.blocks:
                                parts = [r.pop(0) for r in records]
                                combined.append((torch.cat([p[0] for p in parts],1),torch.cat([p[1] for p in parts],1)))
                                del parts
                        del records
                        self.cache[key] = combined,sizes
                    self.kv,self.sizes = self.cache[key]
                    self.row_biases = [math.log(max(1-self.fidelity,1e-6)) if r in negative and not self.in_uncond else 0.
                                       for r in self.rows]
                self.stream = 'tgt'
                return original(x,timesteps,context,padding_mask=padding_mask,**kwargs)
            except BaseException:
                self.clear()
                raise
            finally:
                self.stream = self.record = self.kv = self.rows = self.row_biases = None

        m.add_object_patch('diffusion_model.forward',forward)
        m.set_transformer_option(KEY,token)
        dtype_size = torch.empty((),dtype=unet.model.computation_dtype).element_size()
        kv_bytes = sum(2*len(dm.blocks)*(r.shape[-2]//dm.patch_spatial)*(r.shape[-1]//dm.patch_spatial)*dm.blocks[0].x_dim*dtype_size for r in self.refs)
        weight_bytes = sum((a.numel()+b.numel())*dtype_size for streams in self.adapter.lora.values() for a,b,_ in streams.values())
        m.add_extra_preserved_memory_during_sampling(kv_bytes+kv_bytes//len(dm.blocks)+weight_bytes)
        return m


def add_reference_guidance(unet, guidance, correction, end, rescale_phi):
    from backend.sampling.sampling_function import calc_cond_uncond_batch
    def post_cfg(args):
        guided,pos = args['denoised'],args['cond_denoised']
        if guidance != 1 and (end >= 1 or float(args['sigma'].flatten()[0]) >= args['model'].predictor.percent_to_sigma(end)-1e-5):
            options = args['model_options'].copy()
            options['transformer_options'] = {**options.get('transformer_options',{}),OFF:True}
            if args['uncond'] and (not math.isclose(args['cond_scale'],1.) or options.get('disable_cfg1_optimization')):
                _,base = calc_cond_uncond_batch(args['model'],[],args['uncond'],args['input'],args['sigma'],options)
                ref = args['uncond_denoised']
            else:
                base,_ = calc_cond_uncond_batch(args['model'],args['cond'],None,args['input'],args['sigma'],options)
                ref = pos
            guided = guided+(guidance-1)*(ref-base)
        return _correct(guided,pos,correction,rescale_phi)
    unet.set_model_sampler_post_cfg_function(post_cfg)
    return unet
