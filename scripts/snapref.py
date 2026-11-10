"""SnapRef's official native panel for Forge Neo."""
from pathlib import Path
import hashlib
import math
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import gradio as gr
from modules import scripts, shared, processing
from modules.prompt_parser import SdConditioning
from modules.ui_components import InputAccordion
from backend import memory_management
from backend.args import dynamic_args
from snapref_forge.core import Adapter, select_references
from snapref_forge.integration import Session, encode_image, add_reference_guidance, cleanup_after

# Forge's post_sample/postprocess callbacks are skipped when a sampler raises.
if not getattr(processing.process_images_inner,'_snapref_guard',False):
    processing.process_images_inner = cleanup_after(processing.process_images_inner)

RELEASES = {
    'snapref_v2a.safetensors':'0ppxnhximxr/snapref-v2a',
    'snapref_v2a_fp8.safetensors':'0ppxnhximxr/snapref-v2a',
    'snapref_v1a.safetensors':'0ppxnhximxr/snapref-v1a',
    'snapref_v1a_fp8.safetensors':'0ppxnhximxr/snapref-v1a',
    'snapref_v1b.safetensors':'0ppxnhximxr/snapref-v1b',
    'snapref_v1b_fp8.safetensors':'0ppxnhximxr/snapref-v1b',
}


def model_files():
    folder = ROOT/'models'
    folder.mkdir(exist_ok=True)
    folders = [folder,Path(shared.cmd_opts.lora_dir),*(Path(p) for p in getattr(shared.cmd_opts,'lora_dirs',[]) or [])]
    found = {}
    for folder in folders:
        if folder.is_dir():
            for p in sorted(folder.rglob('*.safetensors')):
                if 'snapref' in p.name.lower(): found.setdefault(p.name,p)
    return found


def download(name):
    if name not in RELEASES: raise gr.Error('Choose a released SnapRef filename to download.')
    from huggingface_hub import hf_hub_download
    hf_hub_download(RELEASES[name],name,local_dir=str(ROOT/'models'))
    return gr.update(choices=list(dict.fromkeys([*model_files(),*RELEASES])),value=name)


class SnapRefScript(scripts.Script):
    def __init__(self):
        self.session = None
        self.original_unet = self.patched_unet = None

    def title(self): return 'SnapRef'
    def show(self,is_img2img): return scripts.AlwaysVisible

    def ui(self,is_img2img):
        with InputAccordion(False,label='SnapRef') as enabled:
            with gr.Group(elem_classes=['snapref-content']):
                with gr.Row(equal_height=False,elem_classes=['snapref-toolbar']):
                    adapter = gr.Dropdown(list(dict.fromkeys([*model_files(),*RELEASES])),value='snapref_v2a_fp8.safetensors',label='SnapRef adapter')
                    refresh = gr.Button('Refresh',scale=0,min_width=76,size='sm')
                    fetch = gr.Button('Download',scale=0,min_width=88,size='sm')
                refresh.click(lambda:gr.update(choices=list(dict.fromkeys([*model_files(),*RELEASES]))),outputs=[adapter],queue=False)
                fetch.click(download,inputs=[adapter],outputs=[adapter])
                with gr.Row(equal_height=False,elem_classes=['snapref-options']):
                    mode = gr.Radio(['Character & Style','Character'],value='Character & Style',label='Mode')
                    layout = gr.Radio(['Separate references','Side by side'],value='Separate references',label='Combine references')
                    fidelity = gr.Slider(0,1,value=1.,step=.05,label='Fidelity')
                images,weights = [],[]
                with gr.Row(elem_classes=['snapref-references']):
                    for i in range(4):
                        with gr.Column(min_width=190,elem_classes=['snapref-reference']):
                            gr.Markdown(f'Reference {i+1}',elem_classes=['snapref-ref-title'])
                            images.append(gr.Image(type='pil',label=f'Reference {i+1}',sources=['upload','clipboard'],height=200,show_label=False))
                            weights.append(gr.Slider(0,2,value=1.,step=.05,label=f'Strength {i+1}'))
                with gr.Accordion('Advanced',open=False):
                    with gr.Row():
                        mp = gr.Slider(.1,2,value=1.,step=.1,label='Reference size (MP)')
                        caption = gr.Textbox(label='Reference caption',value='')
                    with gr.Row():
                        in_uncond = gr.Checkbox(False,label='Reference in negative branch')
                        rg_on = gr.Checkbox(False,label='Extra reference guidance')
                    with gr.Row():
                        rg = gr.Slider(1,10,value=4.5,step=.1,label='Reference guidance')
                        end = gr.Slider(0,1,value=1.,step=.05,label='Guidance end')
                    with gr.Row():
                        correction = gr.Radio(['apg','rescale','none'],value='apg',label='Guidance correction')
                        phi = gr.Slider(0,1,value=.5,step=.05,label='Rescale phi')
        named = [(adapter,'Adapter'),(mode,'Mode'),(layout,'Layout'),(fidelity,'Fidelity'),(mp,'Reference MP'),
                 (caption,'Reference caption'),(in_uncond,'Reference in negative'),(rg_on,'Extra guidance'),
                 (rg,'Guidance'),(end,'Guidance end'),(correction,'Correction'),(phi,'Rescale phi')]
        named += [(w,f'Strength {i+1}') for i,w in enumerate(weights)]
        self.infotext_fields = [(component,'SnapRef '+name) for component,name in named]
        self.infotext_fields.append((enabled,lambda d:'SnapRef Adapter' in d))
        return [enabled,adapter,mode,layout,fidelity,mp,*[v for pair in zip(images,weights) for v in pair],
                caption,in_uncond,rg_on,rg,end,correction,phi]

    def release(self,p=None):
        if self.session is not None:
            self.session.clear()
            self.session = None
        if p is not None and p.sd_model.forge_objects.unet is self.patched_unet:
            p.sd_model.forge_objects.unet = self.original_unet
        self.original_unet = self.patched_unet = None

    def process_before_every_sampling(self,p,enabled,adapter,mode,layout,fidelity,mp,
                                      ref1,w1,ref2,w2,ref3,w3,ref4,w4,caption='',in_uncond=False,
                                      rg_on=False,rg=4.5,end=1.,correction='apg',phi=.5,*args,**kwargs):
        self.release(p)
        if not enabled: return
        p._snapref_cleanup = self.release
        try:
            unet = p.sd_model.forge_objects.unet
            dm = unet.get_model_object('diffusion_model')
            if dm.__class__.__name__ != 'Anima': raise ValueError('SnapRef requires an Anima checkpoint.')
            if mode not in ('Character & Style','Character'): raise ValueError('Invalid SnapRef mode')
            for value,low,high in [(mp,.1,2),(fidelity,0,1),(rg,1,10),(end,0,1),(phi,0,1)]:
                if not math.isfinite(float(value)) or not low <= float(value) <= high: raise ValueError('Invalid SnapRef setting')
            if correction not in ('apg','rescale','none'): raise ValueError('Invalid guidance correction')
            if shared.opts.anima_do_reference: raise ValueError('Turn off Settings → Stable Diffusion → [Anima] Enable Reference while using SnapRef.')
            refs,weights = select_references([ref1,ref2,ref3,ref4],[w1,w2,w3,w4],layout)
            if not refs: return
            path = model_files().get(adapter)
            if path is None: raise ValueError('Download the selected SnapRef adapter, then refresh.')
            ad = Adapter(path)
            latents = [encode_image(image,p.sd_model.forge_objects.vae,float(mp)) for image in refs]
            context = p.sd_model.get_learned_conditioning(SdConditioning([caption],is_negative_prompt=True))[0]
            dynamic_args.ref_latents.clear()
            self.session = Session(ad,latents,context,'char' if mode=='Character' else 'cs',weights,
                                   float(fidelity),bool(in_uncond or rg_on))
            patched = self.session.patch(unet)
            if rg_on: patched = add_reference_guidance(patched,float(rg),correction,float(end),float(phi))
            self.original_unet,self.patched_unet = unet,patched
            p.sd_model.forge_objects.unet = patched
            settings = {'Adapter':path.name,'Mode':mode,'Layout':layout,'Fidelity':float(fidelity),'Reference MP':float(mp),
                        'Reference caption':caption,'Reference in negative':bool(in_uncond),'Extra guidance':bool(rg_on),
                        'Guidance':float(rg),'Guidance end':float(end),'Correction':correction,'Rescale phi':float(phi)}
            settings.update({f'Strength {i+1}':float(w) for i,w in enumerate([w1,w2,w3,w4])})
            settings['References'] = len(refs)
            settings['Reference hashes'] = ','.join(hashlib.sha256(r.tobytes()).hexdigest()[:12] for r in refs)
            p.extra_generation_params.update({'SnapRef '+k:v for k,v in settings.items()})
        except Exception as exc:
            self.release(p)
            shared.state.interrupt()
            raise gr.Error(f'SnapRef: {exc}') from exc

    def post_sample(self,p,ps,*args): self.release(p)
    def postprocess(self,p,processed,*args): self.release(p)
