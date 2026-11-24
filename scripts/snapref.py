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
from snapref_forge.core import MAX_REFERENCES, Adapter, select_references, read_meta, has_style
from snapref_forge.integration import Session, encode_image, add_reference_guidance, cleanup_after

# Forge's post_sample/postprocess callbacks are skipped when a sampler raises.
if not getattr(processing.process_images_inner,'_snapref_guard',False):
    processing.process_images_inner = cleanup_after(processing.process_images_inner)

RELEASES = {
    'snapref_v2s_pre1.safetensors':'0ppxnhximxr/snapref-v2s',
    'snapref_v2s_pre1_fp8.safetensors':'0ppxnhximxr/snapref-v2s',
    'snapref_v2a.safetensors':'0ppxnhximxr/snapref-v2a',
    'snapref_v2a_fp8.safetensors':'0ppxnhximxr/snapref-v2a',
    'snapref_v1a.safetensors':'0ppxnhximxr/snapref-v1a',
    'snapref_v1a_fp8.safetensors':'0ppxnhximxr/snapref-v1a',
    'snapref_v1b.safetensors':'0ppxnhximxr/snapref-v1b',
    'snapref_v1b_fp8.safetensors':'0ppxnhximxr/snapref-v1b',
}
# size in bytes and SHA-256 of released files; a download that does not match is removed
CHECKSUMS = {
    'snapref_v2s_pre1.safetensors':(543395968,'95f91797827cbf5efd86e29fac5b2646df1505d46acb20f913877b3c9a429609'),
    'snapref_v2s_pre1_fp8.safetensors':(278127904,'55a429f078d74f3e144a89d14aaaaedbd9f51c2ac598728f8f0135a6e67a1600'),
}
STYLE_RELEASES = {'snapref_v2s_pre1.safetensors','snapref_v2s_pre1_fp8.safetensors'}   # released files with Style
# Newest first, FP8 first within a version (the previous default was the v2a FP8 file). v2s keeps Character & Style
# and Character bit-identical to v2a and adds Style.
DEFAULT_ORDER = ('snapref_v2s_pre1_fp8.safetensors','snapref_v2s_pre1.safetensors',
                 'snapref_v2a_fp8.safetensors','snapref_v2a.safetensors')

MODES = {'Character & Style':'cs','Character':'char','Style':'style'}   # as the ComfyUI SnapRef node
MODE_HINTS = {
    'Character & Style':'**Prompt:** write only the scene. The character and its art style come from the pictures.',
    'Character':'**Prompt:** write the scene; add artist tags for the style. Only the character comes from the pictures.',
    'Style':'**Prompt:** write the character and the scene. Only the art style comes from the pictures '
            '(one or more artworks by the same artist); keep Reference caption empty.',
}
STYLE_MISSING = ('**This adapter has no Style mode.** Choose snapref_v2s_pre1 (or its FP8 file) '
                 'or another adapter with a style branch.')


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


def default_adapter(found):
    """The newest release already installed, else the newest release (downloaded with the panel's button)."""
    return next((name for name in DEFAULT_ORDER if name in found),DEFAULT_ORDER[0])


def style_support(name,found=None):
    """True / False: the adapter can / cannot run Style; None: unknown (a file that cannot be read).
    Installed files are judged by their header metadata, released names that are not installed by the release list."""
    path = (model_files() if found is None else found).get(name)
    if path is None:
        return (name in STYLE_RELEASES) if name in RELEASES else None
    try:
        return has_style(read_meta(path))
    except Exception:
        return None   # still downloading or damaged: the run reports any problem


def mode_hint(mode,support=None):
    """One line under the mode: what to write in the prompt, or why Style cannot run with this adapter."""
    if mode == 'Style' and support is False: return STYLE_MISSING
    return MODE_HINTS.get(mode,MODE_HINTS['Character & Style'])


def update_hint(mode,adapter):
    return gr.update(value=mode_hint(mode,style_support(adapter)))


def verify(path,name):
    """Size and SHA-256 of a released file; a mismatching file is removed so it is never loaded."""
    if name not in CHECKSUMS: return
    size,digest = CHECKSUMS[name]
    path = Path(path)
    h = hashlib.sha256()
    if path.stat().st_size == size:
        with open(path,'rb') as handle:
            for chunk in iter(lambda:handle.read(1<<24),b''): h.update(chunk)
    if path.stat().st_size != size or h.hexdigest() != digest:
        path.unlink()
        raise gr.Error(f'SnapRef: the download of {name} is incomplete or damaged (size or SHA-256 mismatch) '
                       'and was removed. Download it again.')


def download(name):
    if name not in RELEASES: raise gr.Error('Choose a released SnapRef filename to download.')
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(RELEASES[name],name,local_dir=str(ROOT/'models'))
    except Exception as exc:   # e.g. 401 while a repository is not public yet, or offline
        raise gr.Error(f'SnapRef: could not download {name} from https://huggingface.co/{RELEASES[name]} '
                       f'({type(exc).__name__}). If it is not public yet or you are offline, download the file '
                       "yourself, put it in this extension's models/ or Forge's models/Lora/, and refresh.") from exc
    verify(path,name)
    return gr.update(choices=list(dict.fromkeys([*model_files(),*RELEASES])),value=name)


class SnapRefScript(scripts.Script):
    def __init__(self):
        self.session = None
        self.original_unet = self.patched_unet = None

    def title(self): return 'SnapRef'
    def show(self,is_img2img): return scripts.AlwaysVisible

    def ui(self,is_img2img):
        found = model_files()
        with InputAccordion(False,label='SnapRef') as enabled:
            with gr.Group(elem_classes=['snapref-content']):
                with gr.Row(equal_height=False,elem_classes=['snapref-toolbar']):
                    adapter = gr.Dropdown(list(dict.fromkeys([*found,*RELEASES])),value=default_adapter(found),label='SnapRef adapter')
                    refresh = gr.Button('Refresh',scale=0,min_width=76,size='sm')
                    fetch = gr.Button('Download',scale=0,min_width=88,size='sm')
                refresh.click(lambda:gr.update(choices=list(dict.fromkeys([*model_files(),*RELEASES]))),outputs=[adapter],queue=False)
                fetch.click(download,inputs=[adapter],outputs=[adapter])
                with gr.Row(equal_height=False,elem_classes=['snapref-options']):
                    mode = gr.Radio(list(MODES),value='Character & Style',label='Mode')
                    layout = gr.Radio(['Separate references','Side by side'],value='Separate references',label='Combine references')
                    fidelity = gr.Slider(0,1,value=1.,step=.05,label='Fidelity')
                hint = gr.Markdown(mode_hint('Character & Style'),elem_classes=['snapref-hint'])
                mode.change(update_hint,inputs=[mode,adapter],outputs=[hint],queue=False)
                adapter.change(update_hint,inputs=[mode,adapter],outputs=[hint],queue=False)
                images,weights = [],[]

                def reference_slots(first,last):
                    with gr.Row(elem_classes=['snapref-references']):
                        for i in range(first,last):
                            with gr.Column(min_width=190,elem_classes=['snapref-reference']):
                                gr.Markdown(f'Reference {i+1}',elem_classes=['snapref-ref-title'])
                                images.append(gr.Image(type='pil',label=f'Reference {i+1}',sources=['upload','clipboard'],height=200,show_label=False))
                                weights.append(gr.Slider(0,2,value=1.,step=.05,label=f'Strength {i+1}'))
                reference_slots(0,4)
                # 5-8 (as the ComfyUI node): each adds about 0.9 GB of reference K/V at 1 MP on the 2B model
                with gr.Accordion('More references (5-8)',open=False):
                    reference_slots(4,MAX_REFERENCES)
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
        pairs = [v for pair in zip(images,weights) for v in pair]
        # references 5-8 go last, so argument lists written for four references (API calls) keep their places
        return [enabled,adapter,mode,layout,fidelity,mp,*pairs[:8],
                caption,in_uncond,rg_on,rg,end,correction,phi,*pairs[8:]]

    def release(self,p=None):
        if self.session is not None:
            self.session.clear()
            self.session = None
        if p is not None and p.sd_model.forge_objects.unet is self.patched_unet:
            p.sd_model.forge_objects.unet = self.original_unet
        self.original_unet = self.patched_unet = None

    def process_before_every_sampling(self,p,enabled,adapter,mode,layout,fidelity,mp,
                                      ref1,w1,ref2,w2,ref3,w3,ref4,w4,caption='',in_uncond=False,
                                      rg_on=False,rg=4.5,end=1.,correction='apg',phi=.5,
                                      ref5=None,w5=1.,ref6=None,w6=1.,ref7=None,w7=1.,ref8=None,w8=1.,*args,**kwargs):
        self.release(p)
        if not enabled: return
        p._snapref_cleanup = self.release
        try:
            unet = p.sd_model.forge_objects.unet
            dm = unet.get_model_object('diffusion_model')
            if dm.__class__.__name__ != 'Anima': raise ValueError('SnapRef requires an Anima checkpoint.')
            if mode not in MODES: raise ValueError('Invalid SnapRef mode')
            for value,low,high in [(mp,.1,2),(fidelity,0,1),(rg,1,10),(end,0,1),(phi,0,1)]:
                if not math.isfinite(float(value)) or not low <= float(value) <= high: raise ValueError('Invalid SnapRef setting')
            if correction not in ('apg','rescale','none'): raise ValueError('Invalid guidance correction')
            if shared.opts.anima_do_reference: raise ValueError('Turn off Settings → Stable Diffusion → [Anima] Enable Reference while using SnapRef.')
            pictures = [ref1,ref2,ref3,ref4,ref5,ref6,ref7,ref8]
            strengths = [w1,w2,w3,w4,w5,w6,w7,w8]
            refs,weights = select_references(pictures,strengths,layout)
            if not refs: return
            path = model_files().get(adapter)
            if path is None: raise ValueError('Download the selected SnapRef adapter, then refresh.')
            ad = Adapter(path)
            ad.check_mode(MODES[mode],path.name)   # Style needs a style branch: say so before any encoding
            latents = [encode_image(image,p.sd_model.forge_objects.vae,float(mp)) for image in refs]
            context = p.sd_model.get_learned_conditioning(SdConditioning([caption],is_negative_prompt=True))[0]
            dynamic_args.ref_latents.clear()
            self.session = Session(ad,latents,context,MODES[mode],weights,
                                   float(fidelity),bool(in_uncond or rg_on))
            patched = self.session.patch(unet)
            if rg_on: patched = add_reference_guidance(patched,float(rg),correction,float(end),float(phi))
            self.original_unet,self.patched_unet = unet,patched
            p.sd_model.forge_objects.unet = patched
            settings = {'Adapter':path.name,'Mode':mode,'Layout':layout,'Fidelity':float(fidelity),'Reference MP':float(mp),
                        'Reference caption':caption,'Reference in negative':bool(in_uncond),'Extra guidance':bool(rg_on),
                        'Guidance':float(rg),'Guidance end':float(end),'Correction':correction,'Rescale phi':float(phi)}
            # Strength 5-8 only when used: infotext of four-reference generations stays as it was
            used = 8 if any(r is not None for r in pictures[4:]) else 4
            settings.update({f'Strength {i+1}':float(w) for i,w in enumerate(strengths[:used])})
            settings['References'] = len(refs)
            settings['Reference hashes'] = ','.join(hashlib.sha256(r.tobytes()).hexdigest()[:12] for r in refs)
            p.extra_generation_params.update({'SnapRef '+k:v for k,v in settings.items()})
        except Exception as exc:
            self.release(p)
            shared.state.interrupt()
            message = str(exc)
            raise gr.Error(message if message.startswith('SnapRef:') else f'SnapRef: {message}') from exc

    def post_sample(self,p,ps,*args): self.release(p)
    def postprocess(self,p,processed,*args): self.release(p)
