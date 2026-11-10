# SnapRef for Forge Neo

Official native Forge Neo extension for SnapRef, maintained alongside ArcAnima/SnapRef.
No ComfyUI installation or third-party reference extension is required.

## Install

Copy this folder into `<Forge Neo>/extensions/snapref`, then restart Forge Neo.
Use the `neo` branch of https://github.com/Haoming02/sd-webui-forge-classic.
The extension adds a **SnapRef** panel in txt2img and img2img.

1. Set up an Anima checkpoint, Qwen3-0.6B text encoder and Qwen-Image VAE as usual.
2. Enable the SnapRef panel and select an adapter. Download with the panel's button, or put
   a released `snapref*.safetensors` file in this extension's `models/` or Forge's `models/Lora/` and refresh.
3. Add up to four pictures, choose the mode, and describe the scene in your prompt.
4. Keep Forge's **[Anima] Enable Reference** setting off. Do not add SnapRef as a normal `<lora:...>` tag.

## Supported controls

- SnapRef v2a stable, v1a and v1b, including row-scaled FP8 adapter files. FP8 adapters are dequantized to the
  current inference dtype; this reduces download/storage size, not necessarily inference VRAM.
- Four separately weighted references of one character, or a side-by-side composite for multiple characters.
  Side by side uses equal strengths for all nonzero references.
- Character & Style / Character, Fidelity (0–1), per-reference Strength (0–2), reference resolution and caption.
- Advanced reference in negative branch, extra reference guidance, guidance end, APG / rescale / no correction.
  Extra guidance overrides Fidelity and requires an additional model call per step.
- Native 28-block Anima and the known 28-to-40-block Anima-2.9B layout. Other layouts fail explicitly.
- Normal CFG, CFG=1, mixed positive/negative batches and Hires resolution changes. At CFG=1, Fidelity has
  no effect because the negative branch is skipped. PNG Info restores settings; reference pictures must be reloaded.

Reference K/V and GPU adapter copies are scoped to a sampling pass and released afterwards, on failures,
and before a new request. Four 1MP references use substantially more VRAM than one; lower Reference size
to 0.5 MP on smaller GPUs. Changes to Forge's attention internals can require an extension update.

Run the CPU checks with `python -m unittest discover -s tests -v`.

Code: Apache-2.0. Model weights remain subject to their own CircleStone-derived license.
See NOTICE and the model cards at https://huggingface.co/0ppxnhximxr/snapref-v2a.
