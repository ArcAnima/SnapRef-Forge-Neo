# SnapRef for Forge Neo

[![Civitai](https://img.shields.io/badge/Civitai-SnapRef-1769ff?style=for-the-badge)](https://civitai.com/models/3028191/snapref?modelVersionId=3457619)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Weights-ffcc4d?style=for-the-badge)](https://huggingface.co/0ppxnhximxr/snapref-v2a)
[![Original SnapRef GitHub](https://img.shields.io/badge/GitHub-Original_SnapRef-24292f?style=for-the-badge&logo=github)](https://github.com/ArcAnima/SnapRef)

Official native Forge Neo extension for SnapRef, maintained alongside ArcAnima/SnapRef.
No ComfyUI installation or third-party reference extension is required.

## Install

Use the `neo` branch of https://github.com/Haoming02/sd-webui-forge-classic.
The extension adds a **SnapRef** panel in txt2img and img2img.

In Forge Neo, open **Extensions → Install from URL** and paste:

```text
https://github.com/ArcAnima/SnapRef-Forge-Neo
```

Click **Install**, then restart Forge Neo from the **Installed** tab.
For updates, use **Extensions → Installed → Check for updates**, apply the update, and restart.
For manual installation, copy this repository into `<Forge Neo>/extensions/snapref` and restart.

1. Set up an Anima checkpoint, Qwen3-0.6B text encoder and Qwen-Image VAE as usual.
2. Enable the SnapRef panel and select an adapter. Download with the panel's button, or put
   a released `snapref*.safetensors` file in this extension's `models/` or Forge's `models/Lora/` and refresh.
   The panel preselects an installed v2s PRE1 or v2a file (newest first, FP8 first), otherwise
   `snapref_v2s_pre1_fp8.safetensors`.
3. Add up to eight pictures (5–8 under **More references**) and choose the mode. In **Character & Style** and
   **Character**, describe only the scene in your prompt; in **Style**, write the character and the scene.
   The line under Mode repeats what to write for the selected mode.
4. Keep Forge's **[Anima] Enable Reference** setting off. Do not add SnapRef as a normal `<lora:...>` tag.

## Style mode (v2s PRE1)

SnapRef v2s PRE1 is v2a stable plus a new **Style** mode (preview). Style takes only the art style of the
pictures: line weight, color palette, lighting and shading. The character comes from your prompt.
Character & Style and Character give bit-identical results to v2a; the Style parts of the file switch on only in Style.

1. Select `snapref_v2s_pre1.safetensors` or `snapref_v2s_pre1_fp8.safetensors`.
2. Choose **Style** and put one or more pictures in the art style you want into the reference slots. Several
   pictures should be by the same artist; each keeps its own Strength.
3. Write the character and the scene in the prompt, as you would without a reference. Leave Reference caption empty.
4. Keep Strength at `1.0` to start; lower a picture's Strength (for example `0.35`) when it pulls too much of its
   own look into the result. Fidelity, Reference in negative branch and extra reference guidance are available in
   Style too.

With a v2a or older adapter, the line under Mode warns that the adapter has no Style mode, and generation stops
with an error that names the adapter before any picture is encoded. Choose a v2s file or another mode.

| file | size | SHA-256 |
|---|---|---|
| `snapref_v2s_pre1.safetensors` | 543,395,968 bytes | `95f91797827cbf5efd86e29fac5b2646df1505d46acb20f913877b3c9a429609` |
| `snapref_v2s_pre1_fp8.safetensors` | 278,127,904 bytes | `55a429f078d74f3e144a89d14aaaaedbd9f51c2ac598728f8f0135a6e67a1600` |

Both files are published at https://huggingface.co/0ppxnhximxr/snapref-v2s. The Download button checks their size
and SHA-256 and removes a download that does not match. While that repository is not public, Download reports an
error; put the file in `models/` yourself and refresh.

The FP8 file stores the Style branch as FP8 too. Its Character & Style and Character results are identical to the
v2a FP8 file (the same tensors); Style results can differ slightly from the BF16 file.

Style limits:

- Style is a preview. Some pictures change the palette, lighting and line quality more than the way faces are drawn.
- A little of a picture's coloring can carry over, such as a dark outfit or a skin tone. Name colors in the prompt
  when they matter.
- Strength above `1.0` brings in more of the pictures' character and follows the prompt less.
- One mode per generation: a character reference and a style reference together are not supported.
- Use pictures you have the right to use. Text or signature-like marks in a picture can be copied; inspect outputs
  before sharing them.

## Supported controls

- SnapRef v2s PRE1 (v2a stable + Style), v2a stable, v1a and v1b, including row-scaled FP8 adapter files. FP8 adapters
  are dequantized to the current inference dtype; this reduces download/storage size, not necessarily inference VRAM.
- Up to eight separately weighted references of one character (or, in Style, by one artist), or a side-by-side
  composite for multiple characters. Side by side uses equal strengths for all nonzero references.
- Character & Style / Character / Style, Fidelity (0–1), per-reference Strength (0–2), reference resolution and caption.
- Mode-gated branch LoRAs as in the ComfyUI node: a reference-side branch runs only while the reference is encoded in
  its mode, a target-side branch only in the batch rows that read the reference.
- Advanced reference in negative branch, extra reference guidance, guidance end, APG / rescale / no correction.
  Extra guidance overrides Fidelity and requires an additional model call per step.
- Native 28-block Anima and the known 28-to-40-block Anima-2.9B layout. Other layouts fail explicitly.
- Normal CFG, CFG=1, mixed positive/negative batches and Hires resolution changes. At CFG=1, Fidelity has
  no effect because the negative branch is skipped. PNG Info restores settings; reference pictures must be reloaded.

Reference K/V and GPU adapter copies are scoped to a sampling pass and released afterwards, on failures,
and before a new request. Four 1MP references use substantially more VRAM than one; lower Reference size
to 0.5 MP on smaller GPUs. Style also keeps the Style branch on the GPU during sampling (about 160 MB at BF16).
Changes to Forge's attention internals can require an extension update.

Run the CPU checks with `python -m unittest discover -s tests -v`.

Code: Apache-2.0. Model weights remain subject to their own CircleStone-derived license.
See NOTICE and the model cards at https://huggingface.co/0ppxnhximxr/snapref-v2a and
https://huggingface.co/0ppxnhximxr/snapref-v2s.
