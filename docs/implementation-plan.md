# Official SnapRef for Forge Neo

Scope: standalone native Forge Neo extension, reusing SnapRef's Apache-2.0 inference math. Do not modify the existing ComfyUI implementation or bundle Forge.

1. Test the actual adapter loader and attention math on CPU: BF16/row-scaled FP8, weighted references, Fidelity in mixed CFG batches, 28/40 block mapping and invalid inputs.
2. Implement a Forge model-patcher integration, per-sampling cache, four reference images and weights, Character/Character & Style, Fidelity, separate/stitch layouts and advanced reference guidance/correction. Preserve vanilla generation when disabled and release GPU tensors after sampling/errors.
3. Add native txt2img/img2img UI, PNG infotext settings, local model discovery and installation documentation with licenses. No third-party extension is required.
4. Install pinned Forge Neo on the user's existing Lambda A10; download Anima and released SnapRef weights; run meaningful real inference for FP8, four references, Fidelity/strength and a subsequent disabled generation. Inspect output images.
5. Expose only the requested WebUI port in Lambda's existing firewall, verify public HTTP, and return its URL. Keep the instance running for the user's testing; do not terminate it.

Review focus: Forge attention signatures differ from ComfyUI; target LoRA must remain active in negative passes; FP8 per-row scales; cache isolation after errors/model changes; CFG=1 and batched cond/uncond; Hires resolution changes; four-reference VRAM on A10.
