"""Mode-gated branch LoRAs (SnapRef v2s and later), as the ComfyUI SnapRef node and train/style_branch.py:
a reference-stream branch runs while the reference is encoded in one of its modes, a target-stream branch only in
target forwards that read that mode's reference (only in the batch rows that read it). cs / char activate no
branch, so their outputs are bit-identical to an adapter file without branch keys.

The tiny model in tiny_anima.py stands in for Forge's Anima; the real Session.patch / capture / forward run on it.
An independent oracle (style_branch._forward and asym.encode_reference / target_forward written out, on the raw file
tensors) checks the tensors. With SNAPREF_TRAIN_DIR set to the SnapRef research train/ folder, the training code
itself (style_branch.load_adapter, asym.encode_reference / target_forward) is the oracle as well."""
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from snapref_forge.core import Adapter, has_style, read_meta
from snapref_forge.integration import OFF, Session

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import tiny_anima as tiny  # noqa: E402


class AdapterFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory(dir=HERE, ignore_cleanup_errors=True)
        d = Path(cls._dir.name)
        cls.paths = {}
        for kind in ("v2a", "v2s", "v2t"):
            for fp8 in (False, True):
                name = kind + ("_fp8" if fp8 else "")
                cls.paths[name] = tiny.write_adapter(d / f"snapref_{name}.safetensors", kind, fp8)
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def adapter(self, name):
        return Adapter(self.paths[name])

    def rewrite(self, name, change, src="v2s"):
        """Copy of a synthetic file with change(tensors, meta) applied."""
        import json
        from safetensors import safe_open
        tensors = load_file(str(self.paths[src]))
        with safe_open(str(self.paths[src]), "pt") as f:
            meta = json.loads(f.metadata()["anima_ref_kv"])
        change(tensors, meta)
        path = self.dir / name
        save_file(tensors, str(path), metadata={"anima_ref_kv": json.dumps(meta)})
        return path


def run(adapter, mode, flags=(0, 1), fidelity=1.0, in_uncond=False, off=False, per=1, refs=1, weights=None, calls=1):
    """One Forge sampling pass on a fresh tiny model: Session.patch, then `calls` diffusion-model calls with
    cond_or_uncond = flags. Returns (outputs of the last call, the session's reference K/V or None)."""
    batch = per * len(flags)
    x, t, ctx, ref, cap = tiny.inputs(batch)
    session = Session(adapter, [ref * (1 + i) for i in range(refs)], cap[0], mode,
                      weights or [1.0] * refs, fidelity, in_uncond)
    with tiny.forge_backend(), torch.inference_mode():
        m = session.patch(tiny.Patcher(tiny.Anima()))
        model = m.apply()
        options = {**m.options, "cond_or_uncond": list(flags)}
        if off:
            options[OFF] = True
        for _ in range(calls):
            out = model(x, t, ctx, transformer_options=dict(options))
        kv = next((v for k, v in session.cache.items() if k[0] == "kv"), None)
        assert session.branches == () and session.stream is None and not session.reads_ref and session.ref_rows is None
    return out, kv


def kv_equal(a, b):
    return len(a[0]) == len(b[0]) and a[1] == b[1] and all(
        torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a[0], b[0]))


class BranchLoadTests(AdapterFiles):
    def test_branch_keys_parse_with_scale(self):
        ad = self.adapter("v2t")
        self.assertEqual(list(ad.branches), ["style", "style_tgt"])
        self.assertEqual((ad.branches["style"]["stream"], ad.branches["style_tgt"]["stream"]), ("ref", "tgt"))
        self.assertEqual(ad.branches["style"]["modes"], ["style"])
        self.assertEqual(len(ad.branches["style"]["modules"]), tiny.BLOCKS * len(tiny.STYLE_TARGETS))
        self.assertEqual(sorted(ad.branches["style_tgt"]["modules"]),
                         sorted(f"blocks.{b}.{t}" for b in range(tiny.BLOCKS) for t in tiny.TGT_TARGETS))
        down, up, scale = ad.branches["style"]["modules"]["blocks.0.mlp.layer1"]
        self.assertEqual((tuple(down.shape), tuple(up.shape), scale), ((tiny.RANK, tiny.DIM), (2 * tiny.DIM, tiny.RANK),
                                                                       tiny.ALPHA / tiny.RANK))
        self.assertEqual(down.dtype, torch.bfloat16)
        raw = load_file(str(self.paths["v2t"]))
        self.assertTrue(torch.equal(down, raw["anima_ref_kv.branch.style.blocks.0.mlp.layer1.down"]))
        self.assertEqual(self.adapter("v2s").branches_for("style"), ("style",))
        self.assertEqual(ad.branches_for("style"), ("style", "style_tgt"))
        for mode in ("cs", "char"):
            self.assertEqual(ad.branches_for(mode), ())
        self.assertEqual(self.adapter("v2a").branches, {})
        self.assertEqual(ad.indices, {0, 1})

    def test_stream_loras_are_untouched_by_branch_keys(self):
        plain = self.adapter("v2a")
        for name in ("v2s", "v2t"):
            ad = self.adapter(name)
            self.assertEqual(set(ad.lora), set(plain.lora))
            for module, streams in plain.lora.items():
                self.assertEqual(set(ad.lora[module]), set(streams))
                for stream, (down, up, scale) in streams.items():
                    other = ad.lora[module][stream]
                    self.assertTrue(torch.equal(other[0], down) and torch.equal(other[1], up) and other[2] == scale)
            self.assertTrue(torch.equal(ad.mode_embed[:2], plain.mode_embed))

    def test_fp8_branch_uses_row_scales(self):
        ad, raw = self.adapter("v2t_fp8"), load_file(str(self.paths["v2t_fp8"]))
        root = "anima_ref_kv.branch.style_tgt.blocks.1.self_attn.q_proj."
        for i, part in enumerate(("down", "up")):
            self.assertEqual(raw[root + part].dtype, torch.float8_e4m3fn)
            want = (raw[root + part].float() * raw[root + part + "_scale"]).to(torch.bfloat16)
            self.assertTrue(torch.equal(ad.branches["style_tgt"]["modules"]["blocks.1.self_attn.q_proj"][i], want))
        # FP8 stream LoRAs are those of the FP8 v2a-like file, so cs / char stay identical to it (as the release)
        plain = self.adapter("v2a_fp8")
        for module, streams in plain.lora.items():
            for stream, weights in streams.items():
                self.assertTrue(torch.equal(ad.lora[module][stream][0], weights[0]))
        bf16 = self.adapter("v2t").branches["style_tgt"]["modules"]["blocks.1.self_attn.q_proj"][0]
        torch.testing.assert_close(ad.branches["style_tgt"]["modules"]["blocks.1.self_attn.q_proj"][0].float(),
                                   bf16.float(), rtol=0.07, atol=0.01)

    def test_branch_without_metadata_raises(self):
        path = self.rewrite("nometa.safetensors", lambda t, m: m.pop("branches"))
        with self.assertRaisesRegex(ValueError, r"branches\.style"):
            Adapter(path)

    def test_unsupported_branch_spec_raises(self):
        for bad in ({"stream": "both"}, {"combine": "replace"}):
            path = self.rewrite(f"bad_{list(bad)[0]}.safetensors", lambda t, m: m["branches"]["style"].update(bad))
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "not supported"):
                Adapter(path)

    def test_fp8_branch_without_scale_raises(self):
        def drop(tensors, meta):
            key = "anima_ref_kv.branch.style.blocks.0.self_attn.q_proj.down"
            tensors[key] = tensors[key].to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "row scales"):
            Adapter(self.rewrite("noscale.safetensors", drop))

    def test_branches_follow_metadata_order(self):
        def reorder(tensors, meta):
            spec = dict(meta["branches"]["style"])
            meta["branches"] = {"style_b": {**spec, "stream": "tgt"}, "style": spec}
            for key in [k for k in tensors if k.startswith("anima_ref_kv.branch.style.blocks.0.self_attn.q_proj.")]:
                tensors[key.replace("branch.style.", "branch.style_b.")] = tensors[key].clone()
        ad = Adapter(self.rewrite("order.safetensors", reorder))
        self.assertEqual(ad.branches_for("style"), ("style_b", "style"))

    def test_check_mode_message(self):
        plain = self.adapter("v2a")
        with self.assertRaises(ValueError) as caught:
            plain.check_mode("style", "snapref_v2a.safetensors")
        # the wording of the ComfyUI SnapRef node (_Adapter.check_mode)
        self.assertEqual(str(caught.exception),
                         "SnapRef: snapref_v2a.safetensors has no 'style' mode (its modes: cs, char). Style mode needs "
                         "an adapter with a style branch, e.g. snapref_v2s_pre; use mode cs or char with this adapter.")
        with self.assertRaisesRegex(ValueError, r"^SnapRef: snapref_v2a\.safetensors has no 'style' mode"):
            plain.check_mode("style")   # the file name by default
        for mode in ("cs", "char"):
            plain.check_mode(mode, "snapref_v2a.safetensors")   # never an error
            self.adapter("v2s").check_mode(mode)
        self.adapter("v2s").check_mode("style")
        self.adapter("v2s_fp8").check_mode("style")
        _, _, _, ref, cap = tiny.inputs()
        with self.assertRaisesRegex(ValueError, "has no 'style' mode"):
            Session(plain, [ref], cap[0], "style", [1.0])

    def test_check_mode_needs_branch_weights_for_a_listed_style_mode(self):
        def strip(tensors, meta):
            for key in [k for k in tensors if k.startswith("anima_ref_kv.branch.")]:
                del tensors[key]
        ad = Adapter(self.rewrite("listed.safetensors", strip))
        self.assertEqual(ad.branches, {})
        with self.assertRaisesRegex(ValueError, "lists the 'style' mode but has no mode_embed row or branch weights"):
            ad.check_mode("style")

    def test_header_metadata_tells_style_support(self):
        self.assertTrue(has_style(read_meta(self.paths["v2s"])))
        self.assertTrue(has_style(read_meta(self.paths["v2t_fp8"])))
        self.assertFalse(has_style(read_meta(self.paths["v2a"])))
        self.assertFalse(has_style({}))   # no modes: a cs-only adapter
        cut = self.dir / "cut.safetensors"
        cut.write_bytes(Path(self.paths["v2s"]).read_bytes()[:20])
        with self.assertRaises(ValueError):
            read_meta(cut)
        (self.dir / "empty.safetensors").write_bytes(b"")
        with self.assertRaises(ValueError):
            read_meta(self.dir / "empty.safetensors")


CONFIGS = [dict(flags=f, fidelity=fid, in_uncond=iu, off=o, per=p)
           for f in ((0, 1), (1, 0), (0,), (1,)) for fid in (1.0, 0.5) for iu in (False, True) for o in (False, True)
           for p in (1, 2) if not (len(f) == 1 and p == 2)]


class CharacterModesUnchangedTests(AdapterFiles):
    def test_cs_and_char_are_bit_identical_to_a_file_without_branches(self):
        for plain, branched in (("v2a", "v2s"), ("v2a", "v2t"), ("v2a_fp8", "v2s_fp8"), ("v2a_fp8", "v2t_fp8")):
            a, b = self.adapter(plain), self.adapter(branched)
            for mode in ("cs", "char"):
                for config in CONFIGS:
                    with self.subTest(plain=plain, branched=branched, mode=mode, **config):
                        out_a, kv_a = run(a, mode, **config)
                        out_b, kv_b = run(b, mode, **config)
                        self.assertTrue(torch.equal(out_a, out_b))
                        self.assertEqual(kv_a is None, kv_b is None)
                        if kv_a is not None:
                            self.assertTrue(kv_equal(kv_a, kv_b))

    def test_cs_and_char_patch_only_the_stream_lora_modules(self):
        _, _, _, ref, cap = tiny.inputs()
        for mode in ("cs", "char"):
            patched = {}
            for name in ("v2a", "v2t"):
                m = Session(self.adapter(name), [ref], cap[0], mode, [1.0]).patch(tiny.Patcher(tiny.Anima()))
                patched[name] = (set(m.patches), m.preserved)
            self.assertEqual(patched["v2a"], patched["v2t"])

    def test_cs_after_style_on_the_same_adapter_runs_no_branch(self):
        ad, plain = self.adapter("v2t"), self.adapter("v2a")
        for config in (dict(), dict(in_uncond=True), dict(fidelity=0.5), dict(flags=(0,))):
            with self.subTest(**config):
                styled, _ = run(ad, "style", **config)
                for mode in ("cs", "char"):
                    out, kv = run(ad, mode, **config)
                    want, want_kv = run(plain, mode, **config)
                    self.assertTrue(torch.equal(out, want))
                    self.assertTrue(kv_equal(kv, want_kv))
                self.assertFalse(torch.equal(styled, want))

    def test_multiple_references_and_strengths(self):
        for mode in ("cs", "char"):
            config = dict(refs=3, weights=[1.0, 0.4, 2.0])
            a, _ = run(self.adapter("v2a"), mode, **config)
            b, _ = run(self.adapter("v2t"), mode, **config)
            self.assertTrue(torch.equal(a, b))


class Oracle:
    """train/style_branch.py semantics written out, independently of snapref_forge: every targeted Linear of a fresh
    tiny model computes DualLoRALinear._forward with branches (base, then the stream LoRA, then each active branch of
    that stream, a target branch only while the forward reads a reference); the reference is encoded as
    asym.encode_reference (sigma 0, mode_embed row on the time embedding, self-attention K/V recorded) and a target
    row reads it as asym.target_forward (reference keys first), with a per-reference log weight on its keys."""

    def __init__(self, path, mode):
        tensors, meta = load_file(str(path)), read_meta(path)

        def weight(key):
            w = tensors[key]
            return (w.float() * tensors[key + "_scale"] if key + "_scale" in tensors else w).to(torch.bfloat16)

        self.dm = tiny.Anima()
        self.state = {"stream": "tgt", "active": None, "reads": False}
        self.active = tuple(n for n, s in (meta.get("branches") or {}).items() if mode in s["modes"]) or None
        self.row = tensors["anima_ref_kv.mode_embed"][meta["modes"].index(mode)]
        layers = {}
        for key in tensors:
            if key.endswith(".lora_down"):
                _, stream, module = key[:-len(".lora_down")].split(".", 2)
                root = key[:-len("lora_down")]
                layers.setdefault(module, [{}, {}])[0][stream] = (weight(root + "lora_down"), weight(root + "lora_up"),
                                                                  float(tensors[root + "alpha"]) / tiny.RANK)
            elif key.startswith("anima_ref_kv.branch.") and key.endswith(".down"):
                root = key[:-len("down")]
                branch, module = root[len("anima_ref_kv.branch."):-1].split(".", 1)
                stream = meta["branches"][branch]["stream"]
                layers.setdefault(module, [{}, {}])[1][branch] = (weight(root + "down"), weight(root + "up"),
                                                                  float(tensors[root + "alpha"]) / tiny.RANK, stream)
        for module, (adapters, branches) in layers.items():
            parent, attr = module.rsplit(".", 1)
            base = self.dm.get_submodule(module)
            self.dm.get_submodule(parent).__setattr__(attr, _OracleLinear(base, adapters, branches, self))
        for index, block in enumerate(self.dm.blocks):
            block.self_attn.forward = self._attention(block.self_attn, index)
        self.record = self.cache = None

    def _attention(self, attn, index):
        def forward(x, context=None, rope_emb=None, transformer_options=None):
            q, k, v = attn.compute_qkv(x, context, rope_emb=rope_emb)
            if self.record is not None:
                self.record[index] = (k, v)
                return attn.compute_attention(q, k, v)
            if self.cache is None:
                return attn.compute_attention(q, k, v)
            (k_r, v_r), bias = self.cache[index], self.key_bias
            keys, values = torch.cat([k_r, k], 1).transpose(1, 2), torch.cat([v_r, v], 1).transpose(1, 2)
            mask = torch.cat([bias, torch.zeros(k.shape[1])]).to(q.dtype).view(1, 1, 1, -1)
            out = F.scaled_dot_product_attention(q.transpose(1, 2), keys, values, attn_mask=mask).transpose(1, 2)
            return attn.output_dropout(attn.output_proj(out.reshape(q.shape[0], q.shape[1], -1)))
        return forward

    @torch.no_grad()
    def encode(self, ref, cap):
        hidden, _, _ = self.dm.prepare_embedded_sequence(ref)
        emb, adaln = self.dm.t_embedder[1](self.dm.t_embedder[0](torch.zeros(1, 1)))
        emb = self.dm.t_embedding_norm(emb) + self.row[None, None, :]
        self.record = [None] * len(self.dm.blocks)
        self.state.update(stream="ref", active=self.active, reads=False)
        try:
            for block in self.dm.blocks:
                hidden = block(hidden, emb, cap, adaln_lora_B_T_3D=adaln)
            return self.record
        finally:
            self.record = None
            self.state.update(stream="tgt", active=None, reads=False)

    @torch.no_grad()
    def target(self, x, t, ctx, cache=None, biases=(), sizes=()):
        """One target forward; cache = [(k, v)] per block (reads the reference) or None."""
        self.cache = cache
        self.key_bias = torch.cat([torch.full((s,), float(b)) for s, b in zip(sizes, biases)]) if cache else None
        self.state.update(stream="tgt", active=self.active, reads=cache is not None)
        try:
            hidden, _, _ = self.dm.prepare_embedded_sequence(x)
            emb, adaln = self.dm.t_embedder[1](self.dm.t_embedder[0](t.reshape(x.shape[0], 1)))
            emb = self.dm.t_embedding_norm(emb)
            for block in self.dm.blocks:
                hidden = block(hidden, emb, ctx, adaln_lora_B_T_3D=adaln)
            return self.dm.unpatchify(self.dm.final_layer(hidden, emb))[:, :, :1]
        finally:
            self.cache = None
            self.state.update(active=None, reads=False)

    def sample(self, x, t, ctx, refs, cap, reads, biases, row_biases):
        """Every row on its own: a reading row with the reference (and its row bias), the others without."""
        records = [self.encode(r, cap) for r in refs]
        sizes = [rec[0][0].shape[1] for rec in records]
        cache = [(torch.cat([rec[i][0] for rec in records], 1), torch.cat([rec[i][1] for rec in records], 1))
                 for i in range(len(self.dm.blocks))]
        out = []
        for r in range(x.shape[0]):
            sl = slice(r, r + 1)
            if r in reads:
                out.append(self.target(x[sl], t[sl], ctx[sl], cache, [b + row_biases[r] for b in biases], sizes))
            else:
                out.append(self.target(x[sl], t[sl], ctx[sl]))
        return torch.cat(out), (cache, sizes)


class _OracleLinear(torch.nn.Module):
    def __init__(self, base, adapters, branches, oracle):
        super().__init__()
        self.base, self.adapters, self.branches, self._oracle = base, adapters, branches, [oracle]

    def forward(self, x):   # style_branch._forward
        state = self._oracle[0].state
        out = self.base(x)
        stream = state["stream"]
        if stream in self.adapters:
            down, up, scale = self.adapters[stream]
            out = out + F.linear(F.linear(x, down.to(x.dtype)), up.to(x.dtype)) * scale
        if state["active"] is None:
            return out
        for name in state["active"]:
            if name in self.branches:
                down, up, scale, bstream = self.branches[name]
                if stream == bstream and (bstream != "tgt" or state["reads"]):
                    out = out + F.linear(F.linear(x, down.to(x.dtype)), up.to(x.dtype)) * scale
        return out


def reading_rows(flags, per, fidelity, in_uncond, off):
    """Rows that read the reference in one Forge call, and each row's extra log bias (Fidelity on uncond rows)."""
    if off:
        return [], {}
    rows = [i * per + j for i, f in enumerate(flags) for j in range(per) if f == 0 or in_uncond or fidelity < 1]
    bias = {r: (math.log(max(1 - fidelity, 1e-6)) if flags[r // per] == 1 and not in_uncond else 0.0) for r in rows}
    return rows, bias


class StyleModeTests(AdapterFiles):
    def test_reference_encoding_runs_the_ref_branch_only_in_style(self):
        for name in ("v2s", "v2t", "v2s_fp8"):
            ad = self.adapter(name)
            _, cs = run(ad, "cs")
            _, style = run(ad, "style")
            self.assertFalse(any(torch.equal(a[0], b[0]) for a, b in zip(cs[0], style[0])))
            x, t, ctx, ref, cap = tiny.inputs()
            want = Oracle(self.paths[name], "style").encode(ref, cap)
            for (k, v), (k_o, v_o) in zip(style[0], want):
                self.assertTrue(torch.equal(k, k_o) and torch.equal(v, v_o), name)
            want = Oracle(self.paths[name], "cs").encode(ref, cap)
            for (k, v), (k_o, v_o) in zip(cs[0], want):
                self.assertTrue(torch.equal(k, k_o) and torch.equal(v, v_o), name)

    def test_ref_branch_is_off_in_target_forwards(self):
        # v2s has only the reference-stream branch: with the reference K/V fixed, Style's target forward is the
        # plain tgt-LoRA forward. A reference-free call is the same as in cs.
        ad = self.adapter("v2s")
        self.assertTrue(torch.equal(run(ad, "style", off=True)[0], run(ad, "cs", off=True)[0]))
        self.assertTrue(torch.equal(run(ad, "style", flags=(1,))[0], run(ad, "cs", flags=(1,))[0]))

    def test_style_matches_the_training_semantics(self):
        for name in ("v2s", "v2t"):
            for config in CONFIGS:
                if config["per"] != 1:
                    continue
                with self.subTest(name=name, **config):
                    out, kv = run(self.adapter(name), "style", **config)
                    x, t, ctx, ref, cap = tiny.inputs(len(config["flags"]))
                    rows, bias = reading_rows(config["flags"], 1, config["fidelity"], config["in_uncond"], config["off"])
                    want, (cache, sizes) = Oracle(self.paths[name], "style").sample(x, t, ctx, [ref], cap, rows, [0.0], bias)
                    torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-6)
                    if kv is not None:
                        self.assertTrue(kv_equal(kv, (cache, sizes)))

    def test_style_with_several_weighted_references(self):
        weights = [1.0, 0.35, 2.0]
        out, _ = run(self.adapter("v2t"), "style", refs=3, weights=weights)
        x, t, ctx, ref, cap = tiny.inputs()
        refs = [ref * (1 + i) for i in range(3)]
        want, _ = Oracle(self.paths["v2t"], "style").sample(x, t, ctx, refs, cap, [0], [math.log(w) for w in weights],
                                                            {0: 0.0})
        torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-6)

    def test_target_branch_runs_only_in_rows_that_read_the_reference(self):
        # v2t = v2s + the target branch. Rows that do not read the reference are bit-identical to v2s; rows that
        # read it differ (the target branch is on there).
        with_tgt, without = self.adapter("v2t"), self.adapter("v2s")
        cases = [((0, 1), 1, 1.0, False, False, [0]), ((1, 0), 1, 1.0, False, False, [1]),
                 ((1, 0), 2, 1.0, False, False, [2, 3]), ((0, 1), 1, 1.0, True, False, [0, 1]),
                 ((0, 1), 1, 0.5, False, False, [0, 1]), ((0, 1), 2, 0.0, False, False, [0, 1, 2, 3]),
                 ((1,), 1, 1.0, False, False, []), ((1,), 1, 0.5, False, False, [0]), ((0,), 1, 1.0, False, False, [0]),
                 ((0, 1), 1, 1.0, False, True, []), ((0, 1), 1, 0.5, True, True, []), ((0,), 1, 1.0, False, True, [])]
        for flags, per, fidelity, in_uncond, off, rows in cases:
            config = dict(flags=flags, per=per, fidelity=fidelity, in_uncond=in_uncond, off=off)
            with self.subTest(**config):
                self.assertEqual(reading_rows(flags, per, fidelity, in_uncond, off)[0], rows)
                a, _ = run(with_tgt, "style", **config)
                b, _ = run(without, "style", **config)
                for r in range(a.shape[0]):
                    if r in rows:
                        self.assertFalse(torch.equal(a[r], b[r]), r)
                    else:
                        self.assertTrue(torch.equal(a[r], b[r]), r)

    def test_target_branch_on_a_module_without_a_tgt_lora(self):
        ad = self.adapter("v2t")
        self.assertNotIn("tgt", ad.lora["blocks.0.mlp.layer1"])
        _, _, _, ref, cap = tiny.inputs()
        m = Session(ad, [ref], cap[0], "style", [1.0]).patch(tiny.Patcher(tiny.Anima()))
        self.assertIn("diffusion_model.blocks.0.mlp.layer1.forward", m.patches)
        plain = Session(self.adapter("v2a"), [ref], cap[0], "cs", [1.0]).patch(tiny.Patcher(tiny.Anima()))
        self.assertEqual(set(m.patches), set(plain.patches))   # every style module also has a stream LoRA here
        branch_bytes = sum((a.numel() + b.numel()) * 4 for br in ad.branches.values() for a, b, _ in br["modules"].values())
        self.assertEqual(m.preserved - plain.preserved, branch_bytes)

    def test_branch_on_a_module_without_any_stream_lora(self):
        def add(tensors, meta):   # v2t + its target branch on cross_attn.k_proj, which has no LoRA at all
            meta["branches"]["style_tgt"]["targets"].append("cross_attn.k_proj")
            gen = torch.Generator().manual_seed(5)
            for b in range(tiny.BLOCKS):
                root = f"anima_ref_kv.branch.style_tgt.blocks.{b}.cross_attn.k_proj."
                tensors[root + "down"] = (torch.randn(tiny.RANK, tiny.DIM, generator=gen) * 0.5).to(torch.bfloat16)
                tensors[root + "up"] = (torch.randn(tiny.DIM, tiny.RANK, generator=gen) * 0.5).to(torch.bfloat16)
                tensors[root + "alpha"] = torch.tensor(tiny.ALPHA)
        path = self.rewrite("bare.safetensors", add, src="v2t")
        ad = Adapter(path)
        self.assertNotIn("blocks.0.cross_attn.k_proj", ad.lora)
        _, _, _, ref, cap = tiny.inputs()
        attr = "diffusion_model.blocks.0.cross_attn.k_proj.forward"
        self.assertIn(attr, Session(ad, [ref], cap[0], "style", [1.0]).patch(tiny.Patcher(tiny.Anima())).patches)
        self.assertNotIn(attr, Session(ad, [ref], cap[0], "cs", [1.0]).patch(tiny.Patcher(tiny.Anima())).patches)
        for config in (dict(), dict(fidelity=0.5), dict(off=True)):
            with self.subTest(**config):
                out, _ = run(ad, "style", **config)
                x, t, ctx, ref, cap = tiny.inputs()
                rows, bias = reading_rows((0, 1), 1, config.get("fidelity", 1.0), False, config.get("off", False))
                want, _ = Oracle(path, "style").sample(x, t, ctx, [ref], cap, rows, [0.0], bias)
                torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.equal(run(ad, "cs", **config)[0], run(self.adapter("v2a"), "cs", **config)[0]))

    def test_repeated_calls_reuse_the_reference_and_reset_state(self):
        ad = self.adapter("v2t")
        once, kv1 = run(ad, "style")
        twice, kv2 = run(ad, "style", calls=3)
        self.assertTrue(torch.equal(once, twice))
        self.assertTrue(kv_equal(kv1, kv2))

    def test_failure_inside_a_call_clears_the_branch_state(self):
        ad = self.adapter("v2t")
        x, t, ctx, ref, cap = tiny.inputs()
        session = Session(ad, [ref], cap[0], "style", [1.0])
        with tiny.forge_backend(), torch.inference_mode():
            m = session.patch(tiny.Patcher(tiny.Anima()))
            model = m.apply()
            with self.assertRaises(RuntimeError):
                model(x, t, ctx[:, :, :3], transformer_options={**m.options, "cond_or_uncond": [0, 1]})
        self.assertEqual((session.branches, session.reads_ref, session.ref_rows, session.stream), ((), False, None, None))
        self.assertEqual(session.cache, {})


TRAIN = os.environ.get("SNAPREF_TRAIN_DIR")


@unittest.skipUnless(TRAIN and (Path(TRAIN) / "style_branch.py").is_file(),
                     "set SNAPREF_TRAIN_DIR to the SnapRef research train/ folder (style_branch.py, asym.py)")
class TrainingCodeTests(AdapterFiles):
    """The training code itself as the oracle: style_branch.load_adapter puts DualLoRALinear modules with branch
    LoRAs on a tiny model; asym.encode_reference / target_forward run with style_branch's forward."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        sys.path.insert(0, str(Path(TRAIN).resolve()))
        import asym
        import style_branch
        cls.asym, cls.sb = asym, style_branch

    def train(self, name, mode):
        dm = tiny.Anima()
        modules, mode_embed, modes, meta = self.sb.load_adapter(dm, Path(self.paths[name]), "cpu")
        self.sb.BRANCH.active = self.sb.branches_for(meta, mode)
        self.addCleanup(setattr, self.sb.BRANCH, "active", None)
        return dm, mode_embed[modes.index(mode)][None]

    def test_reference_encoding_matches_training(self):
        x, t, ctx, ref, cap = tiny.inputs()
        for name in ("v2s", "v2t"):
            for mode in ("cs", "char", "style"):
                with self.subTest(name=name, mode=mode):
                    dm, row = self.train(name, mode)
                    with torch.no_grad():
                        want = self.asym.encode_reference(dm, ref, cap, tuple(x.shape[-2:]), mode_emb=row)
                    _, kv = run(self.adapter(name), mode)
                    self.assertTrue(all(torch.equal(k, w[0]) and torch.equal(v, w[1]) for (k, v), w in zip(kv[0], want)))

    def test_target_forwards_match_training(self):
        for name in ("v2s", "v2t"):
            for mode in ("cs", "style"):
                with self.subTest(name=name, mode=mode):
                    x, t, ctx, ref, cap = tiny.inputs(1)
                    dm, row = self.train(name, mode)
                    with torch.no_grad():
                        record = self.asym.encode_reference(dm, ref, cap, tuple(x.shape[-2:]), mode_emb=row)
                        reading = self.asym.target_forward(dm, x, t, ctx, record)   # reads the reference
                        free = self.asym.target_forward(dm, x, t, ctx, None)        # reference-free
                    out, _ = run(self.adapter(name), mode, flags=(0,))
                    torch.testing.assert_close(out, reading, rtol=1e-5, atol=1e-6)
                    self.assertTrue(torch.equal(run(self.adapter(name), mode, flags=(0,), off=True)[0], free))
                    self.assertTrue(torch.equal(run(self.adapter(name), mode, flags=(1,))[0], free))


if __name__ == "__main__":
    unittest.main()
