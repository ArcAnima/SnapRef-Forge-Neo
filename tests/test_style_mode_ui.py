"""Style mode in the Forge panel: the Mode choices and their mapping to the adapter modes, the prompt hint per mode,
the warning for adapters without Style, the v2s download entries with their checksums, the default adapter, and
the generate-time paths (a clear error before any encoding when the adapter has no Style). Gradio and Forge are
stubbed as in test_script_args; the real scripts/snapref.py, Adapter and Session run."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import tiny_anima as tiny  # noqa: E402
from test_script_args import Component, load_script  # noqa: E402

OLD_RELEASES = {
    'snapref_v2a.safetensors': '0ppxnhximxr/snapref-v2a',
    'snapref_v2a_fp8.safetensors': '0ppxnhximxr/snapref-v2a',
    'snapref_v1a.safetensors': '0ppxnhximxr/snapref-v1a',
    'snapref_v1a_fp8.safetensors': '0ppxnhximxr/snapref-v1a',
    'snapref_v1b.safetensors': '0ppxnhximxr/snapref-v1b',
    'snapref_v1b_fp8.safetensors': '0ppxnhximxr/snapref-v1b',
}
V2S, V2S_FP8 = 'snapref_v2s_pre1.safetensors', 'snapref_v2s_pre1_fp8.safetensors'


class StyleModeUiTests(unittest.TestCase):
    def setUp(self):
        Component.created = []
        self.mod = load_script()
        self.script = self.mod.SnapRefScript()
        directory = tempfile.TemporaryDirectory(dir=HERE, ignore_cleanup_errors=True)
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def files(self):
        return {'snapref_v2a.safetensors': tiny.write_adapter(self.dir / 'snapref_v2a.safetensors', 'v2a'),
                'snapref_v2s.safetensors': tiny.write_adapter(self.dir / 'snapref_v2s.safetensors', 'v2s')}

    def build(self, found=None):
        with patch.object(self.mod, 'model_files', return_value=found or {}):
            values = self.script.ui(False)
        return values, {type(c).__name__ + ':' + str(c.kwargs.get('label')): c for c in Component.created}

    def test_mode_offers_style_with_the_node_mapping(self):
        values, by_label = self.build()
        mode = by_label['Radio:Mode']
        self.assertIs(values[2], mode)   # argument order unchanged
        self.assertEqual(mode.args[0], ['Character & Style', 'Character', 'Style'])
        self.assertEqual(mode.kwargs['value'], 'Character & Style')
        self.assertEqual(self.mod.MODES, {'Character & Style': 'cs', 'Character': 'char', 'Style': 'style'})
        hint = next(c for c in Component.created if c.kwargs.get('elem_classes') == ['snapref-hint'])
        self.assertEqual(hint.args[0], self.mod.MODE_HINTS['Character & Style'])
        adapter = by_label['Dropdown:SnapRef adapter']
        for component in (mode, adapter):
            self.assertEqual([(e[0], e[1]) for e in component.events], [('change', self.mod.update_hint)])
            self.assertEqual(component.events[0][2]['inputs'], [mode, adapter])
            self.assertEqual(component.events[0][2]['outputs'], [hint])
        self.assertNotIn(hint, values)   # the hint is not a script argument
        names = [name for _, name in self.script.infotext_fields if isinstance(name, str)]
        self.assertEqual(names[:12], ['SnapRef ' + n for n in ('Adapter', 'Mode', 'Layout', 'Fidelity', 'Reference MP',
                                                              'Reference caption', 'Reference in negative',
                                                              'Extra guidance', 'Guidance', 'Guidance end',
                                                              'Correction', 'Rescale phi')])

    def test_prompt_hint_per_mode(self):
        hint = self.mod.mode_hint
        self.assertIn('write only the scene', hint('Character & Style'))
        self.assertIn('add artist tags for the style', hint('Character'))
        self.assertIn('write the character and the scene', hint('Style'))
        self.assertIn('Reference caption empty', hint('Style'))
        self.assertEqual(hint('Style', False), self.mod.STYLE_MISSING)
        self.assertIn('has no Style mode', self.mod.STYLE_MISSING)
        self.assertIn('snapref_v2s_pre1', self.mod.STYLE_MISSING)
        self.assertEqual(hint('Style', None), hint('Style'))     # unknown: no warning, the run reports it
        self.assertEqual(hint('Style', True), hint('Style'))
        self.assertEqual(hint('Character', False), hint('Character'))   # warning only while Style is selected
        with patch.object(self.mod, 'model_files', return_value={}):
            self.assertEqual(self.mod.update_hint('Style', 'snapref_v2a_fp8.safetensors'), {'value': self.mod.STYLE_MISSING})
            self.assertEqual(self.mod.update_hint('Style', V2S_FP8), {'value': hint('Style')})
            self.assertEqual(self.mod.update_hint('Character', 'snapref_v2a_fp8.safetensors'), {'value': hint('Character')})

    def test_style_support_from_header_or_release_list(self):
        found = self.files()
        found[V2S] = tiny.write_adapter(self.dir / 'renamed.safetensors', 'v2a')   # the header decides, not the name
        cut = self.dir / 'snapref_cut.safetensors'
        cut.write_bytes(Path(found['snapref_v2s.safetensors']).read_bytes()[:30])
        found['snapref_cut.safetensors'] = cut
        support = lambda name: self.mod.style_support(name, found)   # noqa: E731
        self.assertTrue(support('snapref_v2s.safetensors'))
        self.assertFalse(support('snapref_v2a.safetensors'))
        self.assertFalse(support(V2S))
        self.assertIsNone(support('snapref_cut.safetensors'))
        self.assertTrue(support(V2S_FP8))                                  # released, not installed
        self.assertFalse(support('snapref_v2a_fp8.safetensors'))
        self.assertIsNone(support('snapref_unknown.safetensors'))

    def test_release_list_checksums_and_default(self):
        self.assertEqual(list(self.mod.RELEASES)[:2], [V2S, V2S_FP8])
        self.assertEqual({k: self.mod.RELEASES[k] for k in (V2S, V2S_FP8)},
                         {V2S: '0ppxnhximxr/snapref-v2s', V2S_FP8: '0ppxnhximxr/snapref-v2s'})
        self.assertEqual({k: v for k, v in self.mod.RELEASES.items() if k not in (V2S, V2S_FP8)}, OLD_RELEASES)
        self.assertEqual(self.mod.CHECKSUMS, {
            V2S: (543395968, '95f91797827cbf5efd86e29fac5b2646df1505d46acb20f913877b3c9a429609'),
            V2S_FP8: (278127904, '55a429f078d74f3e144a89d14aaaaedbd9f51c2ac598728f8f0135a6e67a1600')})
        default = self.mod.default_adapter
        self.assertEqual(default({}), V2S_FP8)
        self.assertEqual(default({'snapref_mine.safetensors': 'p'}), V2S_FP8)
        self.assertEqual(default({'snapref_v2a_fp8.safetensors': 'p'}), 'snapref_v2a_fp8.safetensors')
        self.assertEqual(default({'snapref_v2a.safetensors': 'p', V2S: 'p'}), V2S)
        found = {'snapref_v2a_fp8.safetensors': self.dir / 'a'}
        _, by_label = self.build(found)
        dropdown = by_label['Dropdown:SnapRef adapter']
        self.assertEqual(dropdown.kwargs['value'], 'snapref_v2a_fp8.safetensors')
        self.assertEqual(dropdown.args[0], list(dict.fromkeys([*found, *self.mod.RELEASES])))

    def fake_hub(self, payload=b'abc', fail=None):
        hub, calls = ModuleType('huggingface_hub'), []

        def hf_hub_download(repo, name, local_dir):
            calls.append((repo, name))
            if fail:
                raise fail
            path = Path(local_dir) / name
            path.write_bytes(payload)
            return str(path)
        hub.hf_hub_download = hf_hub_download
        return patch.dict(sys.modules, {'huggingface_hub': hub}), calls

    def test_download_checks_size_and_sha256(self):
        good = (3, hashlib.sha256(b'abc').hexdigest())
        models = self.dir / 'models'
        with patch.object(self.mod, 'ROOT', self.dir), patch.object(self.mod, 'model_files', return_value={}):
            for checksum, ok in ((good, True), ((3, '0' * 64), False), ((4, good[1]), False)):
                hub, calls = self.fake_hub()
                with self.subTest(checksum=checksum), hub, patch.dict(self.mod.CHECKSUMS, {V2S: checksum}):
                    models.mkdir(exist_ok=True)
                    if ok:
                        self.assertEqual(self.mod.download(V2S)['value'], V2S)
                    else:
                        with self.assertRaisesRegex(self.mod.gr.Error, 'SHA-256 mismatch'):
                            self.mod.download(V2S)
                    self.assertEqual(calls, [('0ppxnhximxr/snapref-v2s', V2S)])
                    self.assertEqual((models / V2S).exists(), ok)
            hub, _ = self.fake_hub()
            with hub:   # files without a published checksum are kept as downloaded (as before)
                self.assertEqual(self.mod.download('snapref_v2a.safetensors')['value'], 'snapref_v2a.safetensors')
                self.assertTrue((models / 'snapref_v2a.safetensors').exists())
            hub, _ = self.fake_hub(fail=OSError('401 Client Error'))
            with hub, self.assertRaises(self.mod.gr.Error) as caught:
                self.mod.download(V2S_FP8)
            self.assertIn('https://huggingface.co/0ppxnhximxr/snapref-v2s', str(caught.exception))
            self.assertIn('not public yet', str(caught.exception))
            with self.assertRaises(self.mod.gr.Error):
                self.mod.download('snapref_custom.safetensors')

    def sample(self, adapter, mode, found, encoded=None):
        encoded = [] if encoded is None else encoded
        unet = tiny.Patcher(tiny.Anima())
        p = SimpleNamespace(sd_model=SimpleNamespace(
            forge_objects=SimpleNamespace(unet=unet, vae=None),
            get_learned_conditioning=lambda cond: [torch.zeros(2, tiny.DIM)]), extra_generation_params={})
        args = [True, adapter, mode, 'Separate references', 1.0, 1.0, Image.new('RGB', (16, 16), 'red'), 1.0,
                None, 1.0, None, 1.0, None, 1.0, '', False, False, 4.5, 1.0, 'apg', 0.5]
        with patch.object(self.mod, 'model_files', return_value=found), \
                patch.object(self.mod, 'SdConditioning', lambda texts, is_negative_prompt=False: texts), \
                patch.object(self.mod, 'encode_image', lambda image, vae, mp: encoded.append(mp) or torch.zeros(1, 4, 1, 2, 2)):
            self.script.process_before_every_sampling(p, *args)
        return p, unet, encoded

    def test_style_without_a_style_adapter_fails_before_encoding(self):
        found = self.files()
        encoded = []
        with self.assertRaises(self.mod.gr.Error) as caught:
            self.sample('snapref_v2a.safetensors', 'Style', found, encoded)
        self.assertEqual(str(caught.exception),
                         "SnapRef: snapref_v2a.safetensors has no 'style' mode (its modes: cs, char). Style mode needs "
                         "an adapter with a style branch, e.g. snapref_v2s_pre; use mode cs or char with this adapter.")
        self.assertEqual(encoded, [])   # no picture was encoded
        self.assertIsNone(self.script.session)
        with self.assertRaisesRegex(self.mod.gr.Error, '^SnapRef: Invalid SnapRef mode$'):
            self.sample('snapref_v2s.safetensors', 'Painting', found)

    def test_each_mode_reaches_the_session(self):
        found = self.files()
        for label, key, branches in (('Style', 'style', ('style',)), ('Character', 'char', ()),
                                     ('Character & Style', 'cs', ())):
            with self.subTest(mode=label):
                p, unet, encoded = self.sample('snapref_v2s.safetensors', label, found)
                session = self.script.session
                self.assertEqual((session.mode, session.mode_branches), (key, branches))
                self.assertEqual(encoded, [1.0])
                self.assertIs(p.sd_model.forge_objects.unet, self.script.patched_unet)
                self.assertEqual(p.extra_generation_params['SnapRef Mode'], label)
                self.assertEqual(p.extra_generation_params['SnapRef Adapter'], 'snapref_v2s.safetensors')
                self.script.release(p)
                self.assertIs(p.sd_model.forge_objects.unet, unet)
        p, _, _ = self.sample('snapref_v2a.safetensors', 'Character', found)   # v2a keeps working
        self.assertEqual(self.script.session.mode, 'char')
        self.script.release(p)


if __name__ == '__main__':
    unittest.main()
