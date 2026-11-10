import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from snapref_forge.core import Adapter, attention, block_map, select_references
from snapref_forge.integration import cleanup_after


class CoreTests(unittest.TestCase):
    def test_sampler_failure_and_success_release_request_state(self):
        from types import SimpleNamespace
        for fail in [False, True]:
            released = []
            p = SimpleNamespace(_snapref_cleanup=lambda p:released.append(p))
            def run(p):
                if fail: raise RuntimeError('sampler failed after forward')
                return 7
            if fail:
                with self.assertRaises(RuntimeError): cleanup_after(run)(p)
            else:
                self.assertEqual(cleanup_after(run)(p),7)
            self.assertEqual(released,[p])
            self.assertFalse(hasattr(p,'_snapref_cleanup'))

    def test_reference_weights_and_fidelity_are_attention_weights(self):
        # Zero logits: target value 2, references 4 and 8 with weights 1 and 3.
        q = k = torch.zeros(2, 1, 1, 2)
        v = torch.full_like(q, 2)
        refs = (torch.zeros(1, 2, 1, 2), torch.tensor([[[[4., 4.]], [[8., 8.]]]]))
        out = attention(q, k, v, refs, [1, 1], [0., math.log(3)], [0], [0.])
        torch.testing.assert_close(out[:, 0], torch.tensor([[6., 6.], [2., 2.]]))
        out = attention(q, k, v, refs, [1, 1], [0., math.log(3)], [0, 1], [0., math.log(.5)])
        torch.testing.assert_close(out[:, 0], torch.tensor([[6., 6.], [16/3, 16/3]]))

    def test_fp8_uses_row_scales_and_preserves_two_streams(self):
        import json
        tensors = {'anima_ref_kv.mode_embed': torch.zeros(2, 4)}
        for stream in ['ref', 'tgt']:
            root = f'anima_ref_kv.{stream}.blocks.0.self_attn.q_proj.'
            tensors[root+'lora_down'] = torch.ones(2, 4).to(torch.float8_e4m3fn)
            tensors[root+'lora_down_scale'] = torch.tensor([[2.], [3.]])
            tensors[root+'lora_up'] = torch.ones(4, 2).to(torch.bfloat16)
            tensors[root+'alpha'] = torch.tensor(2.)
        # Released v2a has ref-only cross-attention/MLP modules as well as shared modules.
        tensors.update({k.replace('self_attn.q_proj','cross_attn.q_proj'):v.clone()
                        for k,v in list(tensors.items()) if k.startswith('anima_ref_kv.ref.')})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p = Path(d)/'adapter.safetensors'
            save_file(tensors, str(p), metadata={'snapref':json.dumps({'format':'anima_ref_kv/1','modes':['cs','char']})})
            ad = Adapter(p)
        self.assertEqual(set(ad.lora['blocks.0.self_attn.q_proj']), {'ref', 'tgt'})
        torch.testing.assert_close(ad.lora['blocks.0.self_attn.q_proj']['ref'][0], torch.tensor([[2.]*4, [3.]*4], dtype=torch.bfloat16))

    def test_layout_and_reference_validation(self):
        from PIL import Image
        refs, weights = select_references([Image.new('RGB',(16,32)), Image.new('RGB',(32,32))], [1., .4], 'Separate references')
        self.assertEqual(len(refs), 2)
        self.assertEqual(weights, [1., .4])
        refs, weights = select_references(refs, weights, 'Side by side')
        self.assertEqual(refs[0].size, (48,32))
        self.assertEqual(weights, [1.])
        with self.assertRaises(ValueError):
            select_references([refs[0]]*5,[1.]*5,'Separate references')
        with self.assertRaises(ValueError):
            select_references([refs[0]],[float('nan')],'Separate references')

    def test_api_reference_base64_decodes_without_reading_local_paths(self):
        import base64
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new('RGB',(16,16),'red').save(buffer,format='PNG')
        encoded = 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        refs,_ = select_references([encoded],[1.],'Separate references')
        self.assertEqual(refs[0].getpixel((0,0)),(255,0,0))
        with self.assertRaises(ValueError): select_references(['C:/local.png'],[1.],'Separate references')

    def test_block_mapping_rejects_unknown_model(self):
        self.assertEqual(block_map(set(range(28)),40), tuple(i for i in range(40) if i not in {2,5,8,11,14,17,21,24,27,30,33,36}))
        with self.assertRaises(ValueError): block_map(set(range(28)),32)


if __name__ == '__main__': unittest.main()
