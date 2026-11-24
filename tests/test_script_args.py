"""Eight reference slots in the Forge panel. Slots 5-8 come after the existing arguments, so argument lists
written for four references (API alwayson_scripts, saved presets) keep their meaning.
Gradio and Forge are stubbed; the real scripts/snapref.py runs."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Component:
    created = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.events = []
        Component.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def click(self, *args, **kwargs):
        return None

    def change(self, fn, *args, **kwargs):
        self.events.append(("change", fn, kwargs))
        return None


def kind(name):
    return type(name, (Component,), {})


def stubs():
    gr = ModuleType("gradio")
    for name in ("Group", "Row", "Column", "Accordion", "Dropdown", "Button", "Radio", "Slider", "Image",
                 "Markdown", "Textbox", "Checkbox"):
        setattr(gr, name, kind(name))
    gr.update = lambda **kw: kw
    gr.Error = type("Error", (Exception,), {})
    modules = {"gradio": gr}
    for name in ("modules", "modules.scripts", "modules.shared", "modules.processing", "modules.prompt_parser",
                 "modules.ui_components", "backend", "backend.memory_management", "backend.args"):
        modules[name] = ModuleType(name)
    modules["modules.scripts"].Script = object
    modules["modules.scripts"].AlwaysVisible = object()
    modules["modules.shared"].cmd_opts = SimpleNamespace(lora_dir=str(ROOT / "models"), lora_dirs=[])
    modules["modules.shared"].opts = SimpleNamespace(anima_do_reference=False)
    modules["modules.shared"].state = SimpleNamespace(interrupt=lambda: None)
    modules["modules.processing"].process_images_inner = lambda p, *a, **k: None
    modules["modules.prompt_parser"].SdConditioning = list
    modules["modules.ui_components"].InputAccordion = kind("InputAccordion")
    modules["backend.args"].dynamic_args = SimpleNamespace(ref_latents=[])
    for parent, child in (("modules", "scripts"), ("modules", "shared"), ("modules", "processing"),
                          ("modules", "prompt_parser"), ("modules", "ui_components"),
                          ("backend", "memory_management"), ("backend", "args")):
        setattr(modules[parent], child, modules[f"{parent}.{child}"])
    return modules


def load_script():
    with patch.dict(sys.modules, stubs()):
        spec = importlib.util.spec_from_file_location("snapref_forge_script_test", ROOT / "scripts" / "snapref.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ScriptArgsTests(unittest.TestCase):
    def setUp(self):
        Component.created = []
        self.mod = load_script()
        self.script = self.mod.SnapRefScript()

    def test_ui_has_eight_slots_and_keeps_the_old_argument_order(self):
        with patch.object(self.mod, "model_files", return_value={}):
            values = self.script.ui(False)
        images = [c for c in Component.created if type(c).__name__ == "Image"]
        self.assertEqual([c.kwargs["label"] for c in images], [f"Reference {i}" for i in range(1, 9)])
        labels = [getattr(v, "kwargs", {}).get("label") for v in values]
        self.assertEqual(labels[1:6], ["SnapRef adapter", "Mode", "Combine references", "Fidelity", "Reference size (MP)"])
        self.assertEqual(labels[6:14], ["Reference 1", "Strength 1", "Reference 2", "Strength 2",
                                        "Reference 3", "Strength 3", "Reference 4", "Strength 4"])
        self.assertEqual(labels[14:21], ["Reference caption", "Reference in negative branch", "Extra reference guidance",
                                         "Reference guidance", "Guidance end", "Guidance correction", "Rescale phi"])
        self.assertEqual(labels[21:], ["Reference 5", "Strength 5", "Reference 6", "Strength 6",
                                       "Reference 7", "Strength 7", "Reference 8", "Strength 8"])
        strengths = [name for _, name in self.script.infotext_fields if "Strength" in str(name)]
        self.assertEqual(strengths, [f"SnapRef Strength {i}" for i in range(1, 9)])

    def run_sampling(self, *args):
        seen = {}

        def select(images, weights, layout):
            seen.update(images=images, weights=weights)
            raise RuntimeError("stop after selection")

        dm = type("Anima", (), {})()
        unet = SimpleNamespace(get_model_object=lambda name: dm)
        p = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))
        with patch.object(self.mod, "select_references", select), self.assertRaises(self.mod.gr.Error):
            self.script.process_before_every_sampling(p, *args)
        return seen

    def test_four_reference_argument_lists_still_work(self):
        head = [True, "snapref_v2a.safetensors", "Character", "Separate references", 1.0, 1.0]
        refs = ["a", 1.0, "b", 0.4, None, 1.0, None, 1.0]
        tail = ["", False, False, 4.5, 1.0, "apg", 0.5]
        seen = self.run_sampling(*head, *refs, *tail)
        self.assertEqual(seen["images"], ["a", "b", None, None, None, None, None, None])
        self.assertEqual(seen["weights"], [1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        seen = self.run_sampling(*head, *refs, *tail, "e", 0.3, None, 1.0, None, 1.0, "h", 2.0)
        self.assertEqual(seen["images"], ["a", "b", None, None, "e", None, None, "h"])
        self.assertEqual(seen["weights"], [1.0, 0.4, 1.0, 1.0, 0.3, 1.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
