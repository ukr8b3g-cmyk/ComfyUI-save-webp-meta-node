import importlib
import sys
import types
import unittest


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: "."
folder_paths.get_save_image_path = lambda *args, **kwargs: (".", "test", 0, "", "")
sys.modules.setdefault("folder_paths", folder_paths)

webp_save = importlib.import_module("webp_save")


class PromptResolverTests(unittest.TestCase):
    def test_qwen21_dual_output_empty_negative(self):
        prompt = {
            "enc": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "prompt": "positive qwen prompt",
                    "negative_prompt": "",
                },
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["enc", 0],
                    "negative": ["enc", 1],
                    "steps": 25,
                    "cfg": 1,
                },
            },
        }
        node = webp_save.SaveWebPMeta()
        info = node._metadata_from_prompt(prompt)
        self.assertEqual(info["prompt"], "positive qwen prompt")
        self.assertIn("negative_prompt", info)
        self.assertEqual(info["negative_prompt"], "")

    def test_qwen21_dual_output_nonempty_negative(self):
        prompt = {
            "enc": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "prompt": "positive",
                    "negative_prompt": "negative",
                },
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["enc", 0],
                    "negative": ["enc", 1],
                },
            },
        }
        node = webp_save.SaveWebPMeta()
        info = node._metadata_from_prompt(prompt)
        self.assertEqual(info["prompt"], "positive")
        self.assertEqual(info["negative_prompt"], "negative")

    def test_classic_clip_text_encode_regression(self):
        prompt = {
            "p": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}, "_meta": {"title": "Positive"}},
            "n": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad anatomy"}, "_meta": {"title": "Negative"}},
            "s": {"class_type": "KSampler", "inputs": {"positive": ["p", 0], "negative": ["n", 0]}},
        }
        node = webp_save.SaveWebPMeta()
        info = node._metadata_from_prompt(prompt)
        self.assertEqual(info["prompt"], "cat")
        self.assertEqual(info["negative_prompt"], "bad anatomy")

    def test_conditioning_zero_out_is_explicit_empty_negative(self):
        prompt = {
            "p": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
            "z": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["p", 0]}},
            "s": {"class_type": "KSampler", "inputs": {"positive": ["p", 0], "negative": ["z", 0]}},
        }
        node = webp_save.SaveWebPMeta()
        info = node._metadata_from_prompt(prompt)
        self.assertEqual(info["prompt"], "cat")
        self.assertIn("negative_prompt", info)
        self.assertEqual(info["negative_prompt"], "")

    def test_prompt_graph_overrides_workflow_fallback(self):
        prompt = {
            "enc": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": "real positive", "negative_prompt": ""}},
            "s": {"class_type": "KSampler", "inputs": {"positive": ["enc", 0], "negative": ["enc", 1]}},
        }
        workflow = {
            "nodes": [
                {"id": 1, "type": "CLIPTextEncode", "title": "Negative", "widgets_values": ["wrong fallback"]},
            ]
        }
        node = webp_save.SaveWebPMeta()
        info = node._extract_metadata(prompt=prompt, extra_pnginfo={"workflow": workflow})
        self.assertEqual(info["prompt"], "real positive")
        self.assertEqual(info["negative_prompt"], "")


if __name__ == "__main__":
    unittest.main()
