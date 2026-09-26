import importlib
import sys
import types
import unittest


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: "."
folder_paths.get_save_image_path = lambda *args, **kwargs: (".", "test", 0, "", "")
sys.modules.setdefault("folder_paths", folder_paths)

webp_save = importlib.import_module("webp_save")


class ModernActivePathTests(unittest.TestCase):
    def test_qwen21_switch_cache_and_resolution_selector(self):
        structured = '{"aspect_ratio":"4:5","high_level_description":"test"}'
        prompt = {
            "13": {
                "class_type": "ResolutionSelector",
                "inputs": {"aspect_ratio": "3:4 (Portrait Standard)", "megapixels": 1.0, "multiple": 32},
            },
            "477": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": r"qwen\qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"},
            },
            "478": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"},
            },
            "479": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            "480": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": ["13", 0], "height": ["13", 1], "batch_size": 1},
            },
            "482": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 458890895551063,
                    "steps": 25,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["484", 0],
                    "positive": ["485", 0],
                    "negative": ["485", 1],
                    "latent_image": ["483", 0],
                },
            },
            "483": {
                "class_type": "ComfySwitchNode",
                "inputs": {"switch": True, "on_false": ["485", 2], "on_true": ["480", 0]},
            },
            "484": {
                "class_type": "QwenImage21Cache",
                "inputs": {"device": "auto", "dtype": "default", "model": ["477", 0]},
            },
            "485": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "prompt": ["488", 0],
                    "negative_prompt": "",
                    "resolution": 1024,
                    "clip": ["478", 0],
                    "vae": ["479", 0],
                },
            },
            "488": {
                "class_type": "ComfySwitchNode",
                "inputs": {"switch": True, "on_false": ["492", 0], "on_true": ["492", 0]},
            },
            "492": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": structured}},
            "500": {"class_type": "VAEDecode", "inputs": {"samples": ["482", 0], "vae": ["479", 0]}},
            "501": {"class_type": "SaveWebPMeta", "inputs": {"images": ["500", 0]}},
        }
        info = webp_save.SaveWebPMeta()._metadata_from_prompt(prompt, id="501")
        self.assertEqual(info["prompt"], structured)
        self.assertEqual(info["negative_prompt"], "")
        self.assertEqual(info["model"], r"qwen\qwen_image_2.1_int8_convrot.safetensors")
        self.assertEqual(info["text_encoder"], "qwen3vl_8b_int8_convrot.safetensors")
        self.assertEqual(info["vae"], "qwen_image_2.1_vae_bf16.safetensors")
        self.assertEqual(info["width"], 896)
        self.assertEqual(info["height"], 1184)
        self.assertEqual(info["model_family"], "Qwen Image 2.1")
        self.assertEqual(info["denoise"], 1.0)

    def test_krea2_active_lora_and_zeroout(self):
        prompt = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": r"krea2\kres2_BT_00001_.safetensors", "weight_dtype": "default"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 10,
                    "cfg": 1.0,
                    "sampler_name": "er_sde",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["143", 0],
                    "positive": ["6", 0],
                    "negative": ["8", 0],
                    "latent_image": ["10", 0],
                },
            },
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "5": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["4", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "krea positive", "clip": ["13", 0]}},
            "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
            "10": {"class_type": "EmptyLatentImage", "inputs": {"width": 888, "height": 1184, "batch_size": 1}},
            "13": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"},
            },
            "143": {
                "class_type": "Power Lora Loader (rgthree)",
                "inputs": {
                    "lora_1": {"on": False, "lora": r"krea2\KNPV3_1.safetensors", "strength": 0.2},
                    "lora_2": {"on": True, "lora": r"krea2\active_krea2.safetensors", "strength": 0.3},
                    "model": ["1", 0],
                },
            },
            "150": {"class_type": "SaveWebPMeta", "inputs": {"images": ["5", 0]}},
        }
        info = webp_save.SaveWebPMeta()._metadata_from_prompt(prompt, id="150")
        self.assertEqual(info["prompt"], "krea positive")
        self.assertEqual(info["negative_prompt"], "")
        self.assertEqual(info["model"], r"krea2\kres2_BT_00001_.safetensors")
        self.assertEqual(info["text_encoder"], "qwen3vl_4b_fp8_scaled.safetensors")
        self.assertEqual(info["vae"], "qwen_image_vae.safetensors")
        self.assertEqual(info["width"], 888)
        self.assertEqual(info["height"], 1184)
        self.assertEqual(info["model_family"], "Krea2")
        self.assertIn("active_krea2", info["loras"])
        self.assertNotIn("KNPV3_1", info["loras"])

    def test_save_node_selects_correct_sampler_when_multiple_exist(self):
        prompt = {
            "p1": {"class_type": "CLIPTextEncode", "inputs": {"text": "wrong"}},
            "p2": {"class_type": "CLIPTextEncode", "inputs": {"text": "right"}},
            "s1": {"class_type": "KSampler", "inputs": {"positive": ["p1", 0], "negative": ["p1", 0], "seed": 1}},
            "s2": {"class_type": "KSampler", "inputs": {"positive": ["p2", 0], "negative": ["p2", 0], "seed": 2}},
            "d2": {"class_type": "VAEDecode", "inputs": {"samples": ["s2", 0]}},
            "save": {"class_type": "SaveWebPMeta", "inputs": {"images": ["d2", 0]}},
        }
        info = webp_save.SaveWebPMeta()._metadata_from_prompt(prompt, id="save")
        self.assertEqual(info["seed"], 2)
        self.assertEqual(info["prompt"], "right")


if __name__ == "__main__":
    unittest.main()
