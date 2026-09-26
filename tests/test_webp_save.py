import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import piexif
import piexif.helper
from PIL import Image


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: ""
folder_paths.get_save_image_path = lambda prefix, out_dir, width, height: (out_dir, prefix, 0, "", prefix)
sys.modules["folder_paths"] = folder_paths
spec = importlib.util.spec_from_file_location("webp_save", Path(__file__).resolve().parents[1] / "webp_save.py")
webp_save = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webp_save)


def api_node(node_type, **inputs):
    return {"class_type": node_type, "inputs": inputs}


def workflow_node(node_id, node_type, values=None, links=None):
    return {
        "id": node_id,
        "type": node_type,
        "widgets_values": values or [],
        "inputs": [{"name": name, "link": link} for name, link in (links or {}).items()],
    }


class ImageTensor:
    def __init__(self):
        self.array = np.zeros((8, 12, 3), dtype=np.float32)
        self.shape = self.array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.saver = webp_save.SaveWebPMeta()

    def qwen_prompt(self, negative="avoid blur"):
        return {
            "old_sampler": api_node("KSampler", seed=999, positive=["old_text", 0], model=["old_model", 0]),
            "old_text": api_node("CLIPTextEncode", text="unrelated"),
            "old_model": api_node("CheckpointLoaderSimple", ckpt_name="unrelated.safetensors"),
            "459:452": api_node("TextEncodeQwenImage21", prompt='{"scene":"editorial"}', negative_prompt=negative, clip=["clip", 0]),
            "clip": api_node("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image"),
            "model": api_node("UNETLoader", unet_name="qwen\\qwen_image_2.1_int8_convrot.safetensors"),
            "vae": api_node("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors"),
            "size": api_node("ResolutionSelector", aspect_ratio="1:1 (Square)", megapixels=1, multiple=32),
            "latent": api_node("EmptySD3LatentImage", width=["size", 0], height=["size", 1]),
            "sampler": api_node("KSampler", seed=42, steps=25, cfg=1, sampler_name="euler", scheduler="simple", denoise=1,
                                positive=["459:452", 0], negative=["459:452", 1], model=["model", 0], latent_image=["latent", 0]),
            "decode": api_node("VAEDecode", samples=["sampler", 0], vae=["vae", 0]),
            "save": api_node("SaveWebPMeta", images=["decode", 0]),
        }

    def test_qwen_slots_and_active_path(self):
        info = self.saver._extract_metadata(self.qwen_prompt(), id="save")
        self.assertEqual(info["prompt"], '{"scene":"editorial"}')
        self.assertEqual(info["negative_prompt"], "avoid blur")
        self.assertEqual(info["seed"], 42)
        self.assertEqual(info["steps"], 25)
        self.assertEqual(info["cfg"], 1)
        self.assertEqual(info["sampler"], "euler")
        self.assertEqual(info["scheduler"], "simple")
        self.assertEqual(info["denoise"], 1)
        self.assertEqual(info["model"], "qwen\\qwen_image_2.1_int8_convrot.safetensors")
        self.assertEqual(info["model_family"], "Qwen Image 2.1")
        self.assertEqual(info["text_encoder"], "qwen3vl_8b_int8_convrot.safetensors")
        self.assertEqual(info["vae"], "qwen_image_2.1_vae_bf16.safetensors")
        self.assertEqual((info["width"], info["height"]), (1024, 1024))

    def test_qwen_empty_negative_is_resolved(self):
        prompt = self.qwen_prompt(negative="")
        self.assertEqual(webp_save._prompt_reference(["459:452", 1]), ("459:452", 1))
        self.assertEqual(webp_save._prompt_resolve_text(prompt, ["459:452", 1], True), ("", True))
        info = self.saver._extract_metadata(prompt, id="save")
        self.assertIn("negative_prompt", info)
        self.assertEqual(info["negative_prompt"], "")
        self.assertNotIn("Negative prompt:", self.saver._build_a1111_parameters(info, 1024, 1024))

    def test_api_prompt_overrides_workflow(self):
        prompt = self.qwen_prompt(negative="")
        workflow = {
            "nodes": [workflow_node(1, "KSampler", [99, "fixed", 5, 7, "heun", "normal", 0.5]),
                      workflow_node(2, "CLIPTextEncode", ["stale positive"]),
                      workflow_node(3, "CLIPTextEncode", ["stale negative"])],
            "links": [],
        }
        info = self.saver._extract_metadata(prompt, {"workflow": workflow}, id="save")
        self.assertEqual(info["seed"], 42)
        self.assertEqual(info["negative_prompt"], "")
        self.assertEqual(info["prompt"], '{"scene":"editorial"}')

    def test_switch_follows_only_active_branch(self):
        prompt = self.qwen_prompt()
        prompt["true"] = api_node("CLIPTextEncode", text="selected")
        prompt["false"] = api_node("CLIPTextEncode", text="stale")
        prompt["switch"] = api_node("ComfySwitchNode", switch=True, on_true=["true", 0], on_false=["false", 0])
        prompt["sampler"]["inputs"]["positive"] = ["switch", 0]
        self.assertEqual(self.saver._extract_metadata(prompt, id="save")["prompt"], "selected")
        prompt["switch"]["inputs"]["switch"] = False
        self.assertEqual(self.saver._extract_metadata(prompt, id="save")["prompt"], "stale")

    def test_primitive_strings_and_concatenation(self):
        prompt = self.qwen_prompt(negative="")
        prompt["part"] = api_node("PrimitiveStringMultiline", text="first")
        prompt["concat"] = api_node("StringConcatenate", string_a=["part", 0], string_b="second", delimiter=" ")
        prompt["459:452"]["inputs"]["prompt"] = ["concat", 0]
        info = self.saver._extract_metadata(prompt, id="save")
        self.assertEqual(info["prompt"], "first second")
        self.assertEqual(info["negative_prompt"], "")

    def test_krea_zero_out_and_active_power_lora(self):
        prompt = self.qwen_prompt()
        prompt["positive"] = api_node("CLIPTextEncode", text="Krea scene", clip=["clip", 0])
        prompt["zero"] = api_node("ConditioningZeroOut", conditioning=["positive", 0])
        prompt["clip"]["inputs"]["type"] = "krea2"
        prompt["clip"]["inputs"]["clip_name"] = "qwen3vl_4b_fp8_scaled.safetensors"
        prompt["model"]["inputs"]["unet_name"] = "krea2\\kres2_BT_00001_.safetensors"
        prompt["vae"]["inputs"]["vae_name"] = "qwen_image_vae.safetensors"
        prompt["power"] = api_node("Power Lora Loader (rgthree)", model=["model", 0],
                                   lora_1={"on": False, "lora": "KNPV3_1.safetensors", "strength": 0.2},
                                   lora_2={"on": True, "lora": "lucomk_ill_lora4am3_krea2_safetensors.safetensors", "strength": 0.3})
        prompt["sampler"]["inputs"].update(positive=["positive", 0], negative=["zero", 0], model=["power", 0],
                                           steps=10, cfg=1, seed=42, sampler_name="er_sde", scheduler="simple")
        info = self.saver._extract_metadata(prompt, id="save")
        self.assertEqual(info["prompt"], "Krea scene")
        self.assertEqual(info["negative_prompt"], "")
        self.assertEqual(info["model_family"], "Krea2")
        self.assertEqual(info["text_encoder"], "qwen3vl_4b_fp8_scaled.safetensors")
        self.assertEqual(info["vae"], "qwen_image_vae.safetensors")
        self.assertEqual(info["model"], "krea2\\kres2_BT_00001_.safetensors")
        self.assertIn("lucomk_ill_lora4am3_krea2_safetensors", info["loras"])
        self.assertNotIn("KNPV3_1", info["loras"])

    def test_standard_lora_zero_strength_is_inactive(self):
        prompt = self.qwen_prompt()
        prompt["lora"] = api_node("LoraLoader", model=["model", 0], lora_name="inactive.safetensors", strength_model=0)
        prompt["sampler"]["inputs"]["model"] = ["lora", 0]
        self.assertNotIn("loras", self.saver._extract_metadata(prompt, id="save"))
        prompt["lora"]["inputs"]["strength_model"] = 0.7
        self.assertIn("inactive", self.saver._extract_metadata(prompt, id="save")["loras"])

    def test_sdxl_and_anima_regression(self):
        prompt = self.qwen_prompt()
        prompt["positive"] = api_node("CLIPTextEncode", text="SDXL positive")
        prompt["negative"] = api_node("CLIPTextEncode", text="SDXL negative")
        prompt["sampler"]["inputs"].update(positive=["positive", 0], negative=["negative", 0])
        prompt["model"] = api_node("CheckpointLoaderSimple", ckpt_name="sdxl.safetensors")
        info = self.saver._extract_metadata(prompt, id="save")
        self.assertEqual((info["prompt"], info["negative_prompt"], info["model"]),
                         ("SDXL positive", "SDXL negative", "sdxl.safetensors"))
        prompt["anima"] = api_node("AnimaRegionalCanvas", quality_prompt="quality", scene_prompt="scene",
                                   red_prompt="red", blue_prompt="blue", negative_prompt="no artifacts")
        prompt["sampler"]["inputs"].update(positive=["anima", 0], negative=["anima", 1])
        info = self.saver._extract_metadata(prompt, id="save")
        self.assertEqual(info["prompt"], "quality\n\nscene\n\nred\n\nblue")
        self.assertEqual(info["negative_prompt"], "no artifacts")

    def test_workflow_slots_and_active_sampler(self):
        nodes = [workflow_node(1, "KSampler", [99, "fixed", 5, 7, "heun", "normal", 0.5]),
                 workflow_node(2, "TextEncodeQwenImage21", ["selected", ""]),
                 workflow_node(3, "KSampler", [42, "fixed", 25, 1, "euler", "simple", 1],
                               {"positive": 12, "negative": 13, "model": 14}),
                 workflow_node(4, "VAEDecode", links={"samples": 15}),
                 workflow_node(5, "SaveWebPMeta", links={"images": 16}),
                 workflow_node(6, "UNETLoader", ["qwen_image_2.1.safetensors"])]
        workflow = {"nodes": nodes,
                    "links": [[12, 2, 0, 3, 1, "CONDITIONING"], [13, 2, 1, 3, 2, "CONDITIONING"],
                              [14, 6, 0, 3, 0, "MODEL"], [15, 3, 0, 4, 0, "LATENT"],
                              [16, 4, 0, 5, 0, "IMAGE"]]}
        self.assertEqual(webp_save._graph_link_sources(workflow)[13], (2, 1))
        info = self.saver._extract_metadata(extra_pnginfo={"workflow": workflow}, id=5)
        self.assertEqual(info["prompt"], "selected")
        self.assertEqual(info["negative_prompt"], "")
        self.assertEqual(info["seed"], 42)
        self.assertEqual(info["model"], "qwen_image_2.1.safetensors")

    def test_workflow_krea_active_lora_switch_and_vae(self):
        nodes = [workflow_node(1, "CLIPTextEncode", ["Krea portrait"], {"clip": 10}),
                 workflow_node(2, "CLIPTextEncode", ["unused"]),
                 workflow_node(3, "ConditioningZeroOut", links={"conditioning": 11}),
                 workflow_node(4, "ComfySwitchNode", [True], {"on_true": 12, "on_false": 13}),
                 workflow_node(5, "ComfySwitchNode", [True], {"on_true": 14, "on_false": 15}),
                 workflow_node(6, "KSampler", [42, "fixed", 10, 1, "er_sde", "simple", 1],
                               {"positive": 16, "negative": 17, "model": 18, "latent_image": 19}),
                 workflow_node(7, "Power Lora Loader (rgthree)",
                               [{"on": False, "lora": "KNPV3_1.safetensors", "strength": 0.2},
                                {"on": True, "lora": "active_krea.safetensors", "strength": 0.3}], {"model": 20}),
                 workflow_node(8, "EmptyLatentImage", [1024, 768]),
                 workflow_node(9, "UNETLoader", ["krea2\\kres2_BT_00001_.safetensors"]),
                 workflow_node(10, "CLIPLoader", ["qwen3vl_4b_fp8_scaled.safetensors", "krea2"]),
                 workflow_node(11, "VAELoader", ["qwen_image_vae.safetensors"]),
                 workflow_node(12, "VAEDecode", links={"samples": 21, "vae": 22}),
                 workflow_node(13, "SaveWebPMeta", links={"images": 23})]
        edges = [(10, 10, 0, 1), (11, 1, 0, 3), (12, 1, 0, 4), (13, 2, 0, 4),
                 (14, 3, 0, 5), (15, 2, 0, 5), (16, 4, 0, 6), (17, 5, 0, 6),
                 (18, 7, 0, 6), (19, 8, 0, 6), (20, 9, 0, 7), (21, 6, 0, 12),
                 (22, 11, 0, 12), (23, 12, 0, 13)]
        workflow = {"nodes": nodes, "links": [[link, source, slot, target, 0, "*"] for link, source, slot, target in edges]}
        info = self.saver._extract_metadata(extra_pnginfo={"workflow": workflow}, id=13)
        self.assertEqual(info["prompt"], "Krea portrait")
        self.assertEqual(info["negative_prompt"], "")
        self.assertEqual(info["model_family"], "Krea2")
        self.assertEqual(info["text_encoder"], "qwen3vl_4b_fp8_scaled.safetensors")
        self.assertEqual(info["vae"], "qwen_image_vae.safetensors")
        self.assertEqual(info["model"], "krea2\\kres2_BT_00001_.safetensors")
        self.assertEqual(info["loras"], "<lora:active_krea:0.3>")
        self.assertEqual((info["width"], info["height"]), (1024, 768))

    def test_filename_patterns(self):
        info = {"seed": 42, "width": 1024, "height": 768, "prompt": "portrait", "negative_prompt": "blur", "model": "model.safetensors"}
        result = self.saver._format_filename("%seed%_%width%_%height%_%pprompt:4%_%nprompt:2%_%model:3%_%date:yyyy%_%KSampler.steps%", info,
                                             prompt={"1": api_node("KSampler", steps=25)})
        self.assertRegex(result, r"^42_1024_768_port_bl_mod_\d{4}_25$")

    def test_legacy_a1111_fields(self):
        info = {"prompt": "portrait", "negative_prompt": "blur", "steps": 20,
                "sampler": "euler", "scheduler": "karras", "cfg": 7, "seed": 42,
                "clip_skip": 2, "rng_source": "CPU", "eta_noise_seed_delta": 31337,
                "emphasis_mode": "Original", "method": "Full"}
        text = self.saver._build_a1111_parameters(info, 1024, 768)
        for expected in ("Negative prompt: blur", "Sampler: Euler Karras", "Clip skip: 2",
                         "RNG source: CPU", "Eta noise seed delta: 31337", "Emphasis: Original", "Method: Full"):
            self.assertIn(expected, text)

    def test_png_and_webp_round_trip(self):
        prompt = self.qwen_prompt(negative="")
        prompt["sampler"]["inputs"]["latent_image"] = ["459:452", 2]
        workflow = {"nodes": [], "links": []}
        with tempfile.TemporaryDirectory() as output_dir:
            folder_paths.get_output_directory = lambda: output_dir
            for file_format in ("png", "webp"):
                result = self.saver.save_webp([ImageTensor()], filename_prefix=f"{file_format}_%width%x%height%", file_format=file_format,
                                              id="save", prompt=prompt, extra_pnginfo={"workflow": workflow})
                filename = result["ui"]["images"][0]["filename"]
                self.assertTrue(filename.startswith(f"{file_format}_12x8"))
                with Image.open(Path(output_dir) / filename) as image:
                    if file_format == "png":
                        parameters = image.info["parameters"]
                        self.assertEqual(json.loads(image.info["prompt"]), prompt)
                        self.assertEqual(json.loads(image.info["workflow"]), workflow)
                    else:
                        exif = piexif.load(image.info["exif"])
                        parameters = piexif.helper.UserComment.load(exif["Exif"][piexif.ExifIFD.UserComment])
                        self.assertEqual(exif["0th"][piexif.ImageIFD.ImageDescription].decode(), parameters)
                        tags = [exif["0th"].get(tag, b"").decode() for tag in webp_save.EXIF_TEXT_TAGS]
                        self.assertEqual(json.loads(next(tag[7:] for tag in tags if tag.startswith("prompt:"))), prompt)
                        self.assertEqual(json.loads(next(tag[9:] for tag in tags if tag.startswith("workflow:"))), workflow)
                    self.assertIn('{"scene":"editorial"}', parameters)
                    self.assertIn("Size: 12x8", parameters)
                    self.assertNotIn("Negative prompt:", parameters)


if __name__ == "__main__":
    unittest.main()
