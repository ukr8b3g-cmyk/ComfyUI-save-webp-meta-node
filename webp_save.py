import hashlib
import json
import logging
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, Optional

import folder_paths

try:
    from .metadata_resolver import resolve_metadata, to_legacy_info, MetadataSession
except ImportError:
    from metadata_resolver import resolve_metadata, to_legacy_info, MetadataSession
from PIL import Image
from PIL.PngImagePlugin import PngInfo

LORA_MANAGER_PY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "comfyui-lora-manager", "py")
if os.path.isdir(LORA_MANAGER_PY) and LORA_MANAGER_PY not in sys.path:
    sys.path.append(LORA_MANAGER_PY)

try:
    from services.service_registry import ServiceRegistry
except Exception:
    ServiceRegistry = None

try:
    import piexif
    import piexif.helper
except Exception:
    piexif = None


EXIF_USER_COMMENT = piexif.ExifIFD.UserComment if piexif else 37510
EXIF_IMAGE_DESCRIPTION = piexif.ImageIFD.ImageDescription if piexif else 270
EXIF_MAKE = piexif.ImageIFD.Make if piexif else 271
EXIF_MODEL = piexif.ImageIFD.Model if piexif else 272
A1111_EXIF_BYTES = b"UNICODE\0"
EXIF_TEXT_TAGS = (
    piexif.ImageIFD.Make,
    piexif.ImageIFD.Software,
    piexif.ImageIFD.Artist,
    piexif.ImageIFD.Copyright,
    piexif.ImageIFD.DocumentName,
    piexif.ImageIFD.DateTime,
    piexif.ImageIFD.HostComputer,
) if piexif else (271, 305, 315, 33432, 269, 306, 316)

SAMPLER_MAP = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "heun": "Heun",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "lms": "LMS",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "ddim": "DDIM",
    "lcm": "LCM",
}

SCHEDULER_MAP = {
    "normal": "Simple",
    "karras": "Karras",
    "exponential": "Exponential",
    "sgm_uniform": "SGM Uniform",
    "sgm_quadratic": "SGM Quadratic",
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _png_compress_level_from_quality(value: Any) -> int:
    quality = max(1, min(100, _as_int(value, 80)))
    return max(0, min(9, round((100 - quality) * 9 / 99)))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _exif_text_bytes(value: Any) -> bytes:
    text = _as_text(value)
    return text.encode("utf-8", errors="replace")


def _clean_model_name(value: Any) -> str:
    text = _as_text(value).strip()
    if not text:
        return ""
    return os.path.splitext(os.path.basename(text.replace("\\", "/")))[0]


def _sanitize_filename_part(value: Any) -> str:
    text = _as_text(value).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r'[<>:"\\|?*]', "_", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .")
    return text or "unknown"


def _format_date_pattern(fmt: str) -> str:
    from datetime import datetime

    now = datetime.now()
    date_table = {
        "yyyy": f"{now.year:04d}",
        "yy": f"{now.year % 100:02d}",
        "MM": f"{now.month:02d}",
        "dd": f"{now.day:02d}",
        "hh": f"{now.hour:02d}",
        "mm": f"{now.minute:02d}",
        "ss": f"{now.second:02d}",
    }
    for key, value in date_table.items():
        fmt = fmt.replace(key, value)
    return fmt


def _limit_text(value: str, parts: list[str]) -> str:
    if len(parts) >= 2:
        try:
            return value[: int(parts[1])]
        except Exception:
            return value
    return value


def _tensor_to_pil(image_tensor) -> Image.Image:
    img = image_tensor.detach().cpu().numpy()
    if img.ndim != 3 or img.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported image tensor shape: {img.shape}")
    if img.shape[-1] == 1:
        img = img.repeat(3, axis=-1)
    if img.shape[-1] == 4:
        rgb = (img[..., :3] * 255.0).clip(0, 255).astype("uint8")
        alpha = (img[..., 3] * 255.0).clip(0, 255).astype("uint8")
        pil = Image.fromarray(rgb, mode="RGB")
        pil.putalpha(Image.fromarray(alpha, mode="L"))
        return pil
    return Image.fromarray((img * 255.0).clip(0, 255).astype("uint8"), mode="RGB")


def _node_inputs(node: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    return node.get("inputs") if isinstance(node.get("inputs"), dict) else {}


def _first_node(prompt: Any, class_names: Iterable[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    names = set(class_names)
    for node in prompt.values():
        if isinstance(node, dict) and node.get("class_type") in names:
            return node
    return None


def _prompt_reference(value: Any) -> Optional[tuple[str, int]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[1], int):
        return str(value[0]), value[1]
    return None


def _prompt_path_node(prompt: Any, value: Any, types: Iterable[str], input_names: Iterable[str], seen=None):
    reference = _prompt_reference(value)
    if not isinstance(prompt, dict) or reference is None:
        return None
    seen = set() if seen is None else seen
    if reference in seen:
        return None
    seen.add(reference)
    node = prompt.get(reference[0])
    if not isinstance(node, dict):
        return None
    if node.get("class_type") in types:
        return node
    inputs = _node_inputs(node)
    if node.get("class_type") == "ComfySwitchNode":
        switch = inputs.get("switch")
        active = switch is True or _as_text(switch).lower() in {"true", "1"}
        return _prompt_path_node(prompt, inputs.get("on_true" if active else "on_false"), types, input_names, seen)
    for name in input_names:
        found = _prompt_path_node(prompt, inputs.get(name), types, input_names, seen)
        if found is not None:
            return found
    return None


def _prompt_active_sampler(prompt: Any, save_id: Any) -> Optional[Dict[str, Any]]:
    save_node = prompt.get(str(save_id)) if isinstance(prompt, dict) and save_id is not None else None
    if isinstance(save_node, dict) and save_node.get("class_type") == "SaveWebPMeta":
        sampler = _prompt_path_node(
            prompt, _node_inputs(save_node).get("images"),
            ("KSampler", "KSamplerAdvanced"),
            ("images", "image", "samples", "latent", "latent_image"),
        )
        if sampler is not None:
            return sampler
    return _first_node(prompt, ("KSampler", "KSamplerAdvanced"))


def _prompt_node_title(node: Dict[str, Any]) -> str:
    meta = node.get("_meta")
    if isinstance(meta, dict):
        return _as_text(meta.get("title"))
    return ""


ANIMA_REGIONAL_TYPES = {"AnimaRegionalCanvas"}
ANIMA_REGION_PROMPTS = ("red_prompt", "blue_prompt", "yellow_prompt", "green_prompt", "magenta_prompt")


def _anima_prompt_from_inputs(inputs: Dict[str, Any], negative: bool) -> str:
    if negative:
        return _as_text(inputs.get("negative_prompt", "")).strip()
    quality = _as_text(inputs.get("quality_prompt", "") or inputs.get("base_prompt", "")).strip()
    scene = _as_text(inputs.get("scene_prompt", "")).strip()
    parts = [quality, scene]
    parts.extend(_as_text(inputs.get(name, "")).strip() for name in ANIMA_REGION_PROMPTS)
    return "\n\n".join(part for part in parts if part)


def _prompt_resolve_text(prompt: Any, value: Any, negative: bool, seen=None) -> tuple[str, bool]:
    if isinstance(value, str):
        return value, True
    reference = _prompt_reference(value)
    if not isinstance(prompt, dict) or reference is None:
        return "", False
    marker = (reference[0], reference[1], "negative" if negative else "positive")
    seen = set() if seen is None else seen
    if marker in seen:
        return "", False
    seen = seen | {marker}
    node = prompt.get(reference[0])
    if not isinstance(node, dict):
        return "", False
    inputs = _node_inputs(node)
    node_type = node.get("class_type")

    if node_type == "ConditioningZeroOut" and negative:
        return "", True
    if node_type == "ComfySwitchNode":
        switch = inputs.get("switch")
        active = switch is True or _as_text(switch).lower() in {"true", "1"}
        return _prompt_resolve_text(prompt, inputs.get("on_true" if active else "on_false"), negative, seen)
    if node_type == "TextEncodeQwenImage21":
        key = "negative_prompt" if reference[1] == 1 else "prompt" if reference[1] == 0 else None
        if key is not None and key in inputs:
            return _prompt_resolve_text(prompt, inputs[key], negative, seen)
        return "", False
    if node_type in ANIMA_REGIONAL_TYPES:
        return _anima_prompt_from_inputs(inputs, negative), True
    if node_type == "StringConcatenate":
        parts = []
        resolved = False
        for key in ("string_a", "string_b"):
            text, found = _prompt_resolve_text(prompt, inputs.get(key), negative, seen)
            resolved |= found
            if text:
                parts.append(text)
        delimiter = inputs.get("delimiter", "")
        return (_as_text(delimiter).join(parts), True) if resolved else ("", False)
    if node_type in {"CLIPTextEncode", "PrimitiveStringMultiline", "PrimitiveString"}:
        for key in ("text", "value", "string"):
            if key in inputs:
                return _prompt_resolve_text(prompt, inputs[key], negative, seen)
    for key in ("negative" if negative else "positive", "conditioning", "text", "prompt", "value", "string"):
        if key in inputs:
            text, found = _prompt_resolve_text(prompt, inputs[key], negative, seen)
            if found:
                return text, True
    return "", False


def _prompt_text_node(prompt: Any, negative: bool, sampler=None) -> tuple[str, bool]:
    if sampler is not None:
        value = _node_inputs(sampler).get("negative" if negative else "positive")
        text, resolved = _prompt_resolve_text(prompt, value, negative)
        if _prompt_reference(value) is not None:
            return text, resolved
    for node_id, node in prompt.items() if isinstance(prompt, dict) else []:
        if not isinstance(node, dict) or node.get("class_type") != "CLIPTextEncode":
            continue
        if ("negative" in _prompt_node_title(node).lower()) == negative:
            text, resolved = _prompt_resolve_text(prompt, (node_id, 0), negative)
            if resolved and text:
                return text, True
    return "", False


def _prompt_conditioning_node(prompt: Any, value: Any, seen=None):
    reference = _prompt_reference(value)
    if reference is None or not isinstance(prompt, dict):
        return None
    seen = set() if seen is None else seen
    if reference in seen:
        return None
    seen.add(reference)
    node = prompt.get(reference[0])
    if not isinstance(node, dict):
        return None
    if node.get("class_type") in {"CLIPTextEncode", "TextEncodeQwenImage21"}:
        return node
    inputs = _node_inputs(node)
    if node.get("class_type") == "ComfySwitchNode":
        switch = inputs.get("switch")
        active = switch is True or _as_text(switch).lower() in {"true", "1"}
        return _prompt_conditioning_node(prompt, inputs.get("on_true" if active else "on_false"), seen)
    for key in ("conditioning", "positive", "negative"):
        found = _prompt_conditioning_node(prompt, inputs.get(key), seen)
        if found is not None:
            return found
    return None


def _prompt_loras(node: Dict[str, Any]) -> list[str]:
    inputs = _node_inputs(node)
    tags = []
    if "lora_name" in inputs:
        strength = inputs.get("strength_model", 1)
        if _as_text(strength) not in {"0", "0.0"}:
            tags.append(_format_lora_tag(inputs["lora_name"], strength))
    for value in inputs.values():
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, dict) or row.get("on") is False:
                continue
            name = row.get("lora") or row.get("name") or row.get("file_name")
            if name:
                tags.append(_format_lora_tag(name, row.get("strength", 1)))
    return tags


def _prompt_model_path(prompt: Any, sampler: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    value = _node_inputs(sampler).get("model") if sampler else None
    seen = set()
    tags = []
    while (reference := _prompt_reference(value)) is not None and reference not in seen:
        seen.add(reference)
        node = prompt.get(reference[0])
        if not isinstance(node, dict):
            break
        node_type = _as_text(node.get("class_type"))
        inputs = _node_inputs(node)
        if "lora" in node_type.lower():
            tags.extend(_prompt_loras(node))
        if node_type in {"UNETLoader", "CheckpointLoaderSimple", "CheckpointLoader"}:
            for key in ("unet_name", "ckpt_name", "checkpoint", "model_name"):
                if key in inputs:
                    info["model"] = inputs[key]
                    break
            break
        value = inputs.get("model")
    if tags:
        info["loras"] = " ".join(tags)
    return info


def _resolution_dimensions(inputs: Dict[str, Any]) -> Optional[tuple[int, int]]:
    ratio = _as_text(inputs.get("aspect_ratio", "")).strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)", ratio)
    try:
        aspect = float(match[1]) / float(match[2]) if match else float(ratio)
        pixels = float(inputs.get("megapixels")) * 1024 * 1024
        multiple = max(1, int(inputs.get("multiple", 1)))
        if aspect <= 0 or pixels <= 0:
            return None
        width = max(multiple, round(math.sqrt(pixels * aspect) / multiple) * multiple)
        height = max(multiple, round(math.sqrt(pixels / aspect) / multiple) * multiple)
        return width, height
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _prompt_number(prompt: Any, value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(value)
    reference = _prompt_reference(value)
    if reference is None or not isinstance(prompt, dict):
        return None
    node = prompt.get(reference[0])
    if not isinstance(node, dict):
        return None
    inputs = _node_inputs(node)
    if node.get("class_type") == "ResolutionSelector":
        dimensions = _resolution_dimensions(inputs)
        return dimensions[reference[1]] if dimensions and reference[1] in (0, 1) else None
    for key in ("value", "number", "int"):
        if isinstance(inputs.get(key), (int, float)):
            return int(inputs[key])
    return None


def _graph_nodes(workflow: Any) -> list[Dict[str, Any]]:
    if not isinstance(workflow, dict):
        return []
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, dict)]


def _graph_first(nodes: list[Dict[str, Any]], types: Iterable[str]) -> Optional[Dict[str, Any]]:
    wanted = set(types)
    for node in nodes:
        if node.get("type") in wanted:
            return node
    return None


def _graph_values(node: Optional[Dict[str, Any]]) -> list[Any]:
    if not isinstance(node, dict):
        return []
    values = node.get("widgets_values")
    return values if isinstance(values, list) else []


def _graph_widget_value(node: Optional[Dict[str, Any]], input_names: Iterable[str]) -> Any:
    if not isinstance(node, dict):
        return None
    wanted = set(input_names)
    inputs = node.get("inputs")
    values = _graph_values(node)
    if not isinstance(inputs, list) or not values:
        return None
    widget_index = 0
    for item in inputs:
        if not isinstance(item, dict):
            continue
        if "widget" not in item:
            continue
        if item.get("name") in wanted and widget_index < len(values):
            return values[widget_index]
        widget_index += 1
    return None


def _graph_anima_prompt(node: Optional[Dict[str, Any]], negative: bool) -> str:
    if not isinstance(node, dict) or node.get("type") not in ANIMA_REGIONAL_TYPES:
        return ""
    props = node.get("properties")
    saved = props.get("animaPrompts") if isinstance(props, dict) else None
    if isinstance(saved, dict):
        if negative:
            return _as_text(saved.get("negative_prompt", "")).strip()
        quality = _as_text(saved.get("quality_prompt", "") or saved.get("base_prompt", "")).strip()
        scene = _as_text(saved.get("scene_prompt", "")).strip()
        parts = [quality, scene]
        parts.extend(_as_text(saved.get(name, "")).strip() for name in ANIMA_REGION_PROMPTS)
        text = "\n\n".join(part for part in parts if part)
        if text.strip():
            return text

    values = _graph_values(node)
    if negative:
        if len(values) > 12:
            return _as_text(values[12]).strip()
        return _as_text(values[11] if len(values) > 11 else "").strip()
    indexes = (5, 6, 7, 8, 9, 10, 11) if len(values) > 12 else (5, 6, 7, 8, 9, 10)
    return "\n\n".join(_as_text(values[index]).strip() for index in indexes if len(values) > index and _as_text(values[index]).strip())


def _graph_node_names(node: Dict[str, Any]) -> set[str]:
    names = set()
    for value in (
        node.get("title"),
        node.get("type"),
        node.get("properties", {}).get("Node name for S&R") if isinstance(node.get("properties"), dict) else None,
    ):
        text = _as_text(value).strip()
        if text:
            names.add(text)
    return names


def _graph_cross_node_value(workflow: Any, node_name: str, widget_name: str) -> Any:
    nodes = _graph_nodes(workflow)
    # Match by Node name for S&R/title/type. Prefer exact, then case-insensitive.
    for case_sensitive in (True, False):
        for node in nodes:
            names = _graph_node_names(node)
            if not case_sensitive:
                names = {name.lower() for name in names}
                target = node_name.lower()
            else:
                target = node_name
            if target not in names:
                continue
            value = _graph_widget_value(node, (widget_name,))
            if value is not None:
                return value
            values = _graph_values(node)
            # Common aliases/fallbacks for built-in nodes.
            alias_indexes = {
                "seed": 0,
                "steps": 2,
                "cfg": 3,
                "sampler_name": 4,
                "sampler": 4,
                "scheduler": 5,
                "width": 0,
                "height": 1,
                "ckpt_name": 0,
                "model": 0,
            }
            index = alias_indexes.get(widget_name)
            if index is not None and len(values) > index:
                return values[index]
    return None


def _prompt_cross_node_value(prompt: Any, node_name: str, widget_name: str) -> Any:
    if not isinstance(prompt, dict):
        return None
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        names = {
            _as_text(node.get("class_type")),
            _as_text(node.get("_meta", {}).get("title")) if isinstance(node.get("_meta"), dict) else "",
        }
        if node_name not in names and node_name.lower() not in {name.lower() for name in names if name}:
            continue
        inputs = _node_inputs(node)
        if widget_name in inputs:
            return inputs[widget_name]
    return None


def _clip_skip_value(value: Any) -> Any:
    try:
        number = int(value)
    except Exception:
        return value
    if number < 0:
        return abs(number)
    return number


def _find_named_value(data: Any, names: Iterable[str]) -> Any:
    wanted = {name.lower() for name in names}
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in wanted and value not in (None, "", []):
                return value
        for value in data.values():
            found = _find_named_value(value, wanted)
            if found not in (None, "", []):
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_named_value(item, wanted)
            if found not in (None, "", []):
                return found
    return None


def _looks_like_rng_source(value: Any) -> bool:
    text = _as_text(value).strip().lower()
    return text in {"cpu", "gpu", "cuda", "nv", "default"}


def _graph_input_link(node: Dict[str, Any], input_name: str) -> Optional[int]:
    inputs = node.get("inputs")
    if not isinstance(inputs, list):
        return None
    for item in inputs:
        if isinstance(item, dict) and item.get("name") == input_name:
            link = item.get("link")
            return link if isinstance(link, int) else None
    return None


def _graph_link_sources(workflow: Any) -> Dict[int, tuple[int, int]]:
    if not isinstance(workflow, dict):
        return {}
    sources: Dict[int, tuple[int, int]] = {}
    links = workflow.get("links")
    if not isinstance(links, list):
        return sources
    for link in links:
        if isinstance(link, list) and len(link) >= 3 and all(isinstance(item, int) for item in link[:3]):
            sources[link[0]] = (link[1], link[2])
    return sources


def _graph_node_map(nodes: list[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {node["id"]: node for node in nodes if isinstance(node.get("id"), int)}


def _graph_source(node, input_name, node_map, link_sources):
    reference = link_sources.get(_graph_input_link(node, input_name)) if isinstance(node, dict) else None
    return (node_map.get(reference[0]), reference[1]) if reference else (None, 0)


def _graph_path_node(node, types, input_names, node_map, link_sources, seen=None):
    if not isinstance(node, dict):
        return None
    seen = set() if seen is None else seen
    node_id = node.get("id")
    if node_id in seen:
        return None
    seen.add(node_id)
    if node.get("type") in types:
        return node
    if node.get("type") == "ComfySwitchNode":
        switch = _graph_widget_value(node, ("switch",))
        if switch is None:
            values = _graph_values(node)
            switch = values[0] if values else False
        active = switch is True or _as_text(switch).lower() in {"true", "1"}
        source, _ = _graph_source(node, "on_true" if active else "on_false", node_map, link_sources)
        return _graph_path_node(source, types, input_names, node_map, link_sources, seen)
    for key in input_names:
        source, _ = _graph_source(node, key, node_map, link_sources)
        found = _graph_path_node(source, types, input_names, node_map, link_sources, seen)
        if found is not None:
            return found
    return None


def _graph_active_sampler(nodes, save_id, node_map, link_sources):
    try:
        save_node = node_map.get(int(save_id))
    except (TypeError, ValueError):
        save_node = None
    if isinstance(save_node, dict) and save_node.get("type") == "SaveWebPMeta":
        source, _ = _graph_source(save_node, "images", node_map, link_sources)
        sampler = _graph_path_node(
            source, ("KSampler", "KSamplerAdvanced"),
            ("images", "image", "samples", "latent", "latent_image"), node_map, link_sources,
        )
        if sampler is not None:
            return sampler
    return _graph_first(nodes, ("KSampler", "KSamplerAdvanced"))


def _graph_resolve_text(node, slot, negative, node_map, link_sources, seen=None) -> tuple[str, bool]:
    if not isinstance(node, dict):
        return "", False
    marker = (node.get("id"), slot, "negative" if negative else "positive")
    seen = set() if seen is None else seen
    if marker in seen:
        return "", False
    seen = seen | {marker}
    node_type = node.get("type")
    values = _graph_values(node)

    if node_type == "ConditioningZeroOut" and negative:
        return "", True
    if node_type == "ComfySwitchNode":
        switch = _graph_widget_value(node, ("switch",))
        if switch is None and values:
            switch = values[0]
        active = switch is True or _as_text(switch).lower() in {"true", "1"}
        source, source_slot = _graph_source(node, "on_true" if active else "on_false", node_map, link_sources)
        return _graph_resolve_text(source, source_slot, negative, node_map, link_sources, seen)
    if node_type in ANIMA_REGIONAL_TYPES:
        return _graph_anima_prompt(node, negative), True
    if node_type == "TextEncodeQwenImage21":
        key = "negative_prompt" if slot == 1 else "prompt" if slot == 0 else None
        if key is None:
            return "", False
        source, source_slot = _graph_source(node, key, node_map, link_sources)
        if source is not None:
            return _graph_resolve_text(source, source_slot, negative, node_map, link_sources, seen)
        text = _graph_widget_value(node, (key,))
        if text is None and len(values) > slot:
            text = values[slot]
        return (_as_text(text), True) if text is not None else ("", False)
    if node_type == "StringConcatenate":
        parts = []
        resolved = False
        for index, key in enumerate(("string_a", "string_b")):
            source, source_slot = _graph_source(node, key, node_map, link_sources)
            if source is not None:
                text, found = _graph_resolve_text(source, source_slot, negative, node_map, link_sources, seen)
            else:
                text = _graph_widget_value(node, (key,))
                text = text if text is not None else (values[index] if len(values) > index else None)
                found = isinstance(text, str)
            resolved |= found
            if text:
                parts.append(_as_text(text))
        delimiter = _graph_widget_value(node, ("delimiter",))
        delimiter = delimiter if delimiter is not None else (values[2] if len(values) > 2 else "")
        return (_as_text(delimiter).join(parts), True) if resolved else ("", False)
    if node_type in {"CLIPTextEncode", "PrimitiveStringMultiline", "PrimitiveString"}:
        for key in ("text", "value", "string"):
            source, source_slot = _graph_source(node, key, node_map, link_sources)
            if source is not None:
                return _graph_resolve_text(source, source_slot, negative, node_map, link_sources, seen)
            text = _graph_widget_value(node, (key,))
            if text is not None:
                return _as_text(text), True
        if values and isinstance(values[0], str):
            return values[0], True
    for key in ("negative" if negative else "positive", "conditioning"):
        source, source_slot = _graph_source(node, key, node_map, link_sources)
        if source is not None:
            text, found = _graph_resolve_text(source, source_slot, negative, node_map, link_sources, seen)
            if found:
                return text, True
    return "", False


def _graph_text_node(workflow: Any, nodes: list[Dict[str, Any]], negative: bool, sampler=None) -> tuple[str, bool]:
    node_map = _graph_node_map(nodes)
    link_sources = _graph_link_sources(workflow)
    if sampler is not None:
        source, slot = _graph_source(sampler, "negative" if negative else "positive", node_map, link_sources)
        text, resolved = _graph_resolve_text(source, slot, negative, node_map, link_sources)
        if _graph_input_link(sampler, "negative" if negative else "positive") is not None:
            return text, resolved
    for node in nodes:
        if node.get("type") in ANIMA_REGIONAL_TYPES:
            text = _graph_anima_prompt(node, negative=negative)
            if text.strip():
                return text, True

    # Prefer titled negative CLIP nodes for negative prompt, and non-negative CLIP nodes for prompt.
    for node in nodes:
        if node.get("type") != "CLIPTextEncode":
            continue
        title = _as_text(node.get("title") or node.get("properties", {}).get("Node name for S&R"))
        is_negative = "negative" in title.lower()
        if is_negative != negative:
            continue
        text, _ = _graph_resolve_text(node, 0, negative, node_map, link_sources)
        if text.strip():
            return text, True
    # Fallback to primitive/string nodes for positive prompt only.
    if not negative:
        for node in nodes:
            if node.get("type") in {"PrimitiveStringMultiline", "PrimitiveString", "StringConcatenate"}:
                text, _ = _graph_resolve_text(node, 0, negative, node_map, link_sources)
                if text.strip():
                    return text, True
    return "", False


def _format_loras_from_any(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        matches = re.findall(r"<lora:([^:>]+):([^>]+)>", value)
        if matches:
            return " ".join(_format_lora_tag(name, strength) for name, strength in matches)
        return value
    if isinstance(value, dict):
        name = value.get("lora") or value.get("name") or value.get("file_name")
        strength = value.get("strength", value.get("weight", value.get("multiplier", 1.0)))
        if name:
            return _format_lora_tag(name, strength)
        return " ".join(_format_lora_tag(k, v) for k, v in value.items() if k)
    if isinstance(value, list):
        tags = []
        for item in value:
            tag = _format_loras_from_any(item)
            if tag:
                tags.append(tag)
        return " ".join(tags)
    return ""


def _format_lora_tag(name: Any, strength: Any) -> str:
    name_text = _lora_basename(name)
    try:
        strength_text = f"{float(strength):.15g}"
    except Exception:
        strength_text = _as_text(strength).strip() or "1.0"
    return f"<lora:{name_text}:{strength_text}>"


def _lora_basename(name: Any) -> str:
    name_text = _as_text(name).replace("\\", "/").strip()
    return os.path.splitext(os.path.basename(name_text))[0]


def _get_lora_scanner() -> Any:
    # First try the ServiceRegistry imported by this node.
    registries = []
    if ServiceRegistry is not None:
        registries.append(ServiceRegistry)

    # LoRA Manager may have loaded its ServiceRegistry under a package-specific
    # module name. Reusing that class is necessary because its _services holds
    # the already-created lora_scanner.
    for module_name, module in list(sys.modules.items()):
        if not module_name.endswith("service_registry"):
            continue
        registry = getattr(module, "ServiceRegistry", None)
        if registry is not None and registry not in registries:
            registries.append(registry)

    for registry in registries:
        try:
            scanner = getattr(registry, "_services", {}).get("lora_scanner")
        except Exception:
            scanner = None
        if scanner is not None and hasattr(scanner, "get_hash_by_filename"):
            return scanner
    return None


def _lora_metadata_hash_by_name(candidates: Iterable[str]) -> str:
    try:
        directories = folder_paths.get_folder_paths("loras")
    except Exception:
        return ""
    for directory in directories:
        for candidate in candidates:
            base = _lora_basename(candidate)
            for suffix in (".metadata.json", ".safetensors.metadata.json"):
                path = os.path.join(directory, base + suffix)
                try:
                    if os.path.getsize(path) > 1024 * 1024:
                        continue
                    with open(path, encoding="utf-8") as file:
                        data = json.load(file)
                    if isinstance(data, dict) and data.get("sha256"):
                        return str(data["sha256"])
                except (OSError, ValueError):
                    continue
    return ""


def _lora_file_hash_by_name(candidates: Iterable[str]) -> str:
    try:
        lora_dirs = folder_paths.get_folder_paths("loras")
    except Exception:
        lora_dirs = []

    candidate_names = {_as_text(candidate) for candidate in candidates if candidate}
    candidate_bases = {_lora_basename(candidate) for candidate in candidate_names}

    for candidate in candidate_names:
        try:
            path = folder_paths.get_full_path("loras", candidate)
        except Exception:
            path = None
        if path and os.path.isfile(path):
            return _sha256_file(path)

    for lora_dir in lora_dirs:
        for root, _dirs, files in os.walk(lora_dir):
            for file_name in files:
                if os.path.splitext(file_name)[1].lower() not in {".safetensors", ".pt", ".ckpt"}:
                    continue
                file_base = _lora_basename(file_name)
                if file_name not in candidate_names and file_base not in candidate_bases:
                    continue
                return _sha256_file(os.path.join(root, file_name))
    return ""


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scanner_hash_by_name(scanner: Any, candidates: Iterable[str]) -> str:
    try:
        raw = getattr(getattr(scanner, "_cache", None), "raw_data", None)
        if not isinstance(raw, list) or len(raw) > 10000:
            return ""
        matches = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            names = {str(item.get(key) or "").replace("\\", "/") for key in ("file_name", "name", "model_name", "file_path", "path")}
            if any(str(candidate).replace("\\", "/") in names for candidate in candidates):
                digest = str(item.get("sha256") or item.get("hash") or "")
                if re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                    matches.add(digest.lower())
        return next(iter(matches)) if len(matches) == 1 else ""
    except Exception:
        return ""


def _verified_lora_hashes(entries, tags):
    """Only exact-path sidecars / unambiguous preloaded cache; never read weights."""
    enabled_names = {name for name, _ in re.findall(r"<lora:([^:>]+):([^>]+)>", tags)}
    hashes = {}
    try:
        roots = folder_paths.get_folder_paths("loras")
    except Exception:
        roots = []
    scanner = _get_lora_scanner()
    for row in entries:
        if row.get("state") != "known" or row.get("target") != "model":
            continue
        raw = str(row["name"]).replace("\\", "/")
        base = _lora_basename(raw)
        if base not in enabled_names or raw.startswith("/") or ":" in raw or ".." in raw.split("/"):
            continue
        candidates = [raw]
        digests = set()
        cached = _scanner_hash_by_name(scanner, candidates) if scanner else ""
        if cached:
            digests.add(cached)
        for root in roots:
            for relative in (raw + ".metadata.json", os.path.splitext(raw)[0] + ".metadata.json"):
                file = os.path.join(root, *relative.split("/"))
                try:
                    if os.path.getsize(file) > 1024 * 1024:
                        continue
                    with open(file, encoding="utf-8") as stream:
                        data = json.load(stream)
                    digest = str(data.get("sha256") or "") if isinstance(data, dict) else ""
                    if re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                        digests.add(digest.lower())
                except (OSError, ValueError):
                    continue
        hashes.setdefault(base, set()).update(digests)
    selected = [f"{name}: {next(iter(values))[:12]}" for name, values in hashes.items() if len(values) == 1]
    return 'Lora hashes: "' + ", ".join(selected) + '"' if selected else ""


def _lora_hashes_text(loras: str) -> str:
    if not loras:
        return ""
    scanner = _get_lora_scanner()

    hashes: Dict[str, str] = {}
    for lora_name, _strength in re.findall(r"<lora:([^:>]+):([^>]+)>", loras):
        base_name = _lora_basename(lora_name)
        candidates = (
            base_name,
            f"{base_name}.safetensors",
            f"{base_name}.pt",
            f"{base_name}.ckpt",
            lora_name,
        )
        hash_value = _scanner_hash_by_name(scanner, candidates) if scanner is not None else ""
        if not hash_value:
            hash_value = _lora_metadata_hash_by_name(candidates)
        if hash_value:
            hashes[base_name] = hash_value[:10]
    if not hashes:
        return ""
    return f'Lora hashes: "{", ".join(f"{name}: {hash_value}" for name, hash_value in hashes.items())}"'


def _graph_loras(node: Dict[str, Any]) -> list[str]:
    tags = []
    values = _graph_values(node)
    if node.get("type") == "LoraLoader":
        name = _graph_widget_value(node, ("lora_name",))
        name = name if name is not None else (values[0] if values else None)
        strength = _graph_widget_value(node, ("strength_model",))
        strength = strength if strength is not None else (values[1] if len(values) > 1 else 1)
        if name and _as_text(strength) not in {"0", "0.0"}:
            tags.append(_format_lora_tag(name, strength))
    for value in values:
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, dict) or row.get("on") is False:
                continue
            name = row.get("lora") or row.get("name") or row.get("file_name")
            if name:
                tags.append(_format_lora_tag(name, row.get("strength", 1)))
    return tags


def _graph_model_path(sampler, node_map, link_sources) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    node, _ = _graph_source(sampler, "model", node_map, link_sources)
    tags = []
    seen = set()
    while isinstance(node, dict) and node.get("id") not in seen:
        seen.add(node.get("id"))
        node_type = _as_text(node.get("type"))
        if "lora" in node_type.lower():
            tags.extend(_graph_loras(node))
        if node_type in {"UNETLoader", "CheckpointLoaderSimple", "CheckpointLoader"}:
            values = _graph_values(node)
            model = _graph_widget_value(node, ("unet_name", "ckpt_name", "checkpoint", "model_name"))
            if model is None and values:
                model = values[0]
            if model is not None:
                info["model"] = model
            break
        node, _ = _graph_source(node, "model", node_map, link_sources)
    if tags:
        info["loras"] = " ".join(tags)
    return info


def _graph_conditioning_node(node, node_map, link_sources, seen=None):
    if not isinstance(node, dict):
        return None
    seen = set() if seen is None else seen
    if node.get("id") in seen:
        return None
    seen.add(node.get("id"))
    if node.get("type") in {"CLIPTextEncode", "TextEncodeQwenImage21"}:
        return node
    if node.get("type") == "ComfySwitchNode":
        switch = _graph_widget_value(node, ("switch",))
        if switch is None:
            values = _graph_values(node)
            switch = values[0] if values else False
        active = switch is True or _as_text(switch).lower() in {"true", "1"}
        source, _ = _graph_source(node, "on_true" if active else "on_false", node_map, link_sources)
        return _graph_conditioning_node(source, node_map, link_sources, seen)
    for key in ("conditioning", "positive", "negative"):
        source, _ = _graph_source(node, key, node_map, link_sources)
        found = _graph_conditioning_node(source, node_map, link_sources, seen)
        if found is not None:
            return found
    return None


def _graph_dimension(latent, key, index, node_map, link_sources):
    source, slot = _graph_source(latent, key, node_map, link_sources)
    if source is not None and source.get("type") == "ResolutionSelector":
        values = _graph_values(source)
        inputs = {}
        for position, name in enumerate(("aspect_ratio", "megapixels", "multiple")):
            value = _graph_widget_value(source, (name,))
            inputs[name] = value if value is not None else (values[position] if len(values) > position else None)
        dimensions = _resolution_dimensions(inputs)
        return dimensions[slot] if dimensions and slot in (0, 1) else None
    value = _graph_widget_value(latent, (key,))
    values = _graph_values(latent)
    value = value if value is not None else (values[index] if len(values) > index else None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class SaveWebPMeta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "comfy_%model%_%date%"}),
                "file_format": (["webp", "webp_lossless", "png", "jpg", "avif"], {"default": "webp"}),
                "quality": ("INT", {"default": 70, "min": 1, "max": 100}),
            },
            "hidden": {
                "id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_webp"
    OUTPUT_NODE = True
    CATEGORY = "image/save"
    pattern_format = re.compile(r"(%[^%]+%)")

    def _format_filename(self, filename: str, info: Dict[str, Any], prompt=None, extra_pnginfo=None) -> str:
        workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        for segment in re.findall(self.pattern_format, filename):
            raw = segment[1:-1]
            parts = raw.split(":")
            key = parts[0]
            replacement = None

            if "." in key:
                node_name, widget_name = key.rsplit(".", 1)
                replacement = _graph_cross_node_value(workflow, node_name, widget_name)
                if replacement is None:
                    replacement = _prompt_cross_node_value(prompt, node_name, widget_name)
            elif key == "seed" and "seed" in info:
                replacement = info.get("seed")
            elif key == "width":
                replacement = info.get("width")
            elif key == "height":
                replacement = info.get("height")
            elif key == "pprompt" and "prompt" in info:
                text = _sanitize_filename_part(info.get("prompt", ""))
                replacement = _limit_text(text, parts)
            elif key == "nprompt" and "negative_prompt" in info:
                text = _sanitize_filename_part(info.get("negative_prompt", ""))
                replacement = _limit_text(text, parts)
            elif key == "model":
                model = _sanitize_filename_part(_clean_model_name(info.get("model")) or "model_unavailable")
                replacement = _limit_text(model, parts)
            elif key == "date":
                replacement = _format_date_pattern(parts[1] if len(parts) >= 2 else "yyyyMMddhhmmss")

            if replacement is not None:
                filename = filename.replace(segment, _sanitize_filename_part(replacement))
        return filename

    def _metadata_from_prompt(self, prompt: Any, save_id=None) -> Dict[str, Any]:
        return to_legacy_info(resolve_metadata(prompt=prompt, save_id=save_id))

    def _metadata_from_workflow(self, workflow: Any, save_id=None) -> Dict[str, Any]:
        return to_legacy_info(resolve_metadata(workflow=workflow, save_id=save_id))

    def _extract_metadata(self, prompt=None, extra_pnginfo=None, id=None) -> Dict[str, Any]:
        workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        return to_legacy_info(resolve_metadata(prompt=prompt, workflow=workflow, save_id=id))

    def metadata_report(self, prompt=None, extra_pnginfo=None, id=None, image=None, image_index=None):
        """Diagnostic report. It is not embedded in the image or substituted for its workflow."""
        workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        return resolve_metadata(prompt=prompt, workflow=workflow, save_id=id, image=image, image_index=image_index)

    def _build_a1111_parameters(self, info: Dict[str, Any], width: int, height: int) -> str:
        prompt = _as_text(info.get("prompt", "")).strip()
        negative_prompt = _as_text(info.get("negative_prompt", "")).strip()
        loras = _format_loras_from_any(info.get("loras", ""))

        sampler_raw = _as_text(info.get("sampler", "")).strip()
        scheduler_raw = _as_text(info.get("scheduler", "")).strip()
        sampler_display = SAMPLER_MAP.get(sampler_raw, sampler_raw)
        scheduler_display = SCHEDULER_MAP.get(scheduler_raw, scheduler_raw)
        sampler_text = sampler_display
        if scheduler_display:
            sampler_text = f"{sampler_display} {scheduler_display}" if sampler_display else scheduler_display

        lines = []
        if prompt:
            lines.append(prompt)
        if loras:
            lines.append(loras)
        if negative_prompt:
            lines.append(f"Negative prompt: {negative_prompt}")

        params = []
        if "steps" in info:
            params.append(f"Steps: {info['steps']}")
        if sampler_text:
            params.append(f"Sampler: {sampler_text}")
        if "cfg" in info:
            params.append(f"CFG scale: {info['cfg']}")
        if "denoise" in info:
            params.append(f"Denoising strength: {info['denoise']}")
        if "seed" in info:
            params.append(f"Seed: {info['seed']}")
        params.append(f"Size: {width}x{height}")
        model = _clean_model_name(info.get("model"))
        if model:
            params.append(f"Model: {model}")
        if info.get("model_family"):
            params.append(f"Model family: {info['model_family']}")
        if info.get("text_encoder"):
            params.append(f"Text encoder: {info['text_encoder']}")
        if info.get("vae"):
            params.append(f"VAE: {info['vae']}")
        try:
            lora_hashes = _verified_lora_hashes(info["_lora_entries"], loras) if "_lora_entries" in info else _lora_hashes_text(loras)
        except Exception:
            lora_hashes = ""  # Optional hash lookup cannot prevent saving valid pixels.
        if lora_hashes:
            params.append(lora_hashes)
        if "clip_skip" in info:
            params.append(f"Clip skip: {info['clip_skip']}")
        if "rng_source" in info:
            params.append(f"RNG source: {info['rng_source']}")
        if "eta_noise_seed_delta" in info:
            params.append(f"Eta noise seed delta: {info['eta_noise_seed_delta']}")
        if "emphasis_mode" in info:
            params.append(f"Emphasis: {info['emphasis_mode']}")
        if "method" in info:
            params.append(f"Method: {info['method']}")

        if params:
            lines.append(", ".join(params))
        return "\n".join(lines)

    def _build_exif(self, parameters: str, prompt=None, extra_pnginfo=None) -> bytes:
        if piexif is None:
            raise RuntimeError("piexif is required to write EXIF metadata. Install piexif in the active ComfyUI Python environment.")
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}
        items = self._raw_metadata_items(prompt, extra_pnginfo)
        if len(items) > len(EXIF_TEXT_TAGS):
            raise ValueError("EXIF metadata exceeds the 7 text-tag slots. Use PNG; no workflow data was discarded.")
        for exif_tag, (key, value) in zip(EXIF_TEXT_TAGS, items):
            exif_dict["0th"][exif_tag] = _exif_text_bytes(f"{key}:{value}")
        exif_dict["Exif"][EXIF_USER_COMMENT] = piexif.helper.UserComment.dump(parameters, encoding="unicode")
        exif_dict["0th"][EXIF_IMAGE_DESCRIPTION] = _exif_text_bytes(parameters)
        return piexif.dump(exif_dict)

    @staticmethod
    def _raw_metadata_items(prompt, extra_pnginfo):
        items = []
        if prompt is not None:
            items.append(("prompt", json.dumps(prompt, allow_nan=False)))
        if isinstance(extra_pnginfo, dict):
            for key, value in extra_pnginfo.items():
                if key == "parameters" or (key == "prompt" and prompt is not None):
                    raise ValueError("Reserved metadata key collision: " + key)
                if not isinstance(key, str):
                    raise ValueError("Metadata keys must be strings")
                items.append((key, json.dumps(value, allow_nan=False)))
        return items

    def _build_pnginfo(self, parameters: str, prompt=None, extra_pnginfo=None) -> PngInfo:
        metadata = PngInfo()
        if parameters:
            metadata.add_text("parameters", parameters)
        for key, text in self._raw_metadata_items(prompt, extra_pnginfo):
            metadata.add_text(key, text)
        return metadata

    def save_webp(self, images, filename_prefix="ComfyUI", file_format="webp", quality=70, id=None, prompt=None, extra_pnginfo=None, **kwargs):
        if len(images) == 0:
            raise ValueError("Save WEBP Meta requires at least one IMAGE")
        file_format = _as_text(file_format).strip().lower()
        if file_format == "jpeg":
            file_format = "jpg"
        if file_format not in {"webp", "webp_lossless", "png", "jpg", "avif"}:
            file_format = "webp"
        webp_quality = max(1, min(100, _as_int(quality, 70)))
        plans = []
        metadata_cache, prefix_cache, diagnostics = {}, {}, set()
        workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        session = MetadataSession(prompt, workflow)
        # Preflight every metadata payload before writing any file in this batch.
        for i, image_tensor in enumerate(images):
            if len(image_tensor.shape) != 3 or image_tensor.shape[-1] not in (1, 3, 4):
                raise ValueError(f"Unsupported image tensor shape: {image_tensor.shape}")
            height, width, channels = map(int, image_tensor.shape)
            if width <= 0 or height <= 0:
                raise ValueError("IMAGE dimensions must be positive")
            report = session.report(id,
                image={"width": width, "height": height, "channels": channels, "batch_index": i}, image_index=i)
            info = to_legacy_info(report)
            # External/unknown producers still have authoritative pixel dimensions.
            info.update(width=width, height=height)
            diagnostics.update(w["code"] for w in report.warnings)
            signature = json.dumps(info, sort_keys=True, allow_nan=False)
            if signature not in prefix_cache:
                prefix_cache[signature] = self._format_filename(filename_prefix or "ComfyUI", info, prompt=prompt, extra_pnginfo=extra_pnginfo)
            prefix = prefix_cache[signature]
            parameters = self._build_a1111_parameters(info, width, height)
            if parameters not in metadata_cache:
                metadata_cache[parameters] = self._build_pnginfo(parameters, prompt, extra_pnginfo) if file_format == "png" else self._build_exif(parameters, prompt, extra_pnginfo)
            metadata = metadata_cache[parameters]
            if file_format == "jpg" and len(metadata) > 65533:
                raise ValueError("JPEG EXIF exceeds the APP1 limit; use PNG or WebP. No metadata was truncated.")
            plans.append((prefix, metadata, width, height))
        if diagnostics:
            logging.warning("[Save WEBP Meta] Partial metadata (pixels and raw graph retained): %s", ", ".join(sorted(diagnostics)[:8]))
        saved, reserved = [], {}
        for image_tensor, (prefix, metadata, width, height) in zip(images, plans):
            img = _tensor_to_pil(image_tensor)
            out_dir = folder_paths.get_output_directory()
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, out_dir, width, height)
            key = (full_output_folder, filename)
            counter = max(counter, reserved.get(key, counter))
            reserved[key] = counter + 1
            os.makedirs(full_output_folder, exist_ok=True)
            extension = "webp" if file_format == "webp_lossless" else file_format
            file = f"{filename}_{counter:05}_.{extension}"
            path = os.path.join(full_output_folder, file)
            if file_format == "png":
                img.save(path, format="PNG", pnginfo=metadata, compress_level=_png_compress_level_from_quality(quality))
            elif file_format == "jpg":
                img.convert("RGB").save(path, format="JPEG", quality=webp_quality, exif=metadata)
            elif file_format == "avif":
                img.save(path, format="AVIF", quality=webp_quality, exif=metadata)
            elif file_format == "webp_lossless":
                img.save(path, format="WEBP", lossless=True, exact=True, quality=webp_quality, method=6, exif=metadata)
            else:
                img.save(path, format="WEBP", quality=webp_quality, lossless=False, method=0, exif=metadata)
            saved.append({"filename": file, "subfolder": subfolder, "type": "output"})
        return {"ui": {"images": saved}}
