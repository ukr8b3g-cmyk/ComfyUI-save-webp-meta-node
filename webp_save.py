import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, Optional

import folder_paths
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
    return os.path.splitext(os.path.basename(text))[0]


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


def _node_inputs(node: Dict[str, Any]) -> Dict[str, Any]:
    return node.get("inputs") if isinstance(node.get("inputs"), dict) else {}


def _first_node(prompt: Any, class_names: Iterable[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    names = set(class_names)
    for node in prompt.values():
        if isinstance(node, dict) and node.get("class_type") in names:
            return node
    return None


def _prompt_ref(value: Any) -> Optional[tuple[str, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    node_id = _as_text(value[0]).strip()
    if not node_id:
        return None
    try:
        output_slot = int(value[1])
    except Exception:
        return None
    return node_id, output_slot


def _prompt_linked_node(prompt: Any, value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    ref = _prompt_ref(value)
    if ref is None:
        return None
    node = prompt.get(ref[0])
    return node if isinstance(node, dict) else None


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


def _prompt_resolve_input(
    prompt: Any,
    value: Any,
    negative: bool,
    seen: set[tuple[str, int, bool]],
) -> tuple[bool, str]:
    if isinstance(value, str):
        return True, value
    ref = _prompt_ref(value)
    if ref is None:
        return False, ""
    return _prompt_resolve_ref(prompt, value, negative=negative, seen=seen)


def _prompt_resolve_ref(
    prompt: Any,
    value: Any,
    negative: bool,
    seen: Optional[set[tuple[str, int, bool]]] = None,
) -> tuple[bool, str]:
    if not isinstance(prompt, dict):
        return False, ""
    ref = _prompt_ref(value)
    if ref is None:
        return False, ""
    marker = (ref[0], ref[1], bool(negative))
    if seen is None:
        seen = set()
    if marker in seen:
        return False, ""
    next_seen = set(seen)
    next_seen.add(marker)
    node = prompt.get(ref[0])
    if not isinstance(node, dict):
        return False, ""
    return _prompt_resolve_node(
        prompt,
        node,
        output_slot=ref[1],
        negative=negative,
        seen=next_seen,
    )


def _prompt_resolve_node(
    prompt: Any,
    node: Dict[str, Any],
    output_slot: int,
    negative: bool,
    seen: set[tuple[str, int, bool]],
) -> tuple[bool, str]:
    inputs = _node_inputs(node)
    class_type = _as_text(node.get("class_type"))

    if class_type in ANIMA_REGIONAL_TYPES:
        return True, _anima_prompt_from_inputs(inputs, negative=negative)

    if class_type == "ConditioningZeroOut":
        return True, ""

    if class_type == "ComfySwitchNode":
        active = inputs.get("on_true") if _prompt_switch_enabled(inputs.get("switch")) else inputs.get("on_false")
        resolved = _prompt_resolve_input(prompt, active, negative=negative, seen=seen)
        if resolved[0]:
            return resolved
        fallback = inputs.get("on_false") if _prompt_switch_enabled(inputs.get("switch")) else inputs.get("on_true")
        return _prompt_resolve_input(prompt, fallback, negative=negative, seen=seen)

    has_prompt = "prompt" in inputs
    has_negative_prompt = "negative_prompt" in inputs
    if has_prompt and has_negative_prompt:
        input_name = "negative_prompt" if output_slot == 1 else "prompt"
        return _prompt_resolve_input(prompt, inputs.get(input_name), negative=negative, seen=seen)

    if negative and has_negative_prompt:
        resolved = _prompt_resolve_input(prompt, inputs.get("negative_prompt"), negative=True, seen=seen)
        if resolved[0]:
            return resolved
    if not negative and has_prompt:
        resolved = _prompt_resolve_input(prompt, inputs.get("prompt"), negative=False, seen=seen)
        if resolved[0]:
            return resolved

    if class_type == "StringConcatenate":
        parts = []
        delimiter = inputs.get("delimiter", "")
        if not isinstance(delimiter, str):
            delimiter = ""
        found = False
        for input_name in ("string_a", "string_b"):
            resolved, text = _prompt_resolve_input(
                prompt,
                inputs.get(input_name),
                negative=negative,
                seen=seen,
            )
            if resolved:
                found = True
            if text.strip():
                parts.append(text)
        return (found, delimiter.join(parts)) if found else (False, "")

    if "text" in inputs:
        resolved = _prompt_resolve_input(prompt, inputs.get("text"), negative=negative, seen=seen)
        if resolved[0]:
            return resolved

    candidates = (
        ("negative", "prompt", "value", "string")
        if negative
        else ("positive", "prompt", "value", "string")
    )
    for input_name in candidates:
        if input_name not in inputs:
            continue
        resolved = _prompt_resolve_input(prompt, inputs.get(input_name), negative=negative, seen=seen)
        if resolved[0]:
            return resolved

    for input_name, value in inputs.items():
        if not re.search(r"(conditioning|positive|negative|prompt|text|string|value)", str(input_name), re.I):
            continue
        resolved = _prompt_resolve_input(prompt, value, negative=negative, seen=seen)
        if resolved[0]:
            return resolved
    return False, ""


def _prompt_resolve_text(
    prompt: Any,
    node: Optional[Dict[str, Any]],
    seen: Optional[set[int]] = None,
) -> str:
    # Legacy wrapper retained for callers that only need text.
    if not isinstance(node, dict):
        return ""
    resolved, text = _prompt_resolve_node(
        prompt,
        node,
        output_slot=0,
        negative=False,
        seen=set(),
    )
    return text if resolved else ""


def _prompt_text_node_result(prompt: Any, negative: bool) -> tuple[bool, str]:
    sampler = _first_node(prompt, ("KSampler", "KSamplerAdvanced"))
    if sampler:
        inputs = _node_inputs(sampler)
        value = inputs.get("negative" if negative else "positive")
        resolved, text = _prompt_resolve_ref(prompt, value, negative=negative)
        if resolved:
            return True, text

    for node in prompt.values() if isinstance(prompt, dict) else []:
        if not isinstance(node, dict) or node.get("class_type") != "CLIPTextEncode":
            continue
        title = _prompt_node_title(node)
        is_negative = "negative" in title.lower()
        if is_negative != negative:
            continue
        resolved, text = _prompt_resolve_node(
            prompt,
            node,
            output_slot=0,
            negative=negative,
            seen=set(),
        )
        if resolved:
            return True, text
    return False, ""


def _prompt_text_node(prompt: Any, negative: bool) -> str:
    return _prompt_text_node_result(prompt, negative=negative)[1]


def _prompt_switch_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = _as_text(value).strip().lower()
    if text in {"true", "on", "yes", "1"}:
        return True
    if text in {"false", "off", "no", "0", ""}:
        return False
    return bool(value)


def _prompt_find_upstream(
    prompt: Any,
    value: Any,
    predicate,
    seen: Optional[set[tuple[str, int]]] = None,
) -> Optional[tuple[str, Dict[str, Any], int]]:
    if not isinstance(prompt, dict):
        return None
    ref = _prompt_ref(value)
    if ref is None:
        return None
    if seen is None:
        seen = set()
    marker = (ref[0], ref[1])
    if marker in seen:
        return None
    next_seen = set(seen)
    next_seen.add(marker)

    node = prompt.get(ref[0])
    if not isinstance(node, dict):
        return None
    if predicate(node):
        return ref[0], node, ref[1]

    inputs = _node_inputs(node)
    class_type = _as_text(node.get("class_type"))
    if class_type == "ComfySwitchNode":
        active = inputs.get("on_true") if _prompt_switch_enabled(inputs.get("switch")) else inputs.get("on_false")
        hit = _prompt_find_upstream(prompt, active, predicate, next_seen)
        if hit is not None:
            return hit

    preferred = (
        "images",
        "image",
        "samples",
        "latent",
        "latent_image",
        "conditioning",
        "positive",
        "negative",
        "model",
        "clip",
        "vae",
    )
    for input_name in preferred:
        if input_name not in inputs:
            continue
        hit = _prompt_find_upstream(prompt, inputs.get(input_name), predicate, next_seen)
        if hit is not None:
            return hit
    for input_value in inputs.values():
        if _prompt_ref(input_value) is None:
            continue
        hit = _prompt_find_upstream(prompt, input_value, predicate, next_seen)
        if hit is not None:
            return hit
    return None


def _prompt_active_sampler(prompt: Any, save_node_id: Any = None) -> Optional[tuple[str, Dict[str, Any]]]:
    if not isinstance(prompt, dict):
        return None
    candidate_ids = []
    if save_node_id is not None:
        candidate_ids.append(_as_text(save_node_id))
    candidate_ids.extend(
        node_id
        for node_id, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type") == "SaveWebPMeta"
    )
    seen_ids = set()
    for node_id in candidate_ids:
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        save_node = prompt.get(node_id)
        if not isinstance(save_node, dict):
            continue
        hit = _prompt_find_upstream(
            prompt,
            _node_inputs(save_node).get("images"),
            lambda node: node.get("class_type") in {"KSampler", "KSamplerAdvanced"},
        )
        if hit is not None:
            return hit[0], hit[1]

    for node_id, node in prompt.items():
        if isinstance(node, dict) and node.get("class_type") in {"KSampler", "KSamplerAdvanced"}:
            return _as_text(node_id), node
    return None


def _prompt_resolution_selector_value(node: Dict[str, Any], output_slot: int) -> Optional[int]:
    inputs = _node_inputs(node)
    aspect = _as_text(inputs.get("aspect_ratio")).strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", aspect)
    try:
        megapixels = float(inputs.get("megapixels"))
    except Exception:
        megapixels = 0.0
    multiple = max(1, _as_int(inputs.get("multiple"), 1))
    if not match or megapixels <= 0:
        return None
    ratio = float(match.group(1)) / float(match.group(2))
    pixels = megapixels * 1024 * 1024
    raw_width = (pixels * ratio) ** 0.5
    raw_height = (pixels / ratio) ** 0.5
    width = max(multiple, round(raw_width / multiple) * multiple)
    height = max(multiple, round(raw_height / multiple) * multiple)
    return int(height if output_slot == 1 else width)


def _prompt_resolve_scalar(
    prompt: Any,
    value: Any,
    seen: Optional[set[tuple[str, int]]] = None,
) -> Any:
    if value is not None and not isinstance(value, (list, tuple, dict)):
        return value
    ref = _prompt_ref(value)
    if ref is None or not isinstance(prompt, dict):
        return None
    if seen is None:
        seen = set()
    marker = (ref[0], ref[1])
    if marker in seen:
        return None
    next_seen = set(seen)
    next_seen.add(marker)
    node = prompt.get(ref[0])
    if not isinstance(node, dict):
        return None
    inputs = _node_inputs(node)
    class_type = _as_text(node.get("class_type"))

    if class_type == "ResolutionSelector":
        return _prompt_resolution_selector_value(node, ref[1])

    if class_type == "ComfySwitchNode":
        active = inputs.get("on_true") if _prompt_switch_enabled(inputs.get("switch")) else inputs.get("on_false")
        resolved = _prompt_resolve_scalar(prompt, active, next_seen)
        if resolved is not None:
            return resolved

    output_names = {
        0: ("value", "width", "seed", "steps", "cfg", "int", "number"),
        1: ("height", "value", "int", "number"),
    }
    for input_name in output_names.get(ref[1], ("value", "int", "number")):
        if input_name not in inputs:
            continue
        resolved = _prompt_resolve_scalar(prompt, inputs.get(input_name), next_seen)
        if resolved is not None:
            return resolved
    return None


def _prompt_active_vae_node(
    prompt: Any,
    save_node_id: Any,
    sampler_id: str,
    sampler: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    save_node = prompt.get(_as_text(save_node_id)) if isinstance(prompt, dict) and save_node_id is not None else None
    if isinstance(save_node, dict):
        decode_hit = _prompt_find_upstream(
            prompt,
            _node_inputs(save_node).get("images"),
            lambda node: node.get("class_type") == "VAEDecode",
        )
        if decode_hit is not None:
            vae_hit = _prompt_find_upstream(
                prompt,
                _node_inputs(decode_hit[1]).get("vae"),
                lambda node: node.get("class_type") == "VAELoader",
            )
            if vae_hit is not None:
                return vae_hit[1]

    positive_hit = _prompt_find_upstream(
        prompt,
        _node_inputs(sampler).get("positive"),
        lambda node: node.get("class_type") == "TextEncodeQwenImage21",
    )
    if positive_hit is not None:
        vae_hit = _prompt_find_upstream(
            prompt,
            _node_inputs(positive_hit[1]).get("vae"),
            lambda node: node.get("class_type") == "VAELoader",
        )
        if vae_hit is not None:
            return vae_hit[1]
    return None


def _prompt_active_model_node(prompt: Any, sampler: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hit = _prompt_find_upstream(
        prompt,
        _node_inputs(sampler).get("model"),
        lambda node: node.get("class_type") in {"UNETLoader", "CheckpointLoaderSimple", "CheckpointLoader"},
    )
    return hit[1] if hit is not None else None


def _prompt_active_encoder_node(prompt: Any, sampler: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hit = _prompt_find_upstream(
        prompt,
        _node_inputs(sampler).get("positive"),
        lambda node: node.get("class_type") in {"TextEncodeQwenImage21", "CLIPTextEncode"} or node.get("class_type") in ANIMA_REGIONAL_TYPES,
    )
    return hit[1] if hit is not None else None


def _prompt_active_text_encoder_name(prompt: Any, encoder: Optional[Dict[str, Any]]) -> str:
    if not isinstance(encoder, dict):
        return ""
    hit = _prompt_find_upstream(
        prompt,
        _node_inputs(encoder).get("clip"),
        lambda node: node.get("class_type") in {"CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader"},
    )
    if hit is None:
        return ""
    inputs = _node_inputs(hit[1])
    for key in ("clip_name", "clip_name1", "clip_name_1", "text_encoder_name"):
        if inputs.get(key):
            return _as_text(inputs.get(key))
    return ""


def _prompt_collect_active_loras(
    prompt: Any,
    value: Any,
    seen: Optional[set[tuple[str, int]]] = None,
) -> str:
    ref = _prompt_ref(value)
    if ref is None or not isinstance(prompt, dict):
        return ""
    if seen is None:
        seen = set()
    marker = (ref[0], ref[1])
    if marker in seen:
        return ""
    next_seen = set(seen)
    next_seen.add(marker)
    node = prompt.get(ref[0])
    if not isinstance(node, dict):
        return ""
    inputs = _node_inputs(node)
    class_type = _as_text(node.get("class_type"))
    tags = []

    if "lora" in class_type.lower():
        for input_name, item in inputs.items():
            if not str(input_name).lower().startswith("lora"):
                continue
            if isinstance(item, dict):
                if item.get("on") is False:
                    continue
                name = item.get("lora") or item.get("name") or item.get("file_name")
                strength = item.get("strength", item.get("weight", item.get("multiplier", 1.0)))
                if name:
                    tags.append(_format_lora_tag(name, strength))
        lora_name = inputs.get("lora_name")
        if isinstance(lora_name, str) and lora_name.strip():
            strength = inputs.get("strength_model", inputs.get("strength", 1.0))
            try:
                active = float(strength) != 0.0
            except Exception:
                active = True
            if active:
                tags.append(_format_lora_tag(lora_name, strength))

    if class_type == "ComfySwitchNode":
        active_ref = inputs.get("on_true") if _prompt_switch_enabled(inputs.get("switch")) else inputs.get("on_false")
        nested = _prompt_collect_active_loras(prompt, active_ref, next_seen)
        if nested:
            tags.append(nested)
    elif _prompt_ref(inputs.get("model")) is not None:
        nested = _prompt_collect_active_loras(prompt, inputs.get("model"), next_seen)
        if nested:
            tags.append(nested)

    return " ".join(dict.fromkeys(tag for tag in tags if tag))


def _prompt_model_family(prompt: Any, model_name: str) -> str:
    lower_model = _as_text(model_name).lower()
    for node in prompt.values() if isinstance(prompt, dict) else []:
        if not isinstance(node, dict):
            continue
        class_type = _as_text(node.get("class_type"))
        inputs = _node_inputs(node)
        if class_type == "TextEncodeQwenImage21":
            return "Qwen Image 2.1"
        if class_type == "CLIPLoader" and _as_text(inputs.get("type")).lower() == "krea2":
            return "Krea2"
    if "qwen_image_2.1" in lower_model or "qwen-image-2.1" in lower_model:
        return "Qwen Image 2.1"
    if "krea2" in lower_model or "kres2" in lower_model:
        return "Krea2"
    return ""


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


def _graph_link_sources(workflow: Any) -> Dict[int, int]:
    if not isinstance(workflow, dict):
        return {}
    sources: Dict[int, int] = {}
    links = workflow.get("links")
    if not isinstance(links, list):
        return sources
    for link in links:
        if isinstance(link, list) and len(link) >= 2 and isinstance(link[0], int) and isinstance(link[1], int):
            sources[link[0]] = link[1]
    return sources


def _graph_node_map(nodes: list[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {node["id"]: node for node in nodes if isinstance(node.get("id"), int)}


def _graph_resolve_string(node: Optional[Dict[str, Any]], node_map: Dict[int, Dict[str, Any]], link_sources: Dict[int, int], seen: Optional[set[int]] = None) -> str:
    if not isinstance(node, dict) or not isinstance(node.get("id"), int):
        return ""
    seen = seen or set()
    node_id = node["id"]
    if node_id in seen:
        return ""
    seen.add(node_id)

    node_type = node.get("type")
    values = _graph_values(node)
    if node_type in {"PrimitiveStringMultiline", "PrimitiveString"}:
        return values[0] if values and isinstance(values[0], str) else ""
    if node_type == "StringConcatenate":
        parts = []
        delimiter = values[2] if len(values) >= 3 and isinstance(values[2], str) else ""
        for idx, input_name in enumerate(("string_a", "string_b")):
            text = ""
            link = _graph_input_link(node, input_name)
            if link in link_sources:
                text = _graph_resolve_string(node_map.get(link_sources[link]), node_map, link_sources, seen)
            elif len(values) > idx and isinstance(values[idx], str):
                text = values[idx]
            if text:
                parts.append(text)
        return delimiter.join(parts)
    if node_type == "CLIPTextEncode":
        link = _graph_input_link(node, "text")
        if link in link_sources:
            return _graph_resolve_string(node_map.get(link_sources[link]), node_map, link_sources, seen)
        return values[0] if values and isinstance(values[0], str) else ""
    return values[0] if values and isinstance(values[0], str) else ""


def _graph_text_node(workflow: Any, nodes: list[Dict[str, Any]], negative: bool) -> str:
    node_map = _graph_node_map(nodes)
    link_sources = _graph_link_sources(workflow)
    for node in nodes:
        if node.get("type") in ANIMA_REGIONAL_TYPES:
            text = _graph_anima_prompt(node, negative=negative)
            if text.strip():
                return text

    # Prefer titled negative CLIP nodes for negative prompt, and non-negative CLIP nodes for prompt.
    for node in nodes:
        if node.get("type") != "CLIPTextEncode":
            continue
        title = _as_text(node.get("title") or node.get("properties", {}).get("Node name for S&R"))
        is_negative = "negative" in title.lower()
        if is_negative != negative:
            continue
        text = _graph_resolve_string(node, node_map, link_sources)
        if text.strip():
            return text
    # Fallback to primitive/string nodes for positive prompt only.
    if not negative:
        for node in nodes:
            if node.get("type") in {"PrimitiveStringMultiline", "PrimitiveString", "StringConcatenate"}:
                text = _graph_resolve_string(node, node_map, link_sources)
                if text.strip():
                    return text
    return ""


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
        strength_text = f"{float(strength):.1f}"
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
            scanner = registry.get_service_sync("lora_scanner")
        except Exception:
            scanner = None
        if scanner is not None and hasattr(scanner, "get_hash_by_filename"):
            return scanner
    return None


def _lora_metadata_hash_by_name(candidates: Iterable[str]) -> str:
    try:
        lora_dirs = folder_paths.get_folder_paths("loras")
    except Exception:
        lora_dirs = []

    candidate_bases = {_lora_basename(candidate) for candidate in candidates if candidate}
    candidate_files = set()
    for base in candidate_bases:
        candidate_files.add(f"{base}.metadata.json")
        candidate_files.add(f"{base}.safetensors.metadata.json")

    for lora_dir in lora_dirs:
        for root, _dirs, files in os.walk(lora_dir):
            for file_name in files:
                if file_name not in candidate_files:
                    continue
                metadata_path = os.path.join(root, file_name)
                try:
                    with open(metadata_path, "r", encoding="utf-8") as file_obj:
                        metadata = json.load(file_obj)
                except Exception:
                    continue
                file_base = _lora_basename(metadata.get("file_name"))
                path_base = _lora_basename(metadata.get("file_path"))
                metadata_base = file_name.removesuffix(".metadata.json")
                metadata_base = _lora_basename(metadata_base.removesuffix(".safetensors"))
                if file_base not in candidate_bases and path_base not in candidate_bases and metadata_base not in candidate_bases:
                    continue
                hash_value = metadata.get("sha256")
                if hash_value:
                    return str(hash_value)
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
    for candidate in candidates:
        try:
            hash_value = scanner.get_hash_by_filename(candidate)
        except Exception:
            hash_value = None
        if hash_value:
            return str(hash_value)

    cache = getattr(scanner, "_cache", None)
    for item in getattr(cache, "raw_data", []) or []:
        if not isinstance(item, dict):
            continue
        known_names = {
            _as_text(item.get("file_name")),
            _as_text(item.get("name")),
            _as_text(item.get("model_name")),
            _as_text(os.path.basename(_as_text(item.get("file_path") or item.get("path")))),
        }
        known_bases = {_lora_basename(name) for name in known_names if name}
        for candidate in candidates:
            if candidate in known_names or _lora_basename(candidate) in known_bases:
                hash_value = item.get("sha256") or item.get("hash")
                if hash_value:
                    return str(hash_value)
    return ""


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
        if not hash_value:
            hash_value = _lora_file_hash_by_name(candidates)
        if hash_value:
            hashes[base_name] = hash_value[:10]
    if not hashes:
        return ""
    return f'Lora hashes: "{", ".join(f"{name}: {hash_value}" for name, hash_value in hashes.items())}"'


def _extract_graph_loras(nodes: list[Dict[str, Any]]) -> str:
    tags = []
    for node in nodes:
        node_type = _as_text(node.get("type")).lower()
        if "lora" not in node_type:
            continue
        values = _graph_values(node)
        # rgthree Power Lora Loader stores lora rows as dicts in widgets_values.
        for value in values:
            if isinstance(value, dict):
                if value.get("on") is False:
                    continue
                name = value.get("lora") or value.get("name") or value.get("file_name")
                strength = value.get("strength", value.get("weight", value.get("multiplier", 1.0)))
                if name:
                    tags.append(_format_lora_tag(name, strength))
            elif isinstance(value, str) and "<lora:" in value:
                formatted = _format_loras_from_any(value)
                if formatted:
                    tags.append(formatted)
    return " ".join(tags)


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

    def _metadata_from_prompt(self, prompt: Any, id=None) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if not isinstance(prompt, dict):
            return info

        active = _prompt_active_sampler(prompt, save_node_id=id)
        sampler_id = active[0] if active else ""
        ksampler = active[1] if active else _first_node(prompt, ("KSampler", "KSamplerAdvanced"))
        if ksampler:
            inputs = _node_inputs(ksampler)
            for src, dst in (
                ("seed", "seed"),
                ("noise_seed", "seed"),
                ("steps", "steps"),
                ("cfg", "cfg"),
                ("sampler_name", "sampler"),
                ("sampler", "sampler"),
                ("scheduler", "scheduler"),
                ("denoise", "denoise"),
            ):
                if src in inputs and inputs[src] is not None:
                    info[dst] = inputs[src]

            latent_hit = _prompt_find_upstream(
                prompt,
                inputs.get("latent_image"),
                lambda node: node.get("class_type") in {"EmptyLatentImage", "EmptySD3LatentImage"},
            )
            if latent_hit is not None:
                latent_inputs = _node_inputs(latent_hit[1])
                width = _prompt_resolve_scalar(prompt, latent_inputs.get("width"))
                height = _prompt_resolve_scalar(prompt, latent_inputs.get("height"))
                if width is not None:
                    info["width"] = width
                if height is not None:
                    info["height"] = height

            model_node = _prompt_active_model_node(prompt, ksampler)
            if model_node is not None:
                model_inputs = _node_inputs(model_node)
                for key in ("ckpt_name", "checkpoint", "unet_name", "model_name"):
                    if model_inputs.get(key):
                        info["model"] = model_inputs[key]
                        break

            encoder = _prompt_active_encoder_node(prompt, ksampler)
            text_encoder = _prompt_active_text_encoder_name(prompt, encoder)
            if text_encoder:
                info["text_encoder"] = text_encoder

            vae_node = _prompt_active_vae_node(prompt, id, sampler_id, ksampler)
            if vae_node is not None:
                vae_name = _node_inputs(vae_node).get("vae_name")
                if vae_name:
                    info["vae"] = vae_name

            loras = _prompt_collect_active_loras(prompt, inputs.get("model"))
            if loras:
                info["loras"] = loras

            model_family = _prompt_model_family(prompt, _as_text(info.get("model")))
            if model_family:
                info["model_family"] = model_family

        if "width" not in info or "height" not in info:
            latent = _first_node(prompt, ("EmptyLatentImage", "EmptySD3LatentImage"))
            if latent:
                inputs = _node_inputs(latent)
                width = _prompt_resolve_scalar(prompt, inputs.get("width"))
                height = _prompt_resolve_scalar(prompt, inputs.get("height"))
                if width is not None:
                    info.setdefault("width", width)
                if height is not None:
                    info.setdefault("height", height)
        if "width" not in info or "height" not in info:
            anima = _first_node(prompt, ANIMA_REGIONAL_TYPES)
            if anima:
                inputs = _node_inputs(anima)
                if "width" in inputs:
                    info.setdefault("width", inputs["width"])
                if "height" in inputs:
                    info.setdefault("height", inputs["height"])

        if "model" not in info:
            ckpt = _first_node(prompt, ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader"))
            if ckpt:
                inputs = _node_inputs(ckpt)
                for key in ("ckpt_name", "checkpoint", "unet_name", "model_name"):
                    if key in inputs:
                        info["model"] = inputs[key]
                        break

        clip_skip = _first_node(prompt, ("CLIPSetLastLayer", "CLIPSkip"))
        if clip_skip:
            inputs = _node_inputs(clip_skip)
            for key in ("stop_at_clip_layer", "clip_skip", "clip_layer"):
                if key in inputs:
                    info["clip_skip"] = _clip_skip_value(inputs[key])
                    break

        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            inputs = _node_inputs(node)
            for key in ("rng_source", "random_generator_source", "random_number_generator_source", "noise_device"):
                if key in inputs:
                    info["rng_source"] = inputs[key]
                    break
            if "rng_source" in info:
                break

        if "clip_skip" not in info:
            clip_skip_value = _find_named_value(prompt, ("stop_at_clip_layer", "clip_skip", "clip_layer"))
            if clip_skip_value is not None:
                info["clip_skip"] = _clip_skip_value(clip_skip_value)
        if "rng_source" not in info:
            rng_source = _find_named_value(
                prompt,
                ("rng_source", "random_generator_source", "random_number_generator_source", "noise_device"),
            )
            if rng_source is not None:
                info["rng_source"] = rng_source

        eta_noise_seed_delta = _find_named_value(
            prompt,
            ("eta_noise_seed_delta", "eta_noise_seed", "noise_seed_delta", "ensd"),
        )
        if eta_noise_seed_delta is not None:
            info["eta_noise_seed_delta"] = eta_noise_seed_delta
        emphasis_mode = _find_named_value(
            prompt,
            ("emphasis_mode", "emphasis", "emphasisMode", "prompt_emphasis", "prompt_parser"),
        )
        if emphasis_mode is not None:
            info["emphasis_mode"] = emphasis_mode

        positive_resolved, positive = _prompt_text_node_result(prompt, negative=False)
        negative_resolved, negative = _prompt_text_node_result(prompt, negative=True)
        if positive_resolved:
            info["prompt"] = positive
        if negative_resolved:
            info["negative_prompt"] = negative

        return info

    def _metadata_from_workflow(self, workflow: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        nodes = _graph_nodes(workflow)
        if not nodes:
            return info

        ksampler = _graph_first(nodes, ("KSampler", "KSamplerAdvanced"))
        values = _graph_values(ksampler)
        # Common KSampler widget order: seed, control_after_generate, steps, cfg, sampler, scheduler, denoise
        if len(values) >= 1:
            info["seed"] = values[0]
        if len(values) >= 3:
            info["steps"] = values[2]
        if len(values) >= 4:
            info["cfg"] = values[3]
        if len(values) >= 5:
            info["sampler"] = values[4]
        if len(values) >= 6:
            info["scheduler"] = values[5]

        latent = _graph_first(nodes, ("EmptyLatentImage", "EmptySD3LatentImage"))
        values = _graph_values(latent)
        if len(values) >= 1:
            info["width"] = values[0]
        if len(values) >= 2:
            info["height"] = values[1]
        if "width" not in info or "height" not in info:
            anima = _graph_first(nodes, ANIMA_REGIONAL_TYPES)
            values = _graph_values(anima)
            if len(values) >= 1:
                info["width"] = values[0]
            if len(values) >= 2:
                info["height"] = values[1]

        ckpt = _graph_first(nodes, ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader"))
        values = _graph_values(ckpt)
        if values:
            info["model"] = values[0]

        clip_skip_node = _graph_first(nodes, ("CLIPSetLastLayer", "CLIPSkip"))
        clip_skip = _graph_widget_value(clip_skip_node, ("stop_at_clip_layer", "clip_skip", "clip_layer"))
        if clip_skip is None:
            values = _graph_values(clip_skip_node)
            if values:
                clip_skip = values[0]
        if clip_skip is None:
            for node in nodes:
                node_type = _as_text(node.get("type")).lower()
                if "clip" not in node_type or ("skip" not in node_type and "lastlayer" not in node_type and "last_layer" not in node_type):
                    continue
                values = _graph_values(node)
                if values:
                    clip_skip = values[0]
                    break
        if clip_skip is None:
            clip_skip = _find_named_value(workflow, ("stop_at_clip_layer", "clip_skip", "clip_layer"))
        if clip_skip is not None:
            info["clip_skip"] = _clip_skip_value(clip_skip)

        rng_source = _find_named_value(
            workflow,
            ("rng_source", "random_generator_source", "random_number_generator_source", "noise_device"),
        )
        if rng_source is None:
            for node in nodes:
                rng_source = _graph_widget_value(
                    node,
                    ("rng_source", "random_generator_source", "random_number_generator_source", "noise_device"),
                )
                if rng_source is not None:
                    break
        if rng_source is None:
            for node in nodes:
                for value in _graph_values(node):
                    if _looks_like_rng_source(value):
                        rng_source = value
                        break
                if rng_source is not None:
                    break
        if rng_source is not None:
            info["rng_source"] = rng_source

        eta_noise_seed_delta = _find_named_value(
            workflow,
            ("eta_noise_seed_delta", "eta_noise_seed", "noise_seed_delta", "ensd"),
        )
        if eta_noise_seed_delta is None:
            for node in nodes:
                eta_noise_seed_delta = _graph_widget_value(
                    node,
                    ("eta_noise_seed_delta", "eta_noise_seed", "noise_seed_delta", "ensd"),
                )
                if eta_noise_seed_delta is not None:
                    break
        if eta_noise_seed_delta is not None:
            info["eta_noise_seed_delta"] = eta_noise_seed_delta

        emphasis_mode = _find_named_value(
            workflow,
            ("emphasis_mode", "emphasis", "emphasisMode", "prompt_emphasis", "prompt_parser"),
        )
        if emphasis_mode is None:
            for node in nodes:
                emphasis_mode = _graph_widget_value(
                    node,
                    ("emphasis_mode", "emphasis", "emphasisMode", "prompt_emphasis", "prompt_parser"),
                )
                if emphasis_mode is not None:
                    break
        if emphasis_mode is not None:
            info["emphasis_mode"] = emphasis_mode

        positive = _graph_text_node(workflow, nodes, negative=False)
        negative = _graph_text_node(workflow, nodes, negative=True)
        if positive:
            info["prompt"] = positive
        if negative:
            info["negative_prompt"] = negative

        loras = _extract_graph_loras(nodes)
        if loras:
            info["loras"] = loras
        return info

    def _extract_metadata(self, prompt=None, extra_pnginfo=None, id=None) -> Dict[str, Any]:
        # SaveImageLM-style: accept hidden id/prompt/extra_pnginfo, but only use structured data.
        # Never use id as metadata text; it is only a runtime key.
        info: Dict[str, Any] = {}
        if isinstance(extra_pnginfo, dict):
            info.update(self._metadata_from_workflow(extra_pnginfo.get("workflow")))
        info.update(self._metadata_from_prompt(prompt, id=id))
        return info

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
        if "seed" in info:
            params.append(f"Seed: {info['seed']}")
        params.append(f"Size: {width}x{height}")
        model = _clean_model_name(info.get("model"))
        if model:
            params.append(f"Model: {model}")
        lora_hashes = _lora_hashes_text(loras)
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
            raise RuntimeError(
                "piexif is required to write WebP EXIF metadata. "
                "Install it in the active ComfyUI Python environment with: python -m pip install piexif"
            )
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}
        metadata_items = []
        if prompt is not None:
            metadata_items.append(("prompt", prompt))
        if isinstance(extra_pnginfo, dict):
            metadata_items.extend(extra_pnginfo.items())
        for exif_tag, (key, value) in zip(EXIF_TEXT_TAGS, metadata_items):
            exif_dict["0th"][exif_tag] = _exif_text_bytes(f"{key}:{json.dumps(value)}")
        exif_dict["Exif"][EXIF_USER_COMMENT] = piexif.helper.UserComment.dump(parameters, encoding="unicode")
        exif_dict["0th"][EXIF_IMAGE_DESCRIPTION] = _exif_text_bytes(parameters)
        return piexif.dump(exif_dict)

    def _build_pnginfo(self, parameters: str, prompt=None, extra_pnginfo=None) -> PngInfo:
        metadata = PngInfo()
        if parameters:
            metadata.add_text("parameters", parameters)
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if isinstance(extra_pnginfo, dict):
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value))
        return metadata

    def save_webp(self, images, filename_prefix="ComfyUI", file_format="webp", quality=70, id=None, prompt=None, extra_pnginfo=None, **kwargs):
        info = self._extract_metadata(prompt=prompt, extra_pnginfo=extra_pnginfo, id=id)
        filename_prefix = self._format_filename(filename_prefix or "ComfyUI", info, prompt=prompt, extra_pnginfo=extra_pnginfo)
        file_format = _as_text(file_format).strip().lower()
        if file_format == "jpeg":
            file_format = "jpg"
        if file_format not in {"webp", "webp_lossless", "png", "jpg", "avif"}:
            file_format = "webp"

        out_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            out_dir,
            images[0].shape[1],
            images[0].shape[0],
        )
        os.makedirs(full_output_folder, exist_ok=True)

        width = _as_int(info.get("width", images[0].shape[1]), images[0].shape[1])
        height = _as_int(info.get("height", images[0].shape[0]), images[0].shape[0])
        parameters = self._build_a1111_parameters(info, width, height)
        webp_quality = max(1, min(100, _as_int(quality, 70)))

        saved = []
        for i, image_tensor in enumerate(images):
            img = _tensor_to_pil(image_tensor)
            extension = "webp" if file_format == "webp_lossless" else file_format
            file = f"{filename}_{counter + i:05}_.{extension}"
            if file_format == "png":
                img.save(
                    os.path.join(full_output_folder, file),
                    format="PNG",
                    pnginfo=self._build_pnginfo(parameters, prompt=prompt, extra_pnginfo=extra_pnginfo),
                    compress_level=_png_compress_level_from_quality(quality),
                )
            elif file_format == "jpg":
                img.convert("RGB").save(
                    os.path.join(full_output_folder, file),
                    format="JPEG",
                    quality=webp_quality,
                    exif=self._build_exif(parameters, prompt=prompt, extra_pnginfo=extra_pnginfo),
                )
            elif file_format == "avif":
                img.save(
                    os.path.join(full_output_folder, file),
                    format="AVIF",
                    quality=webp_quality,
                    exif=self._build_exif(parameters, prompt=prompt, extra_pnginfo=extra_pnginfo),
                )
            elif file_format == "webp_lossless":
                img.save(
                    os.path.join(full_output_folder, file),
                    format="WEBP",
                    lossless=True,
                    quality=webp_quality,
                    method=6,
                    exif=self._build_exif(parameters, prompt=prompt, extra_pnginfo=extra_pnginfo),
                )
            else:
                img.save(
                    os.path.join(full_output_folder, file),
                    format="WEBP",
                    quality=webp_quality,
                    lossless=False,
                    method=0,
                    exif=self._build_exif(parameters, prompt=prompt, extra_pnginfo=extra_pnginfo),
                )
            saved.append({"filename": file, "subfolder": subfolder, "type": "output"})

        return {"ui": {"images": saved}}
