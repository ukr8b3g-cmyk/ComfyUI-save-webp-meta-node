"""Explicit, source-pinned pure compiler providers, using ALREADY loaded modules.

No import-by-node-name, arbitrary FUNCTION lookup, model execution or network.
The manifest pins both Python implementation and its preset data. Unknown or
updated implementations safely return unresolved until verified again.
"""
from __future__ import annotations
import hashlib
import inspect
import json
import sys
from pathlib import Path
from .types import known, unknown
from .graph import MISSING

_MANIFEST_PATH = Path(__file__).with_name("providers.json")


class VerifiedProviders:
    def __init__(self):
        self.verified = {}
        self.results = {}
        self.manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))

    def module(self, kind):
        if kind in self.verified:
            return self.verified[kind]
        self.verified[kind] = None
        spec = self.manifest["nodes"].get(kind)
        if not spec:
            return None
        nodes = sys.modules.get("nodes")
        mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) if nodes else {}
        cls = mappings.get(spec.get("registered_as", kind)) if type(mappings) is dict else None
        module = sys.modules.get(getattr(cls, "__module__", "")) if cls else None
        if module is None or not getattr(module, "__file__", None):
            return None
        path = Path(module.__file__).resolve()
        if path.name != spec["module_file"]:
            return None
        root = path.parent
        try:
            for relative, expected in spec["files"].items():
                file = root / relative
                if file.stat().st_size > 2 * 1024 * 1024:
                    return None
                # Git checkouts can use CRLF on Windows; compare canonical source text.
                raw = file.read_bytes().replace(b"\r\n", b"\n")
                if hashlib.sha256(raw).hexdigest() != expected:
                    return None
        except OSError:
            return None
        self.verified[kind] = module
        return module

    def resolve(self, resolver, ref, node, role):
        spec = self.manifest["nodes"].get(node.kind)
        module = self.module(node.kind)
        if module is None or spec is None:
            return unknown("unverified_provider_source:" + node.kind)
        cache_key = (ref.node_id, ref.slot if node.kind in {"Krea2BBOXPromptEffect", "Krea2PromptEffect", "Krea2BackgroundEffect"} else -1)
        if cache_key in self.results:
            result = self.results[cache_key]
            return result[ref.slot] if type(result) is list else result
        args = {}
        for name, config in spec["inputs"].items():
            # Effect-only output does not depend on the upstream prompt.
            if ref.slot == 1 and node.kind in {"Krea2BBOXPromptEffect", "Krea2PromptEffect", "Krea2BackgroundEffect"} and name == "prompt_in":
                args[name] = ""; continue
            if name not in node.inputs:
                if config.get("function_optional"):
                    continue
                result = unknown("missing_provider_input:" + name)
                self.results[cache_key] = result
                return result
            value = resolver.inp(ref, node, name, config["role"])
            if not value.known:
                return unknown("unresolved_provider_input:" + name)
            v, typ = value.value, config["type"]
            valid = (type(v) is str if typ == "STRING" or config["role"] in {"framing", "prompt_data"} else
                     type(v) is bool if typ == "BOOLEAN" else type(v) is int if typ == "INT" else
                     type(v) in (int, float) if typ == "FLOAT" else type(v) is str)
            if not valid:
                return unknown("invalid_provider_input:" + name)
            args[name] = v
        try:
            if node.kind == "H3CharacterSheetDesigner":
                limit = getattr(sys.modules.get("nodes"), "MAX_RESOLUTION", None)
                if type(limit) is not int or limit < 32:
                    return unknown("missing_core_resolution_limit")
                result = module.compile_state(args["state_json"], max_resolution=limit)
                values = (result["prompt"], result["width"], result["height"])
            else:
                # This is a closed allowlist of audited pure methods, NOT the node's FUNCTION value.
                cls = getattr(module, spec["class_name"])
                method = cls.execute
                if Path(inspect.getfile(method)).resolve() != Path(module.__file__).resolve():
                    return unknown("provider_method_replaced")
                # Unknown preset dictionaries cannot establish the requested effect text.
                if args.get("enable_effect", True):
                    preset, mode = args.get("preset", ""), args.get("mode", "Preset")
                    if node.kind in {"Krea2BBOXPromptEffect", "Krea2PromptEffect"}:
                        preset = {"None": "Realistic Photo", "Black White": "B&W Strong", "Realistic": "Realistic Photo", "Cinematic": "Cinematic Photo"}.get(preset, preset)
                        if mode not in {"Custom", "Custom Preset"} and preset != "Custom Preset" and preset and preset not in module.EFFECT_PRESET_TEXT:
                            return unknown("unknown_effect_preset")
                    if node.kind == "Krea2BackgroundEffect" and not args.get("custom_preset") and mode not in {"Custom", "Custom Preset"} and preset not in {"", "None", "(None)", "Custom Preset"} and preset not in module.BACKGROUND_PRESET_TEXT:
                        return unknown("unknown_background_preset")
                values = method(cls(), **args)
                if type(values) is dict:
                    values = values.get("result")
            if type(values) not in (tuple, list) or len(values) != len(spec["outputs"]):
                return unknown("invalid_provider_output")
            fields = []
            for slot, (value, typ) in enumerate(zip(values, spec["outputs"])):
                if (typ == "INT" and type(value) is not int) or (typ != "INT" and type(value) is not str):
                    return unknown("invalid_provider_output_type")
                if type(value) is str and len(value.encode("utf-8")) > 1024 * 1024:
                    return unknown("provider_text_limit")
                fields.append(known(value, [{"source": "verified_provider", "certainty": "derived", "node_id": ref.node_id,
                                             "output_slot": slot, "provider_id": spec["provider_id"]}]))
            self.results[cache_key] = fields
            return fields[ref.slot]
        except Exception as exc:
            # A pure compiler's validation failure must not prevent saving valid IMAGE pixels.
            result = unknown("provider_error:" + type(exc).__name__)
            self.results[cache_key] = result
            return result
