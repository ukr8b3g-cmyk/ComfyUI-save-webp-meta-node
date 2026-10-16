"""Bounded, output-slot-aware metadata extraction. No GPU, network or graph writes."""
from __future__ import annotations
import math
import re
from dataclasses import replace
from .graph import Graph, MISSING, reference
from .types import Ref, Field, ResolutionReport, known, unknown, na
from . import profiles as P


class Resolver:
    def __init__(self, graph, providers=None):
        self.graph = graph
        self.memo = {}
        self.visiting = set()
        self.providers = providers
        self.provider_memo = {}
        self.calls = 0

    def resolve(self, value, role, index=None):
        if role in ("string", "framing", "prompt_data") and type(value) is str:
            return known(value) if len(value.encode("utf-8")) <= 1024 * 1024 else unknown("text_limit")
        if role == "scalar" and type(value) in (int, float, bool, str):
            return known(value) if type(value) is not float or math.isfinite(value) else unknown("non_finite")
        ref = reference(value)
        if ref is None:
            return unknown("missing_or_invalid_reference")
        key = (ref, role, index)
        if key in self.memo:
            return self.memo[key]
        if key in self.visiting:
            return unknown("cycle")
        if len(self.visiting) >= 192 or len(self.memo) >= 10000 or self.calls >= 50000:
            return unknown("resolution_limit")
        self.calls += 1
        node = self.graph.nodes.get(ref.node_id)
        if node is None or node.blocked:
            return unknown("missing_node_or_unsupported_mode")
        self.visiting.add(key)
        try:
            if ref in self.graph.aliases:
                result = self.resolve(self.graph.aliases[ref], role, index)
            else:
                result = self._resolve(ref, node, role, index)
            if not isinstance(result, Field):
                result = unknown("invalid_adapter_result")
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError, UnicodeError) as exc:
            result = unknown("invalid_node_data:" + type(exc).__name__)
        finally:
            self.visiting.remove(key)
        self.memo[key] = result
        return result

    def inp(self, ref, node, name, role="scalar", index=None):
        result = self.resolve(node.inputs.get(name, MISSING), role, index)
        source = node.sources.get(name, self.graph.source)
        prov = {"source": source, "certainty": "explicit", "node_id": ref.node_id,
                "output_slot": ref.slot, "input_name": name}
        return replace(result, provenance=result.provenance + [prov])

    def _resolve(self, ref, node, role, index):
        kind, inputs, slot = node.kind, node.inputs, ref.slot
        if kind == "ComfySwitchNode" and slot == 0:
            switch = self.inp(ref, node, "switch")
            if not switch.known or type(switch.value) is not bool:
                return unknown("unresolved_selector", True)
            return self.inp(ref, node, "on_true" if switch.value else "on_false", role, index)
        if kind in P.PROVIDER_ROLES:
            slots = P.PROVIDER_ROLES[kind]
            if slot >= len(slots) or role != slots[slot]:
                return unknown("provider_output_role_mismatch")
            if self.providers is None:
                return unknown("provider_unavailable")
            return self.providers.resolve(self, ref, node, role)
        if role in ("string", "scalar"):
            if slot == 0 and kind in {"PrimitiveString", "PrimitiveStringMultiline", "PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean"}:
                expected = "string" if "String" in kind else "scalar"
                if role != expected:
                    return unknown("primitive_type_mismatch")
                name = "value" if "value" in inputs else "text" if "String" in kind else "value"
                f = self.inp(ref, node, name, role)
                if f.known and ((kind == "PrimitiveBoolean" and type(f.value) is not bool)
                                or (kind == "PrimitiveInt" and type(f.value) is not int)
                                or (kind == "PrimitiveFloat" and type(f.value) not in (int, float))):
                    return unknown("primitive_type_mismatch")
                return f
            if role == "scalar" and kind == "ResolutionSelector" and slot in (0, 1):
                ratio, mp, multiple = [self.inp(ref, node, k) for k in ("aspect_ratio", "megapixels", "multiple")]
                if not all(f.known for f in (ratio, mp, multiple)):
                    return unknown("unresolved_resolution")
                match = re.fullmatch(r"(\d+):(\d+)(?: \([^)]*\))?", str(ratio.value))
                if not match or type(mp.value) not in (int, float) or type(multiple.value) is not int or multiple.value <= 0:
                    return unknown("invalid_resolution")
                a, b = map(int, match.groups())
                scale = math.sqrt(mp.value * 1024 * 1024 / (a * b))
                return known(round((a, b)[slot] * scale / multiple.value) * multiple.value)
            if role == "string" and slot == 0 and kind == "StringConcatenate":
                fields = [self.inp(ref, node, k, "string") for k in ("string_a", "string_b", "delimiter")]
                if not all(f.known for f in fields):
                    return unknown("unresolved_string_operand")
                return self.resolve(fields[2].value.join((fields[0].value, fields[1].value)), "string")
            if role == "string" and slot == 0 and kind == "StringSubstring":
                text, start, end = self.inp(ref, node, "string", "string"), self.inp(ref, node, "start"), self.inp(ref, node, "end")
                if all(f.known for f in (text, start, end)) and type(start.value) is int and type(end.value) is int:
                    return known(text.value[start.value:end.value])
                return unknown("unresolved_string_operand")
            return unknown("unsupported_value_node")
        if role in ("image", "latent"):
            if role == "latent" and kind in P.SAMPLERS and slot < P.SAMPLERS[kind]:
                return known({"producer": ref})
            if slot != 0:
                return unknown("invalid_output_slot")
            if role == "image" and kind in {"LoadImage", "LoadImageOutput", "EmptyImage"}:
                return known({"external": True})
            if role == "image" and kind in {"VAEDecode", "VAEDecodeTiled"}:
                path = self.inp(ref, node, "samples", "latent", index)
                if not path.known:
                    return path
                return known({**path.value, "vae": self.inp(ref, node, "vae", "vae")})
            if role == "image" and kind in {"ImageBatch", "ImageBatchMultiple"}:
                if kind != "ImageBatch":
                    return unknown("ambiguous_producer", True)
                left, right = inputs.get("image1", MISSING), inputs.get("image2", MISSING)
                count = self.resolve(left, "count")
                if index is not None and count.known and type(count.value) is int:
                    return self.resolve(left if index < count.value else right, role, index if index < count.value else index - count.value)
                a, b = self.resolve(left, role, index), self.resolve(right, role, index)
                if a.known and b.known and a.value.get("producer") and a.value.get("producer") == b.value.get("producer"):
                    return a
                return unknown("ambiguous_producer", True)
            if role == "image" and kind == "ImageFromBatch":
                count = self.inp(ref, node, "image", "count")
                requested = self.inp(ref, node, "batch_index")
                next_index = None
                if count.known and requested.known and type(requested.value) is int and count.value > 0:
                    start = requested.value + (count.value if requested.value < 0 else 0)
                    next_index = max(0, min(count.value - 1, start)) + (index or 0)
                path = self.inp(ref, node, "image", role, next_index)
                if path.known:
                    return known({**path.value, "frame_requested": requested,
                                  "frame_index": known(next_index) if next_index is not None else unknown("unknown_input_batch_length")})
                return path
            if role == "image" and kind == "RepeatImageBatch":
                count = self.inp(ref, node, "image", "count")
                return self.inp(ref, node, "image", role, index % count.value if index is not None and count.known and count.value else None)
            paths = P.IMAGE_PATHS if role == "image" else P.LATENT_PATHS
            if kind in paths:
                return self.inp(ref, node, paths[kind], role, index)
            return unknown("unsupported_" + role + "_node:" + kind)
        if role == "count":
            if kind in P.SAMPLERS and slot < P.SAMPLERS[kind]:
                return self.inp(ref, node, "latent_image", role)
            if kind in ("MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo") and slot == 1 or kind == "EmptyMiniMaxH3LatentAV" and slot == 0:
                n = self.inp(ref, node, "length")
                if n.known and type(n.value) is int:
                    count = max(5, n.value)
                    return known(count + (5 - count) % 17)
            if slot == 0 and kind in P.WIDGETS and kind.startswith("Empty"):
                n = self.inp(ref, node, "batch_size")
                return n if n.known and type(n.value) is int and n.value > 0 else unknown("unknown_batch_length")
            if slot == 0 and kind in {"VAEDecode", "VAEDecodeTiled"}:
                return self.inp(ref, node, "samples", role)
            if slot == 0 and kind == "ImageFromBatch":
                n, start, length = [self.inp(ref, node, k, "count" if k == "image" else "scalar") for k in ("image", "batch_index", "length")]
                if all(f.known and type(f.value) is int for f in (n, start, length)) and n.value > 0 and length.value > 0:
                    i = max(0, min(n.value - 1, start.value + (n.value if start.value < 0 else 0)))
                    return known(min(length.value, n.value - i))
            if slot == 0 and kind == "RepeatImageBatch":
                n, amount = self.inp(ref, node, "image", role), self.inp(ref, node, "amount")
                if n.known and amount.known and type(amount.value) is int and amount.value > 0:
                    return known(n.value * amount.value)
            if slot == 0 and kind == "ImageBatch":
                a, b = self.inp(ref, node, "image1", role), self.inp(ref, node, "image2", role)
                if a.known and b.known:
                    return known(a.value + b.value)
            if slot == 0 and kind in P.IMAGE_PATHS:
                return self.inp(ref, node, P.IMAGE_PATHS[kind], role)
            return unknown("unknown_batch_length")
        if role in ("model", "clip", "vae"):
            return self._asset(ref, node, role)
        if role in ("positive", "negative"):
            return self._conditioning(ref, node, role)
        return unknown("unsupported_role")

    def _asset(self, ref, node, role):
        kind, inputs, slot = node.kind, node.inputs, ref.slot
        if kind in {"CheckpointLoaderSimple", "CheckpointLoader"} and slot == {"model": 0, "clip": 1, "vae": 2}[role]:
            name = self.inp(ref, node, "ckpt_name", "string")
            return known({"names": [name.value], "loras": [], "checkpoint": True}, name.provenance) if name.known else name
        if slot == 0 and ((role == "model" and kind == "UNETLoader") or (role == "vae" and kind == "VAELoader")):
            name = self.inp(ref, node, "unet_name" if role == "model" else "vae_name", "string")
            return known({"names": [name.value], "loras": []}, name.provenance) if name.known else name
        if role == "clip" and kind in P.CLIP_LOADERS and slot == 0:
            fs = [self.inp(ref, node, k, "string") for k in P.CLIP_LOADERS[kind]]
            if not all(f.known and f.value for f in fs):
                return unknown("unresolved_text_encoder")
            typ = self.inp(ref, node, "type", "string")
            return known({"names": [f.value for f in fs], "family": P.CLIP_FAMILY.get(typ.value) if typ.known else None,
                          "loras": []}, [p for f in fs for p in f.provenance])
        lora_role = None
        if kind in {"LoraLoader", "Power Lora Loader (rgthree)"}:
            lora_role = {0: "model", 1: "clip"}.get(slot)
        elif kind in {"LoraLoaderModelOnly", "LoadLoraModel"} and slot == 0:
            lora_role = "model"
        elif kind == "LoadLoraTextEncoder" and slot == 0:
            lora_role = "clip"
        if lora_role == role:
            source = self.inp(ref, node, role, role)
            if not source.known:
                return source
            rows = self._loras(ref, node, role)
            return known({**source.value, "loras": source.value.get("loras", []) + rows})
        if role == "model" and kind in P.MODEL_PATHS and slot == 0:
            source = self.inp(ref, node, "model", role)
            if source.known:
                settings = {k: self.inp(ref, node, k).as_dict() for k in inputs
                            if k not in {"model", "model_patch", "vae", "audio_vae", "image", "control_video", "mask", "source_video"}}
                modifiers = source.value.get("modifiers", []) + [{"node_id": ref.node_id, "class_type": kind, "settings": settings}]
                return known({**source.value, "modifiers": modifiers})
            return source
        if role == "clip" and kind in {"CLIPSetLastLayer", "CLIPSkip"} and slot == 0:
            source = self.inp(ref, node, "clip", role)
            value = self.inp(ref, node, "stop_at_clip_layer" if kind == "CLIPSetLastLayer" else "clip_skip")
            if source.known:
                return known({**source.value, "clip_skip": known(abs(value.value), value.provenance)
                              if value.known and type(value.value) is int else value})
            return source
        return unknown("unsupported_" + role + "_path:" + kind)

    def _loras(self, ref, node, role):
        raw = []
        kind, inputs = node.kind, node.inputs
        if kind in {"LoadLoraModel", "LoadLoraTextEncoder"}:
            indices = {int(m[1]) for k in inputs if type(k) is str and (m := re.fullmatch(r"loras\.(\d+)\.(?:lora_name|strength|enabled)", k))}
            if any(i > 19 for i in indices):
                return [{"state": "unresolved", "reason": "lora_index_out_of_range"}]
            for i in sorted(indices):
                raw.append((i, {k: inputs.get(f"loras.{i}.{k}", MISSING) for k in ("lora_name", "strength", "enabled")}))
            if type(inputs.get("loras")) is list and not indices:
                if len(inputs["loras"]) > 20:
                    return [{"state": "unresolved", "reason": "lora_index_out_of_range"}]
                raw = list(enumerate(inputs["loras"]))
            if not raw:
                return [{"state": "unresolved", "reason": "missing_lora_rows"}]
        elif kind == "Power Lora Loader (rgthree)":
            for k, row in inputs.items():
                if type(row) is dict and "lora" in row:
                    raw.append((k, {"lora_name": row["lora"], "enabled": row.get("on", True),
                                    "strength": row.get("strengthTwo", row.get("strength", 1)) if role == "clip" else row.get("strength", 1)}))
        else:
            raw.append((0, {"lora_name": inputs.get("lora_name", MISSING), "enabled": True,
                            "strength": inputs.get("strength_" + role, MISSING)}))
        rows = []
        for i, row in raw:
            if type(row) is not dict:
                rows.append({"state": "unresolved", "reason": "malformed_lora"}); continue
            enabled = self.resolve(row.get("enabled", MISSING), "scalar")
            if enabled.known and enabled.value is False:
                continue
            strength = self.resolve(row.get("strength", MISSING), "scalar")
            if strength.known and type(strength.value) in (int, float) and strength.value == 0:
                continue
            name = self.resolve(row.get("lora_name", MISSING), "string")
            if name.known and not name.value:
                continue
            if not (enabled.known and enabled.value is True and strength.known
                    and type(strength.value) in (int, float) and name.known):
                rows.append({"state": "unresolved", "reason": "unresolved_lora_row", "node_id": ref.node_id, "row": i}); continue
            rows.append({"state": "known", "name": name.value, "strength": strength.value, "target": role,
                         "node_id": ref.node_id, "row": i})
        return rows

    def _conditioning(self, ref, node, role):
        kind, inputs, slot = node.kind, node.inputs, ref.slot
        if kind == "ConditioningZeroOut" and slot == 0:
            upstream = self.inp(ref, node, "conditioning", role)
            return known({**(upstream.value if upstream.known else {}), "text": known("")})
        if kind in P.COND_PATHS and slot == 0 or kind in P.DUAL_COND and slot in (0, 1):
            name = P.COND_PATHS[kind] if kind in P.COND_PATHS else ("positive", "negative")[slot]
            source = self.inp(ref, node, name, role)
            if source.known:
                extra = {"node_id": ref.node_id, "class_type": kind, "slot": slot,
                         "settings": {k: self.inp(ref, node, k).as_dict() for k in inputs
                                      if k not in {"conditioning", "positive", "negative", "image", "latent", "vae", "audio_vae", "control_net", "mask"}
                                      and not k.startswith("images.")}}
                out = {**source.value, "modifiers": source.value.get("modifiers", []) + [extra]}
                if kind == "FluxGuidance":
                    out["guidance"] = self.inp(ref, node, "guidance")
                elif kind == "FluxDisableGuidance":
                    out["guidance"] = na("guidance_disabled")
                return known(out)
            return source
        if kind in {"ConditioningCombine", "ConditioningConcat", "ConditioningAverage"} and slot == 0:
            keys = {"ConditioningCombine": ("conditioning_1", "conditioning_2"),
                    "ConditioningConcat": ("conditioning_to", "conditioning_from"),
                    "ConditioningAverage": ("conditioning_to", "conditioning_from")}[kind]
            branches = [self.inp(ref, node, k, role) for k in keys]
            out = {"text": unknown("composite_conditioning_is_not_a_single_prompt"), "branches": [],
                   "modifiers": [{"node_id": ref.node_id, "class_type": kind,
                                  "settings": {k: self.inp(ref, node, k).as_dict() for k in inputs if k not in keys}}]}
            for branch in branches:
                if branch.known:
                    text = branch.value.get("text", unknown("missing_text"))
                    out["branches"].append(text.as_dict())
            return known(out)
        if kind == "AnimaRegionalCanvas" and slot in (0, 1):
            keys = ("negative_prompt",) if slot == 1 else ("quality_prompt", "scene_prompt", "red_prompt", "blue_prompt", "yellow_prompt", "green_prompt", "magenta_prompt")
            fs = [self.inp(ref, node, k, "string") if k in inputs else known("") for k in keys]
            if "quality_prompt" not in inputs and slot == 0 and "base_prompt" in inputs:
                fs[0] = self.inp(ref, node, "base_prompt", "string")
            text = known("\n\n".join(f.value.strip() for f in fs if f.value.strip())) if all(f.known for f in fs) else unknown("unresolved_regional_text")
            return known({"text": text, "family": "Anima"})
        if kind in P.DUAL_ENCODERS and slot in (0, 1):
            keys = ("prompt" if slot == 0 else "negative_prompt",)
        elif kind in P.TEXT_ENCODERS and slot == 0:
            keys = P.TEXT_ENCODERS[kind]
        else:
            return unknown("unsupported_conditioning_node_or_slot:" + kind)
        fs = [self.inp(ref, node, k, "string") for k in keys]
        fragments = [{"encoder_input": k, **f.as_dict()} for k, f in zip(keys, fs)]
        if not all(f.known for f in fs):
            text = unknown("unresolved_runtime_text")
        else:
            distinct = list(dict.fromkeys(f.value for f in fs))
            text = known(distinct[0] if len(distinct) == 1 else "\n".join(f"[{k}] {f.value}" for k, f in zip(keys, fs)),
                         [p for f in fs for p in f.provenance])
        out = {"text": text, "clip": self.inp(ref, node, "clip", "clip"), "fragments": fragments,
               "family": P.ENCODER_FAMILY.get(kind), "references": []}
        for key, value in inputs.items():
            if type(key) is str and (key.startswith(("images.", "ref_images.", "ref_videos.", "ref_audios.", "ref_video_audios."))
                                     or key in {"image", "image1", "image2", "image3", "first_frame", "last_frame"}):
                r = reference(value)
                if r:
                    out["references"].append({"input": key, "node_id": r.node_id, "output_slot": r.slot})
        def order(item):
            key = item["input"]
            m = re.search(r"(\d+)$", key)
            return (re.sub(r"\d+$", "", key), int(m[1]) if m else -1)
        out["references"].sort(key=order)
        if kind == "CLIPTextEncodeFlux":
            out["guidance"] = self.inp(ref, node, "guidance")
        if kind.startswith("MiniMaxH3"):
            out["h3_length_requested"] = self.inp(ref, node, "length")
        return known(out)

    def _setting(self, ref, node, name, value_type="number"):
        f = self.inp(ref, node, name, "string" if value_type == "string" else "scalar")
        if f.known:
            if value_type == "steps":
                return f if type(f.value) is int and f.value >= 0 else unknown("invalid_steps")
            valid = (type(f.value) is int and 0 <= f.value < 2 ** 64) if value_type == "seed" else (
                type(f.value) in (int, float) and math.isfinite(f.value)) if value_type == "number" else type(f.value) is str
            if not valid:
                return unknown("invalid_" + name)
        return f

    def stage(self, ref):
        node = self.graph.nodes[ref.node_id]
        kind = node.kind
        fields = {}
        model_ref = node.inputs.get("model", MISSING)
        positive_ref, negative_ref = node.inputs.get("positive", MISSING), node.inputs.get("negative", MISSING)
        cfg = self._setting(ref, node, "cfg")
        if kind == "SamplerCustomAdvanced":
            noise = self.follow_switch(node.inputs.get("noise", MISSING))
            noise_node = self.graph.nodes.get(noise.node_id) if noise else None
            if noise_node and noise.slot == 0 and noise_node.kind == "RandomNoise":
                fields["seed"] = self._setting(noise, noise_node, "noise_seed", "seed")
            else:
                fields["seed"] = na("noise_disabled") if noise_node and noise_node.kind == "DisableNoise" and noise.slot == 0 else unknown("unknown_noise_node")
            guider_ref = self.follow_switch(node.inputs.get("guider", MISSING))
            guider = self.graph.nodes.get(guider_ref.node_id) if isinstance(guider_ref, Ref) else None
            if guider and guider_ref.slot == 0 and guider.kind in {"BasicGuider", "CFGGuider", "DualCFGGuider", "DualModelGuider"}:
                model_ref = guider.inputs.get("model", MISSING)
                positive_ref = guider.inputs.get("conditioning" if guider.kind == "BasicGuider" else "cond1" if guider.kind == "DualCFGGuider" else "positive", MISSING)
                negative_ref = guider.inputs.get("negative", MISSING)
                cfg = na("no_explicit_cfg") if guider.kind == "BasicGuider" else self._setting(guider_ref, guider, "cfg")
                if guider.kind == "BasicGuider":
                    fields["negative_prompt"] = na("no_negative_conditioning")
                if guider.kind == "DualCFGGuider":
                    cfg = na("dual_cfg")
                    for name in ("cfg_conds", "cfg_cond2_negative", "style"):
                        fields[name] = self._setting(guider_ref, guider, name, "string" if name == "style" else "number")
                    cond2 = self.resolve(guider.inputs.get("cond2", MISSING), "positive")
                    fields["secondary_prompt"] = cond2.value["text"] if cond2.known and "text" in cond2.value else unknown("secondary_conditioning")
                if guider.kind == "DualModelGuider":
                    alt = self.resolve(guider.inputs.get("model_negative", MISSING), "model")
                    fields["model_negative"] = known(alt.value["names"]) if alt.known else alt
                    if "negative" not in guider.inputs:
                        fields["negative_prompt"] = na("text_free_unconditional")
            else:
                cfg = unknown("unknown_guider")
        else:
            fields["seed"] = self._setting(ref, node, "seed" if kind == "KSampler" else "noise_seed", "seed")
        fields["cfg"] = cfg
        if kind in {"KSampler", "KSamplerAdvanced"}:
            for dest, source, typ in (("steps", "steps", "steps"), ("sampler", "sampler_name", "string"), ("scheduler", "scheduler", "string")):
                fields[dest] = self._setting(ref, node, source, typ)
            if kind == "KSampler":
                fields["denoise"] = self._setting(ref, node, "denoise")
            else:
                for name in ("add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise"):
                    fields[name] = self.inp(ref, node, name)
                fields["denoise"] = na("advanced_range_not_denoise")
        else:
            fields.update(self.schedule(node.inputs.get("sigmas", MISSING)))
            fields["sampler"] = self.sampler_name(node.inputs.get("sampler", MISSING))
            if kind == "SamplerCustom":
                fields["add_noise"] = self.inp(ref, node, "add_noise")
        model = self.resolve(model_ref, "model")
        fields["model"] = known(model.value["names"][0], model.provenance) if model.known else model
        positive = self.resolve(positive_ref, "positive")
        negative = self.resolve(negative_ref, "negative")
        fields["prompt"] = positive.value.get("text", unknown("missing_text")) if positive.known else positive
        fields.setdefault("negative_prompt", negative.value.get("text", unknown("missing_text")) if negative.known else negative)
        clips = []
        for label, condition in (("positive", positive), ("negative", negative)):
            if not condition.known:
                continue
            cv = condition.value
            clip = cv.get("clip", unknown("missing_clip"))
            if clip.known:
                clips.append(clip)
            for key in ("guidance", "h3_length_requested"):
                if key in cv and label == "positive":
                    fields["embedded_guidance" if key == "guidance" else key] = cv[key]
            for key in ("fragments", "references", "modifiers", "branches"):
                if cv.get(key):
                    fields[label + "_" + key] = known(cv[key])
        if clips:
            fields["text_encoders"] = known(list(dict.fromkeys(name for c in clips for name in c.value["names"])))
            for c in clips:
                if "clip_skip" in c.value:
                    fields["clip_skip"] = c.value["clip_skip"]
        family = positive.value.get("family") if positive.known else None
        if not family:
            family = next((c.value.get("family") for c in clips if c.value.get("family")), None)
        if family:
            fields["model_family"] = known(family)
        # Filename hints are kept out of the asserted model-family field.
        elif fields["model"].known:
            fields["model_name_hint"] = known(fields["model"].value)
        loras = model.value.get("loras", [])[:] if model.known else []
        seen_clip = set()
        for clip in clips:
            for row in clip.value.get("loras", []):
                marker = (row.get("node_id"), row.get("row"), row.get("target"))
                if marker not in seen_clip:
                    seen_clip.add(marker); loras.append(row)
        if loras:
            fields["lora_entries"] = known(loras)
        if model.known and model.value.get("modifiers"):
            fields["model_modifiers"] = known(model.value["modifiers"])
        # Only the producing sampler may supply these optional explicit values.
        for key in ("rng_source", "eta_noise_seed_delta", "emphasis_mode", "method"):
            if key in node.inputs:
                fields[key] = self.inp(ref, node, key)
        return fields

    def follow_switch(self, value):
        visited = set()
        while (ref := reference(value)) is not None and ref not in visited and len(visited) < 192:
            visited.add(ref)
            if ref in self.graph.aliases:
                value = self.graph.aliases[ref]; continue
            node = self.graph.nodes.get(ref.node_id)
            if node is None or node.blocked:
                return None
            if node.kind != "ComfySwitchNode" or ref.slot != 0:
                return ref
            f = self.inp(ref, node, "switch")
            if not f.known or type(f.value) is not bool:
                return None
            value = node.inputs.get("on_true" if f.value else "on_false", MISSING)
        return None

    def schedule(self, value):
        ref = self.follow_switch(value)
        node = self.graph.nodes.get(ref.node_id) if ref else None
        if not node or ref.slot != 0 or node.kind not in {"BasicScheduler", *P.SCHEDULES}:
            return {"steps": unknown("unsupported_sigmas_transform"), "scheduler": unknown("unsupported_sigmas_transform")}
        fields = {"steps": self._setting(ref, node, "steps", "steps")}
        fields["scheduler"] = self._setting(ref, node, "scheduler", "string") if node.kind == "BasicScheduler" else known(P.SCHEDULES[node.kind])
        if node.kind in {"BasicScheduler", "SDTurboScheduler"}:
            fields["denoise"] = self._setting(ref, node, "denoise")
            if fields["denoise"].known and fields["denoise"].value <= 0 and node.kind == "BasicScheduler":
                fields["steps_requested"] = fields["steps"]
                fields["steps"] = known(0)
        return fields

    def sampler_name(self, value):
        ref = self.follow_switch(value)
        node = self.graph.nodes.get(ref.node_id) if ref else None
        if not node or ref.slot != 0:
            return unknown("unknown_sampler")
        if node.kind == "KSamplerSelect":
            return self._setting(ref, node, "sampler_name", "string")
        if node.kind in P.FIXED_SAMPLERS:
            return known(P.FIXED_SAMPLERS[node.kind])
        if node.kind in {"SamplerDPMPP_SDE", "SamplerDPMPP_2M_SDE"}:
            d = self.inp(ref, node, "noise_device", "string")
            if d.known and d.value in {"cpu", "gpu"}:
                name = "dpmpp_sde" if node.kind == "SamplerDPMPP_SDE" else "dpmpp_2m_sde"
                return known(name + ("_gpu" if d.value == "gpu" else ""))
        return unknown("unsupported_sampler_node")

    def report(self, save_id=None, image=None, index=None):
        key, value = self.graph.save_ref(save_id)
        report = ResolutionReport(save_node_id=key, image=image)
        path = self.resolve(value, "image", index)
        if not path.known:
            if save_id is None and sum(n.kind == "SaveWebPMeta" and not n.blocked for n in self.graph.nodes.values()) > 1:
                path = unknown("ambiguous_save_node", True)
            report.status = "ambiguous" if path.state == "ambiguous" else "partial"
            report.warnings.append({"code": path.reason, "message": "Image provenance could not be resolved; no other branch was substituted."})
            return report
        if path.value.get("external"):
            report.status = "external"
            return report
        ref = path.value.get("producer")
        if not isinstance(ref, Ref):
            return report
        report.fields = self.stage(ref)
        if "vae" in path.value:
            vae = path.value["vae"]
            report.fields["vae"] = known(vae.value["names"][0], vae.provenance) if vae.known else vae
        for k in ("frame_requested", "frame_index"):
            if k in path.value:
                report.fields[k] = path.value[k]
        # Track ancestral stages separately, without mixing their settings into the summary.
        stage_refs, current, seen = [], ref, set()
        while current not in seen and len(seen) < 64:
            seen.add(current); stage_refs.append(current)
            n = self.graph.nodes[current.node_id]
            ancestor = self.resolve(n.inputs.get("latent_image", MISSING), "latent")
            if not ancestor.known or not isinstance(ancestor.value.get("producer"), Ref):
                break
            current = ancestor.value["producer"]
        stage_refs.reverse()
        for i, r in enumerate(stage_refs):
            n = self.graph.nodes[r.node_id]
            report.producers.append({"node_id": r.node_id, "output_slot": r.slot, "class_type": n.kind,
                                     "stage_index": i, "output_variant": "denoised_output" if r.slot == 1 else "output"})
        if len(stage_refs) > 1:
            report.fields["ancestor_stages"] = known([{k: f.as_dict() for k, f in self.stage(r).items()} for r in stage_refs[:-1]])
        # Static latent dimensions are useful for inspection only; the actual IMAGE replaces them at save time.
        latent_ref = self.follow_switch(self.graph.nodes[ref.node_id].inputs.get("latent_image", MISSING))
        latent = self.graph.nodes.get(latent_ref.node_id) if latent_ref else None
        if latent and latent.kind in {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyFlux2LatentImage", "EmptyMiniMaxH3LatentAV"}:
            for axis in ("width", "height"):
                f = self.inp(latent_ref, latent, axis)
                if f.known and type(f.value) is int and f.value > 0:
                    report.fields[axis] = f
        if image:
            for axis in ("width", "height"):
                report.fields[axis] = known(image[axis], [{"source": "image_tensor", "certainty": "explicit"}])
        report.status = "complete"
        rows = report.fields.get("lora_entries")
        if rows and rows.known and any(r.get("state") != "known" for r in rows.value):
            report.status = "partial"
            report.warnings.append({"code": "unresolved_lora_rows", "message": "Some LoRA applications could not be determined.", "field": "lora_entries"})
        for k, f in report.fields.items():
            if f.state in {"unresolved", "ambiguous"}:
                report.status = "partial"
                report.warnings.append({"code": f.reason, "message": "Field omitted rather than guessed.", "field": k})
        return report
