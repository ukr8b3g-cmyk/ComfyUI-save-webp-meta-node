"""Public pure metadata entry points. Safe partial results are intentional."""
from __future__ import annotations
import os
from .graph import Graph
from .resolver import Resolver
from .providers import VerifiedProviders
from .types import ResolutionReport


class MetadataSession:
    """A bounded cache scoped to exactly one save invocation; never reused across Queue runs."""
    def __init__(self, prompt=None, workflow=None):
        self.resolver = None
        self.error = None
        try:
            self.resolver = Resolver(Graph(prompt=prompt, workflow=workflow), VerifiedProviders())
        except Exception as exc:
            self.error = type(exc).__name__ + ": " + str(exc)[:200]

    def report(self, save_id=None, image=None, image_index=None):
        try:
            if self.resolver is not None:
                return self.resolver.report(save_id, image=image, index=image_index)
        except Exception as exc:
            self.error = type(exc).__name__ + ": " + str(exc)[:200]
        report = ResolutionReport(save_node_id=str(save_id) if save_id is not None else None, image=image)
        report.warnings.append({"code": "metadata_analysis_failed", "message": self.error or "Unknown resolver error"})
        return report


def resolve_metadata(prompt=None, workflow=None, save_id=None, image=None, image_index=None):
    return MetadataSession(prompt, workflow).report(save_id, image, image_index)


def to_legacy_info(report):
    keys = {"prompt", "negative_prompt", "seed", "steps", "cfg", "sampler", "scheduler", "denoise", "model",
            "model_family", "width", "height", "vae", "clip_skip", "rng_source", "eta_noise_seed_delta", "emphasis_mode", "method"}
    info = {k: v.value for k, v in report.fields.items() if k in keys and v.known}
    encoders = report.fields.get("text_encoders")
    if encoders and encoders.known:
        info["text_encoder"] = " | ".join(encoders.value)
    entries = report.fields.get("lora_entries")
    if entries and entries.known:
        rows = entries.value
        info["_lora_entries"] = rows
        # Preserve each model row. A different CLIP strength cannot be represented as one tag.
        tags = []
        for row in rows:
            if row.get("state") != "known" or row.get("target") != "model":
                continue
            partner = [r for r in rows if r.get("target") == "clip" and r.get("node_id") == row.get("node_id") and r.get("row") == row.get("row")]
            if any(r.get("state") != "known" or r.get("strength") != row["strength"] for r in partner):
                continue
            name = os.path.splitext(row["name"].replace("\\", "/").rsplit("/", 1)[-1])[0]
            # Do not allow user filenames to become A1111 syntax injection.
            if any(c in name for c in '<>:\n\r'):
                continue
            tags.append(f"<lora:{name}:{row['strength']:.15g}>")
        if tags:
            info["loras"] = " ".join(tags)
    return info
