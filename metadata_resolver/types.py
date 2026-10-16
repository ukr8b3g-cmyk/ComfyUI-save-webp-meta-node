"""Field-level evidence for metadata, never a replacement for the raw workflow."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Ref:
    node_id: str
    slot: int = 0


@dataclass
class Field:
    state: str = "unresolved"
    value: Any = None
    reason: str = "missing_input"
    provenance: list[dict] = field(default_factory=list)

    @property
    def known(self):
        return self.state in ("known", "known_empty")

    def as_dict(self):
        out = {"state": self.state, "provenance": self.provenance}
        if self.known:
            out["value"] = self.value
        else:
            out["reason"] = self.reason
        return out


def known(value, provenance=()):
    if value is None:
        return unknown("null_value")
    return Field("known_empty" if value == "" else "known", value, "", list(provenance))


def unknown(reason, ambiguous=False):
    return Field("ambiguous" if ambiguous else "unresolved", reason=reason)


def na(reason):
    return Field("not_applicable", reason=reason)


@dataclass
class ResolutionReport:
    save_node_id: str | None = None
    image: dict | None = None
    fields: dict[str, Field] = field(default_factory=dict)
    producers: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    status: str = "partial"

    def as_dict(self):
        # A dry-run without an IMAGE has no image dimensions. Saving always supplies them.
        return {"schema_version": "save_webp_meta.resolve/1", "save_node_id": self.save_node_id,
                "image": self.image, "status": self.status,
                "fields": {k: v.as_dict() for k, v in self.fields.items()},
                "producers": self.producers, "warnings": self.warnings}
