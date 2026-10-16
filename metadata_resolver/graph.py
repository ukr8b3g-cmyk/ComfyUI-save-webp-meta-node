"""Read-only API/UI graph normalization; no arbitrary widget or node execution."""
from __future__ import annotations
from dataclasses import dataclass
from .types import Ref
from .profiles import WIDGETS

MISSING = object()
MAX_NODES = 10_000
MAX_EDGES = 50_000


@dataclass
class Node:
    kind: str
    inputs: dict
    sources: dict
    blocked: bool = False


def reference(value):
    if (type(value) in (list, tuple) and len(value) == 2 and type(value[0]) in (str, int)
            and type(value[1]) is int and value[1] >= 0):
        return Ref(str(value[0]), value[1])
    return None


class Graph:
    def __init__(self, prompt=None, workflow=None):
        self.nodes = {}
        self.aliases = {}
        self.warnings = []
        self.edge_count = 0
        self.source = "runtime_prompt" if type(prompt) is dict else "workflow_named"
        # An API prompt, even an empty or incomplete one, is not patched with stale UI data.
        if type(prompt) is dict:
            if len(prompt) > MAX_NODES:
                raise ValueError("graph_node_limit")
            for key, node in prompt.items():
                if type(key) not in (str, int) or type(node) is not dict:
                    continue
                inputs = node.get("inputs")
                self._add(str(key), Node(node.get("class_type", ""), inputs if type(inputs) is dict else {}, {}))
        elif type(workflow) is dict:
            defs = workflow.get("definitions", {})
            defs = defs.get("subgraphs", []) if type(defs) is dict else []
            definitions = {}
            for definition in defs if type(defs) is list else []:
                if type(definition) is dict and type(definition.get("id")) is str:
                    if definition["id"] in definitions:
                        raise ValueError("duplicate_subgraph_definition")
                    definitions[definition["id"]] = definition
            self._ui(workflow, "", definitions, {}, (), 0)

    def _add(self, key, node):
        if key in self.nodes or len(self.nodes) >= MAX_NODES:
            raise ValueError("duplicate_id_or_graph_node_limit")
        self.nodes[key] = node
        self.edge_count += len(node.inputs)
        if self.edge_count > MAX_EDGES:
            raise ValueError("graph_edge_limit")

    def _ui(self, data, prefix, definitions, boundary, ancestry, depth):
        if depth > 32:
            raise ValueError("subgraph_depth_limit")
        raw_nodes = data.get("nodes", [])
        if type(raw_nodes) is not list or len(raw_nodes) + len(self.nodes) > MAX_NODES:
            raise ValueError("invalid_or_large_workflow")
        links = {}
        for link in data.get("links", []):
            if type(link) is list and len(link) >= 6:
                lid, origin, slot = link[:3]
            elif type(link) is dict:
                lid, origin, slot = link.get("id"), link.get("origin_id"), link.get("origin_slot")
            else:
                continue
            ref = reference([origin, slot])
            if ref is None or type(lid) not in (int, str):
                continue
            if lid in links:
                raise ValueError("duplicate_link_id")
            links[lid] = boundary.get((ref.node_id, ref.slot), [prefix + ref.node_id, ref.slot])
            if len(links) > MAX_EDGES:
                raise ValueError("graph_edge_limit")
        for raw in raw_nodes:
            if type(raw) is not dict or type(raw.get("id")) not in (str, int):
                continue
            key, kind = prefix + str(raw["id"]), raw.get("type", "")
            values, origins = {}, {}
            positional = raw.get("widgets_values", [])
            names = WIDGETS.get(kind, "").split()
            if kind == "ResolutionSelector" and type(positional) is list and len(positional) == 4:
                names = ["aspect_ratio", "megapixels", "preview", "multiple"]
            if type(positional) is list:
                # Never use a short/shifted unknown legacy array for provider nodes.
                for name, value in zip(names, positional):
                    values[name], origins[name] = value, "workflow_positional"
                if kind == "Power Lora Loader (rgthree)":
                    for i, value in enumerate(positional):
                        if type(value) is dict and "lora" in value:
                            values[f"lora_{i}"] = value
            named = raw.get("widgets_values_named")
            if type(named) is dict:
                for name, value in named.items():
                    values[name], origins[name] = value, "workflow_named"
            for inp in raw.get("inputs", []):
                if type(inp) is dict and inp.get("link") is not None:
                    name = inp.get("name")
                    # Missing link blocks the old widget value; it must not resurrect it.
                    values[name] = links.get(inp["link"], MISSING)
                    origins[name] = "workflow_named"
            blocked = raw.get("mode", 0) not in (0, None)
            self._add(key, Node(kind, values, origins, blocked))
            definition = definitions.get(kind)
            if definition and not blocked:
                if kind in ancestry:
                    raise ValueError("recursive_subgraph_definition")
                # The official -10/-20 boundary format is only used when present explicitly.
                inode, onode = definition.get("inputNode", {}), definition.get("outputNode", {})
                if type(inode) is not dict or type(onode) is not dict or "id" not in inode or "id" not in onode:
                    continue
                bounds = {(str(inode["id"]), i): values.get(p.get("name"), MISSING)
                          for i, p in enumerate(definition.get("inputs", [])) if type(p) is dict}
                child_prefix = key + ":"
                self._ui(definition, child_prefix, definitions, bounds, ancestry + (kind,), depth + 1)
                for i, port in enumerate(definition.get("outputs", [])):
                    ids = port.get("linkIds", []) if type(port) is dict else []
                    if len(ids) != 1:
                        continue
                    for link in definition.get("links", []):
                        if (type(link) is dict and link.get("id") == ids[0]
                                and link.get("target_id") == onode["id"] and link.get("target_slot") == i):
                            r = reference([link.get("origin_id"), link.get("origin_slot")])
                            if r:
                                self.aliases[Ref(key, i)] = bounds.get((r.node_id, r.slot), [child_prefix + r.node_id, r.slot])

    def save_ref(self, save_id):
        if save_id is not None:
            node = self.nodes.get(str(save_id))
            if node and node.kind == "SaveWebPMeta" and not node.blocked:
                return str(save_id), node.inputs.get("images", MISSING)
            return str(save_id), MISSING
        saves = [(k, n) for k, n in self.nodes.items() if n.kind == "SaveWebPMeta" and not n.blocked]
        if len(saves) != 1:
            return None, MISSING
        key, node = saves[0]
        return key, node.inputs.get("images", MISSING)
