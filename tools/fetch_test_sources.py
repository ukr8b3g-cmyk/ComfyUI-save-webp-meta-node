"""Fetch fixed public source for release parity tests. Never imported by the node."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import urllib.request

PINS = {
    'bbox': ('ukr8b3g-cmyk/Krea2-BBOX-Prompter', 'fd6d5edb8297d8c50ad45a8199ca523282f53832',
             ['nodes_element_framing.py','web/background_presets.json']),
    'pose': ('ukr8b3g-cmyk/custom_nodes-krea2_bbox_pose_gesture_v1', '62f0197195e5a2a12002e5d5b0eaaf6e441cd1bb',
             ['nodes_pose_gesture_injector.py','pose_gesture_presets.json']),
    'h3': ('ukr8b3g-cmyk/H3-Character-Sheet-Designer', '9244cb6abeac2daaeabd42c984948def55f8d7ce',
           ['h3_character_sheet/__init__.py','h3_character_sheet/node.py','h3_character_sheet/compiler.py']),
}

def main():
    root = Path(os.environ['SAVE_META_PROVIDER_SOURCES'])
    root.mkdir(parents=True, exist_ok=True)
    for alias, (repo, sha, paths) in PINS.items():
        for relative in paths:
            request = urllib.request.Request(f'https://raw.githubusercontent.com/{repo}/{sha}/{relative}',headers={'User-Agent':'save-webp-metadata-tests'})
            with urllib.request.urlopen(request, timeout=60) as response:
                content=response.read(2_000_001)
            if len(content)>2_000_000:
                raise ValueError('Oversized test source')
            target=root/alias/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
    (root/'pins.json').write_text(json.dumps(PINS,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
