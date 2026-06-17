import importlib.util
import sys

base = r'D:/Codex/ComfyUI/custom_nodes/webp_a1111_metadata'
for name in ['__init__.py', 'webp_save.py']:
    path = f'{base}/{name}'
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
print('import ok')
