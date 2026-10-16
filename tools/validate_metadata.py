"""Release gate: no silent skips, source syntax, original tests unchanged, and CPU tests."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))

def main():
    original = ROOT/'tests/test_webp_save.py'
    raw = original.read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == 'fe534ca0172eaa3e5f750b742e09892d9f5f8b22', 'Original tests changed'
    assert os.environ.get('SAVE_META_PROVIDER_SOURCES'), 'Provider tests must run for release'
    checked=[]
    for name in ['webp_save.py','__init__.py',*map(str,Path('metadata_resolver').glob('*.py')),*map(str,Path('tests').glob('test_*.py')),*map(str,Path('tools').glob('*.py'))]:
        ast.parse(Path(name).read_text(encoding='utf-8-sig'),filename=name,feature_version=(3,10))
        checked.append(name)
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
    from PIL import __version__ as pillow_version, features
    import numpy
    summary={'python':sys.version,'os':platform.platform(),'pillow':pillow_version,'numpy':numpy.__version__,
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'seconds':round(time.monotonic()-started,3),'syntax_files':checked,'webp':features.check('webp'),'avif':features.check('avif'),
             'scope':'CPU extraction and real codec round trips; no ComfyUI frontend or model/GPU acceptance'}
    output=Path(os.environ.get('METADATA_TEST_REPORT','metadata-test-result.json'))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)

if __name__=='__main__':
    main()
