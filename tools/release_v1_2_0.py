"""One-time v1.2.0 coordinator; delegates publication to the existing release workflow.

This is CI tooling, not runtime code. It never overwrites tags/releases and never
reads a personal credential. The job-scoped GITHUB_TOKEN is used by gh.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import urllib.error
import urllib.request

REPO='ukr8b3g-cmyk/ComfyUI-save-webp-meta-node'
TAG='v1.2.0'

def read(path):
    request=urllib.request.Request('https://api.github.com/repos/'+REPO+path,headers={
        'Authorization':'Bearer '+os.environ['GH_TOKEN'], 'Accept':'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(request,timeout=30) as r:return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code==404:return None
        raise

def main():
    assert os.environ.get('GITHUB_REPOSITORY')==REPO
    assert os.environ.get('GITHUB_REF')=='refs/heads/main'
    version=tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if version!='1.2.0':
        print('Not the authorized one-time v1.2.0 version; no release action.');return
    sha=os.environ['GITHUB_SHA'];assert re.fullmatch('[0-9a-f]{40}',sha)
    assert read('/git/ref/heads/main')['object']['sha']==sha, 'Main advanced; stop before tagging'
    release=read('/releases/tags/'+TAG)
    if release is not None:
        print('Release already exists. No update or Registry republish attempted.');return
    ref=read('/git/ref/tags/'+TAG)
    if ref is not None:
        assert ref['object']=={'sha':sha,'type':'commit','url':f'https://api.github.com/repos/{REPO}/git/commits/{sha}'}, 'Existing tag differs; manual inspection required'
    else:
        subprocess.run(['gh','api','--method','POST',f'repos/{REPO}/git/refs','-f','ref=refs/tags/'+TAG,'-f','sha='+sha],check=True)
    assert read('/git/ref/tags/'+TAG)['object']['sha']==sha
    # workflow_dispatch is explicitly allowed to start a workflow from GITHUB_TOKEN.
    subprocess.run(['gh','workflow','run','release.yml','--repo',REPO,'--ref','main','-f','release_tag='+TAG],check=True)
    print('Created/verified exact source tag and dispatched existing guarded release workflow:',TAG,sha)

if __name__=='__main__':
    main()
