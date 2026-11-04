"""Tag v1.2.2 after CI and dispatch the existing stable release workflow."""

import os
from pathlib import Path
import re
import subprocess
import tomllib

from release_v1_2_1 import REPO, read

TAG = "v1.2.2"


def main():
    assert os.environ["GITHUB_REPOSITORY"] == REPO
    assert os.environ["GITHUB_REF"] == "refs/heads/main"
    version = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if "v" + version != TAG:
        print(f"Package version is {version}, skipping the one-time {TAG} release.")
        return
    sha = os.environ["GITHUB_SHA"]
    assert re.fullmatch("[0-9a-f]{40}", sha)
    assert read("/git/ref/heads/main")["object"]["sha"] == sha, "Main advanced before tagging"

    release = read("/releases/tags/" + TAG)
    ref = read("/git/ref/tags/" + TAG)
    if release is not None or ref is not None:
        assert ref and ref["object"]["type"] == "commit" and ref["object"]["sha"] == sha, "Existing tag points elsewhere"
        if release is not None:
            print(f"{TAG} is already published.")
            return
    else:
        subprocess.run([
            "gh", "api", "--method", "POST", f"repos/{REPO}/git/refs",
            "-f", f"ref=refs/tags/{TAG}", "-f", f"sha={sha}",
        ], check=True)
    assert read("/git/ref/tags/" + TAG)["object"]["sha"] == sha
    subprocess.run([
        "gh", "workflow", "run", "release.yml", "--repo", REPO, "--ref", "main",
        "-f", f"release_tag={TAG}", "-f", "changelog_notes=true",
    ], check=True)


if __name__ == "__main__":
    main()
