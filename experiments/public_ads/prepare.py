"""Restore the pinned source checkpoints and runtime assets for public ADSs."""

from urllib.request import urlopen
from zipfile import ZipFile

from controllers.highway.public_agents import (
    CACHE,
    DT_REV,
    MC_REV,
    RL_AGENTS_REV,
    WEIGHTS,
    sha256,
    weight_bytes,
)
from controllers.highway.public_shield import MODELS

GITHUB = "https://raw.githubusercontent.com"
SOURCE_REPOS = {
    "online_mc": "fomaad/OnlineMC-SafetyShield",
}
DT_FILES = {
    "decision_transformer.py": "f68386420c0f52cf2a2a4f0abb7719c87591c6304179ff0ac8c49ad5102fd710",
    "model.py": "7ae8659cfff47d8193b40828a89cff668d8dcfa3e147a9cbffd3bfce6ef366cd",
    "trajectory_gpt2.py": "86127214e51b3fd422e2890e895eddf2f63141cb479964b7f6fc51b96c72ef4e",
}


def restore(path, url, digest):
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(url, timeout=60) as response:
            content = response.read()
    if sha256(content) != digest:
        raise ValueError(f"Source digest does not match for {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".download")
        pending.write_bytes(content)
        pending.replace(path)
    return path


def main():
    for sut, (filename, digest, revision) in WEIGHTS.items():
        source, filename_in_repo = filename.split("/", 1)
        url = f"{GITHUB}/{SOURCE_REPOS[source]}/{revision}/{filename_in_repo}"
        restore(CACHE / filename, url, digest)
        weight_bytes(sut)
    shield_sources = {
        **MODELS,
        "SafetyShield.py": "2127d54242d2089b105fd996eee2498cf978082999cc65be403a2c7edc6a27cc",
    }
    for filename, digest in shield_sources.items():
        restore(CACHE / "online_mc" / filename,
                f"{GITHUB}/{SOURCE_REPOS['online_mc']}/{MC_REV}/{filename}", digest)
    for filename, digest in DT_FILES.items():
        url = f"{GITHUB}/caixunshiren/Highway-Decision-Transformer/{DT_REV}/modules/{filename}"
        restore(CACHE / "highway_dt/modules" / filename, url, digest)
    source_archive = restore(
        CACHE / "rl-agents.zip",
        f"https://codeload.github.com/eleurent/rl-agents/zip/{RL_AGENTS_REV}",
        "70e1f1e9dce529d1b80a9ecee92309f1905d5e1b8dded107f08a19c41a0b2e7d",
    )
    # The archive is hash-pinned before extracting and contains a single fixed source tree.
    with ZipFile(source_archive) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = CACHE / "upstream" / member.filename
            content = archive.read(member)
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError(f"Extracted upstream source differs: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                pending = path.with_suffix(".download")
                pending.write_bytes(content)
                pending.replace(path)
    print("Pinned public ADS assets verified", flush=True)


if __name__ == "__main__":
    main()
