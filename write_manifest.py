#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest without including the manifest itself."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "MANIFEST_SHA256.txt"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path != OUTPUT
    )
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.name} with {len(lines)} entries")


if __name__ == "__main__":
    main()
