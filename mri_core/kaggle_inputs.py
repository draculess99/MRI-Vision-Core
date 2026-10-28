"""Deterministic, depth-limited discovery of Kaggle input packages (no recursive scan of the competition's DICOM tree).

Kaggle mounts a dataset at /kaggle/input/<slug>/ and, in some layouts, at a deeper path such as /kaggle/input/datasets/<owner>/<slug>/.
The search therefore looks at a few directory levels only and never descends into the competition's `train_series` / `test_series` trees.
Diagnostics contain directory names and required-file presence only - never file contents, study IDs or credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

PRUNE = ("train_series", "test_series")


class InputsError(RuntimeError):
    def __init__(self, code: str, message: str, diagnostics: Optional[dict] = None):
        super().__init__(message)
        self.code, self.diagnostics = code, diagnostics or {}


def find_named_files(root: Path, name: str, max_depth: int = 3, prune: Sequence[str] = PRUNE) -> list:
    """Paths of files called `name` at most `max_depth` directories below root (root itself is depth 0)."""
    root = Path(root)
    found, frontier = [], [root]
    for depth in range(max_depth + 1):
        nxt = []
        for directory in frontier:
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                if entry.is_file() and entry.name == name:
                    found.append(Path(entry.path))
                elif entry.is_dir() and entry.name not in prune and depth < max_depth:
                    nxt.append(Path(entry.path))
        frontier = nxt
    return sorted(found)


def mount_listing(root: Path, depth: int = 2, limit: int = 40, prune: Sequence[str] = PRUNE) -> dict:
    """Directory names (and up to 8 entry names each) under the input root, for diagnosis. Never reads file contents."""
    root, out = Path(root), {}
    frontier = [root]
    for d in range(depth):
        nxt = []
        for directory in frontier:
            try:
                entries = sorted(os.scandir(directory), key=lambda e: e.name)
            except OSError:
                continue
            for entry in entries[:limit]:
                if entry.is_dir() and entry.name not in prune:
                    path = Path(entry.path)
                    try:
                        names = sorted(e.name for e in os.scandir(path))[:8]
                    except OSError:
                        names = []
                    out[str(path.relative_to(root))] = names
                    nxt.append(path)
        frontier = nxt
        if len(out) >= limit:
            break
    return dict(list(out.items())[:limit])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_inputs_package(root: Path, kind: str, *, max_depth: int = 3, override: Optional[Path] = None) -> tuple[Path, dict]:
    """Find exactly one directory holding a PROVENANCE.json of the given kind, verify every listed file's SHA-256, return (dir, provenance)."""
    root = Path(root)
    candidates = [Path(override) / "PROVENANCE.json"] if override else find_named_files(root, "PROVENANCE.json", max_depth)
    matching = []
    for path in candidates:
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("kind") == kind:
                matching.append(path)
        except (OSError, ValueError):
            continue
    diag = {"mounts": mount_listing(root) if root.is_dir() else {}, "provenance_files_seen": [str(p.relative_to(root)) if root in p.parents else str(p) for p in candidates],
            "matching_kind": [str(p.parent.name) for p in matching]}
    if not matching:
        raise InputsError("missing", f"no PROVENANCE.json of kind {kind!r} found within {max_depth} levels of the input root", diag)
    if len(matching) > 1:
        raise InputsError("duplicate", f"{len(matching)} PROVENANCE.json files of kind {kind!r} found (expected exactly one)", diag)
    folder = matching[0].parent
    prov = json.loads(matching[0].read_text(encoding="utf-8"))
    presence = {name: (folder / name).is_file() for name in prov["files"]}
    diag["required_file_presence"] = presence
    missing = [n for n, ok in presence.items() if not ok]
    if missing:
        raise InputsError("file_missing", f"files listed in PROVENANCE.json are absent: {missing}", diag)
    bad = [name for name, info in prov["files"].items() if sha256_file(folder / name) != info["sha256"]]
    if bad:
        raise InputsError("hash_mismatch", f"files do not match PROVENANCE.json hashes: {bad}", diag)
    return folder, prov


def download_verified(url: str, dest: Path, expected_sha256: str, *, connect_timeout: float = 30.0, total_timeout: float = 300.0, max_bytes: int = 200_000_000,
                      opener=None) -> dict:
    """Fetch `url` (HTTPS only) to `dest`, bounded in time and size, and keep the file only if its full SHA-256 equals `expected_sha256`.

    Fails closed: any problem raises InputsError and leaves no file at `dest`. A file already at `dest` is re-verified, never trusted. Returns {"bytes", "sha256", "downloaded"}.
    """
    import time
    import urllib.request
    from urllib.parse import urlparse

    if urlparse(url).scheme != "https":
        raise InputsError("insecure_url", f"refusing a non-HTTPS URL: {url}", {})
    dest = Path(dest)
    if dest.is_file():
        if sha256_file(dest) == expected_sha256:
            return {"bytes": dest.stat().st_size, "sha256": expected_sha256, "downloaded": False}
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + f".{os.getpid()}.part")
    opener = opener or (lambda u: urllib.request.urlopen(u, timeout=connect_timeout))
    digest, size, started = hashlib.sha256(), 0, time.monotonic()
    try:
        with opener(url) as response, open(temp, "wb") as handle:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                size += len(chunk)
                if size > max_bytes:
                    raise InputsError("too_large", f"download exceeded {max_bytes} bytes", {})
                if time.monotonic() - started > total_timeout:
                    raise InputsError("timeout", f"download exceeded {total_timeout} s", {})
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise InputsError("hash_mismatch", f"downloaded file has SHA-256 {digest.hexdigest()}, expected {expected_sha256}", {"bytes": size})
        temp.replace(dest)
    except InputsError:
        temp.unlink(missing_ok=True)
        raise
    except Exception as error:  # network, DNS, TLS, HTTP, disk
        temp.unlink(missing_ok=True)
        raise InputsError("download_failed", f"{type(error).__name__}: {error}", {}) from error
    return {"bytes": size, "sha256": expected_sha256, "downloaded": True}
