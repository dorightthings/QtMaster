#!/usr/bin/env python3
"""Export unchanged sampler bytes into an explicitly named, non-overwriting tar.

No unpickling occurs. A source map, if supplied, is local-only and must not be
committed because it can contain machine-specific source paths.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from datetime import datetime, timezone

SOURCE_SUBDIRS = {
    "csi300": "csi300_author_fix_full_v2",
    "csi800_direct": "csi800_author_fix_full_v2",
}


class HashingReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        value = self.handle.read(size)
        self.digest.update(value)
        return value


def add_json(archive: tarfile.TarFile, name: str, value) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def add_file(archive: tarfile.TarFile, source: Path, name: str, expected=None) -> dict:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected a regular, non-symlink source file: {source}")
    size = source.stat().st_size
    if expected and size != int(expected["bytes"]):
        raise ValueError(f"Source byte-size mismatch: {name}")
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    with source.open("rb") as handle:
        reader = HashingReader(handle)
        archive.addfile(info, reader)
        digest = reader.digest.hexdigest()
    if expected and digest != expected["sha256"]:
        raise ValueError(f"Source SHA256 mismatch: {name}")
    return {"path": name, "bytes": size, "sha256": digest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path, help="Root containing portable dataset folders or original rebuild folders")
    source.add_argument("--source-map", type=Path, help="Local JSON: dataset -> split -> absolute or map-relative source file")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "manifest.json")
    parser.add_argument("--provider-root", type=Path, help="Optional complete Qlib provider directory")
    parser.add_argument("--output", type=Path, required=True, help="New .tar or .tar.gz path; never overwritten")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qtmaster-data-v1":
        parser.error("Unsupported manifest schema")
    output = args.output.absolute()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
        parser.error("Output or .partial file already exists; choose a new output path")
    if not output.parent.is_dir():
        parser.error("Output parent directory must already exist")
    if not (output.name.endswith(".tar") or output.name.endswith(".tar.gz")):
        parser.error("Output must end in .tar or .tar.gz")
    source_map = json.loads(args.source_map.read_text()) if args.source_map else None
    entries = []
    for dataset, definition in manifest["datasets"].items():
        for stage in ("train", "valid", "test"):
            spec = definition["files"][stage]
            relative = Path(spec["path"])
            if relative.is_absolute() or ".." in relative.parts:
                parser.error("Manifest paths must be safe relative paths")
            if source_map is not None:
                original = Path(source_map[dataset][stage])
                if not original.is_absolute():
                    original = args.source_map.resolve().parent / original
            else:
                original = args.source_root / relative
                if not original.is_file():
                    original = args.source_root / SOURCE_SUBDIRS[dataset] / spec["source_filename"]
            if original.is_symlink() or not original.is_file():
                parser.error(f"Missing or symlink source file: {original}")
            if original.stat().st_size != spec["bytes"]:
                parser.error(f"Source size mismatch for {dataset}/{stage}")
            entries.append((original, f"data/{relative.as_posix()}", spec))
    provider_entries = []
    if args.provider_root:
        provider = args.provider_root.resolve()
        if not provider.is_dir() or not (provider / "calendars" / "day.txt").is_file():
            parser.error("provider-root must be a complete Qlib provider with calendars/day.txt")
        try:
            output.resolve().relative_to(provider)
        except ValueError:
            pass
        else:
            parser.error("Output must not be inside provider-root")
        for path in sorted(provider.rglob("*")):
            if path.is_symlink():
                parser.error("Provider contains symlinks; supply a materialized provider")
            if path.is_file():
                provider_entries.append((path, f"data/qlib_provider/{path.relative_to(provider).as_posix()}"))
    records = []
    try:
        with partial.open("xb") as destination:
            mode = "w:gz" if output.name.endswith(".gz") else "w"
            with tarfile.open(fileobj=destination, mode=mode) as archive:
                add_json(archive, "data/manifest.json", manifest)
                for original, name, spec in entries:
                    records.append(add_file(archive, original, name, spec))
                    print(f"Verified and archived {name}", flush=True)
                for original, name in provider_entries:
                    records.append(add_file(archive, original, name))
                bundle_manifest = {
                    "schema_version": "qtmaster-data-bundle-v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "data_manifest_sha256": hashlib.sha256(
                        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
                    ).hexdigest(),
                    "files": records,
                }
                add_json(archive, "data/bundle_manifest.json", bundle_manifest)
            destination.flush()
            os.fsync(destination.fileno())
        # Hard-link publication is atomic and refuses an existing destination.
        os.link(partial, output)
        partial.unlink()
    except BaseException:
        print(f"Export did not complete. Partial bytes, if any, remain at {partial}; no existing output was overwritten.")
        raise
    print(f"Created {output} ({output.stat().st_size} bytes); {len(records)} data files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
