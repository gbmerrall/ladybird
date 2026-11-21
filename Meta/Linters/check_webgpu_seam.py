#!/usr/bin/env python3

# Copyright (c) 2026-present, the Ladybird developers.
#
# SPDX-License-Identifier: BSD-2-Clause

"""Keeps the library that implements WebGPU behind the compositor's Backend trait.

Ladybird's WebGPU is its own down to `Services/Compositor/WebGPU/src/backend.rs`: the page-side API,
the wire protocol, the IPC, and the canvas plumbing. Only the `wgpu_backend` module executes the
commands with wgpu, so that another implementation (Dawn, or Ladybird's own) can replace it without touching
anything else. This checks that no other source names wgpu or its parts, and that no manifest but
the host's depends on them.

The check is textual: it looks for the crate names, not for what a type alias might hide.
"""

import pathlib
import re
import subprocess
import sys

BACKEND = "Services/Compositor/WebGPU/src/wgpu_backend/"
HOST_MANIFEST = "Services/Compositor/WebGPU/Cargo.toml"

# The crates of the wgpu project, as Rust paths and as dependency names.
CRATES = ["wgpu", "wgpu_core", "wgpu_hal", "wgpu_types", "naga"]
SOURCE_PATTERN = re.compile(r"\b(" + "|".join(CRATES) + r")::")
MANIFEST_PATTERN = re.compile(r"^\s*(" + "|".join(name.replace("_", "[-_]") for name in CRATES) + r")\s*=")


def tracked_files(*patterns):
    return subprocess.run(
        ["git", "ls-files", "--", *patterns], capture_output=True, text=True, check=True
    ).stdout.split()


def main():
    failures = []
    for file in tracked_files("*.rs", "*.cpp", "*.h", "*.mm"):
        if file.startswith(BACKEND):
            continue
        for number, line in enumerate(pathlib.Path(file).read_text(errors="replace").splitlines(), 1):
            if SOURCE_PATTERN.search(line):
                failures.append(f"{file}:{number}: names a wgpu crate outside {BACKEND}: {line.strip()}")
    for manifest in tracked_files("*Cargo.toml"):
        if manifest == HOST_MANIFEST:
            continue
        for number, line in enumerate(pathlib.Path(manifest).read_text().splitlines(), 1):
            if MANIFEST_PATTERN.search(line):
                failures.append(f"{manifest}:{number}: depends on a wgpu crate: {line.strip()}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
