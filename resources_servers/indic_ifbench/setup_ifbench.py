# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the upstream AllenAI IFBench package (`ifbench`) importable for the checkers in evaluation.py.

Like resources_servers/ifbench/setup_ifbench.py, the repo is cloned at a pinned commit into a local `.ifbench/`
folder and put first on sys.path. A pip install would not do: the folder resources_servers/ifbench (the English
server) shadows a package named `ifbench` whenever resources_servers/ is on sys.path, as under pytest.

Set IFBENCH_REPO to an existing clone to skip the download.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


# Newer than the English server's pin: includes upstream's later verifier fixes, which the checkers build on.
IFBENCH_COMMIT = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"  # pragma: allowlist secret
IFBENCH_REPO = "https://github.com/allenai/IFBench.git"
IFBENCH_DIR = Path(__file__).parent / ".ifbench"
INSTALL_MARKER = IFBENCH_DIR / ".installed"


def _use(path: Path) -> None:
    """Put `path` (a clone containing the `ifbench/` package) first, dropping a shadowing module already imported."""
    path_str = str(path.resolve())
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
    module = sys.modules.get("ifbench")
    if module is not None and not str(getattr(module, "__file__", "")).startswith(path_str):
        for name in [n for n in sys.modules if n == "ifbench" or n.startswith("ifbench.")]:
            del sys.modules[name]


def ensure_ifbench() -> None:
    """Clone IFBench at the pinned commit once, then make it importable."""
    local = os.environ.get("IFBENCH_REPO")
    if local:
        _use(Path(local))
        return
    if not INSTALL_MARKER.exists():
        if IFBENCH_DIR.exists():
            shutil.rmtree(IFBENCH_DIR)
        print(f"Cloning IFBench @ {IFBENCH_COMMIT} ...")
        subprocess.run(["git", "clone", "--quiet", IFBENCH_REPO, str(IFBENCH_DIR)], check=True)
        subprocess.run(["git", "-C", str(IFBENCH_DIR), "checkout", "--quiet", IFBENCH_COMMIT], check=True)
        INSTALL_MARKER.touch()
    _use(IFBENCH_DIR)


if __name__ == "__main__":
    ensure_ifbench()
