from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_promptfoo_scripts_put_repo_root_on_python_path() -> None:
    scripts = json.loads((ROOT / "package.json").read_text())["scripts"]
    for name, command in scripts.items():
        if "promptfoo eval" in command:
            assert "PYTHONPATH=." in command, f"{name}: the provider imports evals.*"
