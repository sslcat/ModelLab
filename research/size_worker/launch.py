"""Use the established independent deadline and process-group supervisor for a complete model session."""

import hashlib
import json
import shutil

import launcher_core
from controlled import ROOT, SEEDS


def finish(status, error=""):
    files = {}
    for seed in SEEDS:
        d = ROOT if seed == SEEDS[0] else ROOT / "seeds" / str(seed)
        adapter = d / "adapter"
        if (
            status != "completed"
            and d.exists()
            and not (adapter / "adapter_model.safetensors").is_file()
        ):
            for checkpoint in sorted(
                (d / "checkpoints").glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]),
                reverse=True,
            ):
                names = ("adapter_config.json", "adapter_model.safetensors")
                if all((checkpoint / n).is_file() for n in names):
                    adapter.mkdir(exist_ok=True)
                    for name in names:
                        shutil.copyfile(checkpoint / name, adapter / name)
                    break
        for name in (
            "config.json",
            "metrics.jsonl",
            "environment.json",
            "lengths.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
        ):
            p = d / name
            if p.is_file():
                files[p.relative_to(ROOT).as_posix()] = p
    if (ROOT / "evaluation.json").is_file():
        files["evaluation.json"] = ROOT / "evaluation.json"
    entries = {}
    for name, p in files.items():
        with p.open("rb") as f:
            entries[name] = {
                "size": p.stat().st_size,
                "sha256": hashlib.file_digest(f, "sha256").hexdigest(),
            }
    tmp = ROOT / "manifest.tmp"
    tmp.write_text(json.dumps({"status": status, "detail": error, "files": entries}))
    tmp.replace(ROOT / "manifest.json")


if __name__ == "__main__":
    launcher_core.finish = finish
    launcher_core.main()
