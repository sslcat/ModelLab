"""Three independent trainer processes sharing only the same on-pod checkpoint cache."""

import json
import os
import subprocess
import sys


def main():
    if sys.platform != "linux" or not os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("RunPod Linux only.")
    import trainer_core
    from controlled import ROOT, SEEDS, protocol

    cfg = json.loads((ROOT / "config.json").read_text())
    if (
        cfg["protocol"] != protocol()
        or cfg["rank"] != 4
        or cfg["alpha"] != 8
        or cfg["seed"] != SEEDS[0]
    ):
        raise ValueError("Session configuration differs from the size study.")
    if len(sys.argv) == 1:
        for seed in SEEDS:
            subprocess.run([sys.executable, "-u", __file__, str(seed)], check=True)
        return
    seed = int(sys.argv[1])
    if seed not in SEEDS:
        raise ValueError("Unregistered seed.")
    os.environ["MODELLAB_TRAIN_SEED"] = str(seed)
    folder = ROOT if seed == SEEDS[0] else ROOT / "seeds" / str(seed)
    if folder != ROOT:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "data.json").symlink_to(ROOT / "data.json")
        (folder / "config.json").write_text(
            json.dumps(dict(cfg, seed=seed), ensure_ascii=False)
        )
    trainer_core.ROOT = folder
    trainer_core.main()


if __name__ == "__main__":
    main()
