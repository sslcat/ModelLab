"""Materialize free, immutable size-study sessions. This module never provisions a pod."""

import hashlib
import json
from pathlib import Path

from research import size_corpus
from research.size_worker.controlled import SEEDS, protocol
from studio import store
from studio.cloud import estimate
from studio.config import IMAGE, Config
from studio.datasets import canon, digest

HERE = Path(__file__).parent
MODELS = json.loads((HERE / "size_models.json").read_text(encoding="utf8"))
LIMITS = (60, 75, 90, 120)
GPU = "NVIDIA GeForce RTX 4090"
CEILING = 0.89
DESIGN = HERE / "size-design.json"


def sources():
    remote = store.ROOT / "remote"
    dedicated = HERE / "size_worker"
    return {
        "requirements.txt": remote / "requirements.txt",
        "boot.sh": remote / "boot.sh",
        "launcher_core.py": remote / "launch.py",
        "trainer_core.py": remote / "train.py",
        "receipt_scores.py": remote / "receipt_scores.py",
        **{n: dedicated / n for n in ("launch.py", "train.py", "controlled.py")},
    }


def geometry(model):
    width, layers, heads, kv = model["geometry"]
    return 4 * layers * (3 * width + width // heads * kv)


def configurations(meta):
    return [
        Config(
            name=f"Size study · {m['model'].split('/')[-1]} · three seeds",
            dataset=meta["id"],
            model=m["model"],
            minutes=minutes,
            rank=4,
            alpha=8,
            seed=SEEDS[0],
            seq_len=512,
            gpu=GPU,
        ).model_dump()
        for m, minutes in zip(MODELS, LIMITS)
    ]


def specification():
    bundle = size_corpus.locked_bundle()
    return {
        "version": size_corpus.VERSION,
        "protocol": protocol(),
        "image": IMAGE,
        "corpus_sha256": digest(bundle),
        "max_rate_usd": CEILING,
        "analysis": {
            "primary_model": MODELS[1]["model"],
            "primary_seed": 42,
            "reference": MODELS[3]["model"],
            "endpoint": "paired strict complete record",
            "interval": "Bonferroni exact gain/loss 95%",
            "template_sensitivity": "leave one template family out",
            "seed_variation": "sample SD of seed accuracies",
            "frontier": "maximize accuracy, minimize the selected resource; retain ties",
        },
        "workers": {
            n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in sources().items()
        },
        "sessions": [
            {
                "config": cfg,
                "revision": m["revision"],
                "base_parameters": m["total"],
                "expected_trainable": geometry(m),
                "max_usd": estimate(CEILING, cfg["minutes"])["max_usd"],
            }
            for cfg, m in zip(configurations(size_corpus.metadata(bundle)), MODELS)
        ],
    }


def locked_design():
    record = json.loads(DESIGN.read_text(encoding="utf8"))
    if record != specification():
        raise ValueError("Mission 3 differs from its frozen experiment design.")
    return record


def check(doc):
    design = locked_design()
    cell = next((x for x in design["sessions"] if x["config"] == doc["config"]), None)
    if (
        not cell
        or doc.get("size_study") != digest(design)
        or doc["bundle_hash"] != design["corpus_sha256"]
    ):
        raise ValueError(
            "Size study session identity differs from its registered plan."
        )
    return cell


def prepare():
    design = locked_design()
    data = size_corpus.metadata(size_corpus.locked_bundle())
    store.init()
    with store.db(True) as con:
        con.execute(
            "INSERT OR IGNORE INTO datasets VALUES(?,?,?)",
            (data["id"], canon(data), b""),
        )
        existing = [d for d in store.runs(con) if d.get("size_study")]
        ids = []
        for cell in design["sessions"]:
            cfg = cell["config"]
            matches = [d for d in existing if d["config"]["model"] == cfg["model"]]
            if len(matches) > 1:
                raise ValueError("Ambiguous size study sessions.")
            if matches:
                doc = matches[0]
                check(doc)
            else:
                doc = store.new_draft(cfg, con)
                doc["size_study"] = digest(design)
                store.put(con, doc)
            ids.append(doc["id"])
        return ids


if __name__ == "__main__":
    for key in prepare():
        print(key, store.run(key)["config"]["name"])
