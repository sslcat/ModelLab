"""Publish local preregistration files or measured study tables into their own OneDrive folder."""

import argparse
import hashlib
from pathlib import Path

from research.export_results import atomic, csv_bytes, json_bytes
from research.size_corpus import LOCK, locked_bundle
from research.size_design import DESIGN, HERE, locked_design
from research.size_results import sessions, view
from studio import archive, lifecycle, store
from studio.cloud import one_drive
from studio.datasets import canon


def write_package(folder, files, immutable):
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in files.items()}
    files = dict(files, **{"sha256.json": json_bytes(hashes)})
    folder.mkdir(parents=True, exist_ok=True)
    if immutable:
        for name, blob in files.items():
            path = folder / name
            if path.exists() and path.read_bytes() != blob:
                raise ValueError(f"Existing preregistration differs at {name}.")
    for name, blob in files.items():
        path = folder / name
        if not path.exists() or path.read_bytes() != blob:
            atomic(path, blob)
    if any(archive.sumfile(folder / name) != h for name, h in hashes.items()):
        raise ValueError("Package checksum mismatch.")
    return folder


def export(measured=False):
    drive = one_drive() or store.home()
    design = locked_design()
    docs = sessions(store.runs(), "mission3")
    by_model = {d["config"]["model"]: d for d in docs}
    if len(docs) != 4 or set(by_model) != {
        s["config"]["model"] for s in design["sessions"]
    }:
        raise ValueError("Prepare all four registered model sessions first.")
    root = drive / "ModelLab" / "research"
    if not measured:
        files = {
            "design.json": DESIGN.read_bytes(),
            "split-lock.json": LOCK.read_bytes(),
            "models.json": (HERE / "size_models.json").read_bytes(),
            "plan.md": (store.ROOT / "docs/model-size-study.md").read_bytes(),
        }
        for part, rows in locked_bundle().items():
            files[part + ".jsonl"] = ("".join(canon(r) + "\n" for r in rows)).encode(
                "utf8"
            )
        for i, cell in enumerate(design["sessions"]):
            cfg = lifecycle.worker_config(by_model[cell["config"]["model"]])
            files[f"session-{i + 1}.json"] = json_bytes(dict(cfg, deadline=None))
        return write_package(root / "receipt-size-v1", files, True)
    if any(
        d["status"] != "completed" or not d.get("terminated") or not archive.verified(d)
        for d in docs
    ):
        raise ValueError(
            "Measured export requires all four verified, terminated sessions."
        )
    studies = {}
    points = []
    seeds = []
    groups = []
    contrasts = []
    for split in ("development", "locked"):
        for condition in ("base", "tuned"):
            study = view(docs, partition=split, condition=condition)
            studies[split + "-" + condition] = study
            for p in study["points"]:
                identity = {
                    "partition": split,
                    "condition": condition,
                    "model": p["model"],
                }
                points.append(
                    dict(
                        identity,
                        **{
                            k: v
                            for k, v in p.items()
                            if k not in {"model", "seed_rows"}
                        },
                    )
                )
                for r in p["seed_rows"]:
                    seeds.append(
                        dict(
                            identity,
                            **{
                                k: v
                                for k, v in r.items()
                                if k not in {"fields", "own_base", "groups"}
                            },
                            **r["fields"],
                        )
                    )
                    if r["own_base"]:
                        contrasts.append(
                            dict(
                                identity,
                                arm=r["arm"],
                                reference="same-size base",
                                **r["own_base"],
                            )
                        )
                    groups.extend(
                        dict(identity, arm=r["arm"], category=k, **v)
                        for k, v in r["groups"].items()
                    )
            contrasts.extend(
                dict(partition=split, condition=condition, reference="7B base", **p)
                for p in study["pairs"]
            )
    files = {
        "comparisons.json": json_bytes(studies),
        "models.csv": csv_bytes(points),
        "seeds.csv": csv_bytes(seeds),
        "difficulty.csv": csv_bytes(groups),
        "paired.csv": csv_bytes(contrasts),
        "design.json": DESIGN.read_bytes(),
        "evidence.json": json_bytes(
            [
                {
                    "run": d["id"],
                    "model": d["config"]["model"],
                    "pod": d["pod"],
                    "archive": d["archive"],
                    "manifest_sha256": archive.sumfile(
                        Path(d["archive"]) / "manifest.json"
                    ),
                    "posted": d.get("actual"),
                }
                for d in docs
            ]
        ),
    }
    return write_package(root / "receipt-size-v1-results", files, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", action="store_true")
    print(export(parser.parse_args().measured))
