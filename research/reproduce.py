"""Create reproducible input files or free local drafts. Never opens a cloud connection."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from research import receipts, size_corpus, prepare, size_design
from studio import store
from studio.datasets import canon, digest, dispatch_rows, validate


def sample():
    return json.loads(
        gzip.decompress((store.ROOT / "evidence/pilot.json.gz").read_bytes())
    )


def pilot_sources():
    base = store.ROOT / "remote"
    earlier = store.ROOT / "research" / "pilot_worker"
    return {
        n: (earlier if n in ("launch.py", "train.py") else base) / n
        for n in sample()["execution"]["worker_sha256"]
    }


def pilot():
    reference = sample()
    paths = pilot_sources()
    hashes = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()}
    if hashes != reference["execution"]["worker_sha256"]:
        raise ValueError("Pilot worker differs from the recorded revision")
    store.init()
    with store.db(True) as con:
        for run in store.runs(con):
            if run.get("pilot_snapshot"):
                return run["id"]
        run = store.new_draft(reference["config"], con)
        run.update(pilot_snapshot=True, research_worker_sha256=hashes)
        store.put(con, run)
        return run["id"]


def bundles():
    _, data = validate(dispatch_rows(), 42, 0.15)
    data["test"] = dispatch_rows(True)
    return {
        "pilot": data,
        "rank": receipts.locked_bundle(),
        "size": size_corpus.locked_bundle(),
    }


def export(folder):
    for study, splits in bundles().items():
        directory = folder / study
        directory.mkdir(parents=True, exist_ok=True)
        for split, rows in splits.items():
            (directory / (split + ".jsonl")).write_bytes(
                ("".join(canon(row) + "\n" for row in rows)).encode()
            )
        (directory / "fingerprint.txt").write_text(
            digest(splits) + "\n", encoding="ascii"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--datasets", type=Path, metavar="DIRECTORY")
    g.add_argument("--drafts", choices=["pilot", "rank", "size"])
    args = parser.parse_args()
    if args.datasets:
        export(args.datasets)
    else:
        print(
            {"pilot": pilot, "rank": prepare.prepare, "size": size_design.prepare}[
                args.drafts
            ]()
        )
