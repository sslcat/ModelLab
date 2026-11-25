"""Verify a copied OneDrive Mission 3 archive using only Python's standard library."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from remote.receipt_scores import score


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def verified_file(root, name, expected):
    path = (root / name).resolve()
    require(
        path.is_relative_to(root.resolve()) and path.is_file(),
        f"Invalid archive path: {name}",
    )
    with path.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    require(digest == expected, f"Checksum mismatch: {name}")
    return path


def verify(root):
    research = root / "research"
    frozen = research / "receipt-size-v1"
    measured = research / "receipt-size-v1-results"
    for name, digest in read_json(frozen / "sha256.json").items():
        verified_file(frozen, name, digest)
    registered = read_json(Path(__file__).with_name("size-design.json"))
    require(
        read_json(frozen / "design.json")
        == registered
        == read_json(measured / "design.json"),
        "The archived designs differ from the registered design.",
    )
    release = read_json(measured / "release-sha256.json")
    require(
        {
            "findings.md",
            "cloud-closeout.json",
            "seeds.csv",
            "frontiers.csv",
            "size-comparison.png",
            "billing-history.jsonl",
        }
        <= release["reports"].keys(),
        "The release is missing required research outputs.",
    )
    for name, digest in release["reports"].items():
        verified_file(measured, name, digest)
    cells = {s["config"]["model"]: s for s in registered["sessions"]}
    evidence = read_json(measured / "evidence.json")
    require(
        len(evidence) == 4 and {r["model"] for r in evidence} == set(cells),
        "The release must contain four distinct registered models.",
    )
    require(
        len(release["experiments"]) == 4
        and {r["run"] for r in evidence} == set(release["experiments"]),
        "Run identifiers differ between release and export.",
    )
    with (measured / "seeds.csv").open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    scores = {(r["model"], r["partition"], r["arm"]): r for r in rows}
    require(len(rows) == len(scores) == 32, "Expected 32 model/partition/arm cells.")
    splits = {
        split: [
            json.loads(line)
            for line in (frozen / (split + ".jsonl"))
            .read_text(encoding="utf8")
            .splitlines()
        ]
        for split in ("train", "validation", "test")
    }
    canonical = json.dumps(
        splits,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    require(
        hashlib.sha256(canonical.encode("utf8")).hexdigest()
        == registered["corpus_sha256"],
        "Archived receipts differ from the preregistered corpus.",
    )
    corpus = {"development": splits["validation"], "locked": splits["test"]}
    count = 0
    adapters = 0
    for run in evidence:
        rid = run["run"]
        require(re.fullmatch("[0-9a-f]{32}", rid), "Invalid run identifier.")
        folder = root / "experiments" / rid
        supplements = release["experiments"][rid]
        require(
            {
                "manifest.json",
                "report.json",
                "billing.json",
                "recovery.json",
                "cache-evidence.jsonl",
                "observer-metrics.jsonl",
            }
            <= supplements.keys(),
            f"Missing session closeout files: {rid}",
        )
        for name, digest in supplements.items():
            verified_file(folder, name, digest)
        manifest = read_json(
            verified_file(folder, "manifest.json", run["manifest_sha256"])
        )
        require(manifest["status"] == "completed", f"Run did not complete: {rid}")
        require(
            len(manifest["files"]) == 19, f"Wrong number of worker artifacts: {rid}"
        )
        for name, entry in manifest["files"].items():
            path = verified_file(folder, name, entry["sha256"])
            require(path.stat().st_size == entry["size"], f"Wrong byte length: {name}")
        model = run["model"]
        cell = cells[model]
        for seed in (42, 73, 101):
            prefix = "" if seed == 42 else f"seeds/{seed}/"
            needed = (
                "config.json",
                "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors",
                "environment.json",
                "lengths.json",
                "metrics.jsonl",
            )
            require(
                all(prefix + n in manifest["files"] for n in needed),
                f"Missing seed artifacts: {rid}/{seed}",
            )
            config = read_json(folder / prefix / "config.json")
            expected = dict(
                cell["config"],
                seed=seed,
                revision=cell["revision"],
                worker_sha256=registered["workers"],
            )
            require(
                all(config.get(k) == v for k, v in expected.items()),
                f"Wrong seed configuration: {rid}/{seed}",
            )
            adapters += 1
        require("evaluation.json" in manifest["files"], f"Missing predictions: {rid}")
        evaluation = read_json(folder / "evaluation.json")
        require(
            read_json(folder / "report.json")["evaluation"] == evaluation,
            f"Report differs from evaluation: {rid}",
        )
        for part, gold in corpus.items():
            arms = evaluation["partitions"][part]
            require(
                set(arms) == {"base", "seed_42", "seed_73", "seed_101"},
                f"Wrong arms: {rid}/{part}",
            )
            for arm, group in arms.items():
                hits = dict.fromkeys(
                    ("correct", "schema", "item", "quantity", "site"), 0
                )
                for pred, note in zip(group["examples"], gold, strict=True):
                    require(
                        (pred["id"], pred["prompt"], pred["expected"])
                        == (note["id"], note["prompt"], note["completion"]),
                        f"Changed reference or note: {pred['id']}",
                    )
                    scored = score(pred["prediction"], note["completion"])
                    require(
                        scored["exact"] is pred["correct"],
                        f"Incorrect recorded score: {pred['id']}",
                    )
                    for k, v in dict(
                        correct=scored["exact"],
                        schema=scored["schema"],
                        **scored["fields"],
                    ).items():
                        hits[k] += int(v)
                    count += 1
                row = scores[model, part, arm]
                require(
                    int(row["n"]) == group["total"] == len(gold)
                    and group["correct"] == hits["correct"],
                    f"Incorrect denominator or group total: {model}/{part}/{arm}",
                )
                require(
                    all(int(row[k]) == n for k, n in hits.items()),
                    f"Export differs from predictions: {model}/{part}/{arm}",
                )
    require((adapters, count) == (12, 5120), "The archive is incomplete.")
    return adapters, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="ModelLab OneDrive archive root with research/ and experiments/",
    )
    n_adapters, n_predictions = verify(parser.parse_args().root)
    print(
        f"Verified four sessions, {n_adapters} adapters and {n_predictions:,} predictions."
    )
