"""Recompute public research tables from the archived predictions and original generators."""

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from statistics import mean, stdev
from remote.receipt_scores import score, summarize
from research.reproduce import bundles
from research.size_design import locked_design
from research.study_results import matched
from studio.datasets import digest
from studio.demo import EVIDENCE, records, sizes
from research.size_results import frontier


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recompute():
    for name, h in json.loads((EVIDENCE / "sha256.json").read_text("utf8")).items():
        file = EVIDENCE / name
        require(
            file.parent == EVIDENCE
            and hashlib.sha256(file.read_bytes()).hexdigest() == h,
            "Digest mismatch: " + name,
        )
    sets = bundles()
    fingerprints = {k: digest(v) for k, v in sets.items()}
    design = locked_design()
    cells = {c["config"]["model"]: c for c in design["sessions"]}
    scores = []
    charges = []
    predictions = defaultdict(int)
    by_arm = {}
    resources = []
    for r in records().values():
        group = r["study"]
        rid = r["id"]
        require(r["bundle_hash"] == fingerprints[group], "Corpus mismatch: " + rid)
        b = r["actual"]
        require(
            abs(
                b["totalAmount"]
                - sum(b[k] for k in ("gpuAmount", "diskAmount", "cpuAmount"))
            )
            < 1e-6,
            "Billing mismatch",
        )
        charges.append(dict(session=rid, **b))
        for seed, item in r["seeds"].items():
            n = item["trainable"]
            if group == "size":
                cell = cells[r["config"]["model"]]
                require(
                    n == cell["expected_trainable"]
                    and r["execution"]["revision"] == cell["revision"],
                    "Size/revision mismatch",
                )
            else:
                require(
                    n == r["config"]["rank"] * 28 * (3 * 3584 + 512),
                    "7B geometry mismatch",
                )
            resources.append(
                dict(
                    session=rid,
                    seed=int(seed),
                    trainable=n,
                    adapter_bytes=item["adapter_bytes"],
                    adapter_sha256=item["adapter_sha256"],
                )
            )
        e = r["evaluation"]
        if not e:
            continue
        parts = (
            e["partitions"]
            if group != "pilot"
            else {"locked": {k: e[k] for k in ("base", "tuned")}}
        )
        for part, arms in parts.items():
            gold = sets[group]["test" if part == "locked" else "validation"]
            for arm, a in arms.items():
                rows = a["examples"]
                require(len(rows) == a["total"] == len(gold), "Incomplete evaluation")
                for row, note in zip(rows, gold, strict=True):
                    require(
                        (row["prompt"], row["expected"])
                        == (note["prompt"], note["completion"]),
                        "Reference mismatch",
                    )
                    require(
                        row["correct"]
                        is score(row["prediction"], row["expected"])["exact"],
                        "Wrong saved correctness",
                    )
                m = summarize(rows)
                require(m["exact"] == a["correct"], "Wrong group score")
                scores.append(
                    dict(
                        session=rid,
                        partition=part,
                        arm=arm,
                        n=len(rows),
                        valid=m["valid"],
                        schema=m["schema"],
                        exact=m["exact"],
                        normalized=m["content"],
                        **m["fields"],
                    )
                )
                by_arm[rid, part, arm] = rows
                predictions[group] += len(rows)
    primary = {
        p: matched(by_arm["size-7b", p, "base"], by_arm["size-1.5b", p, "seed_42"])
        for p in ("development", "locked")
    }
    size = []
    for m in ("0.5b", "1.5b", "3b", "7b"):
        rows = [
            r
            for r in scores
            if r["session"] == "size-" + m and r["partition"] == "locked"
        ]
        hits = {r["arm"]: r["exact"] for r in rows}
        vals = [hits["seed_" + str(s)] / 256 for s in (42, 73, 101)]
        size.append(dict(size=m, **hits, mean=mean(vals), sample_sd=stdev(vals)))
    require(
        dict(predictions) == dict(pilot=96, rank=2560, size=5120),
        "Unexpected generation count",
    )
    points = [
        dict(p, id=cond + "/" + p["id"])
        for cond in ("base", "tuned")
        for p in sizes(condition=cond)["points"]
    ]
    frontiers = {
        label: frontier(points, axes)
        for label, axes in (
            ("latency", "latency"),
            ("cost", "cost"),
            ("joint", ("latency", "cost")),
        )
    }
    return dict(
        counts=dict(predictions),
        fingerprints=fingerprints,
        charges=charges,
        frontiers=frontiers,
        actual_usd=sum(c["totalAmount"] for c in charges),
        primary=primary,
        size=size,
        adapters=resources,
    ), scores


def outputs():
    summary, scores = recompute()
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=list(scores[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(scores)
    return {
        "summary.json": (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        "scores.csv": target.getvalue().encode(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--write", action="store_true", help="Regenerate the compact derived tables"
    )
    a = p.parse_args()
    for name, blob in outputs().items():
        f = EVIDENCE / name
        if a.write:
            f.write_bytes(blob)
        else:
            require(f.read_bytes() == blob, "Derived public table mismatch: " + name)
    print(
        "Verified the three frozen corpora, 7,776 predictions, 17 adapter counts and ten session bills."
    )
