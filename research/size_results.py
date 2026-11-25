"""Verified size-study artifacts and matched comparisons. All measurements originate on RunPod."""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from remote.receipt_scores import score, summarize
from research import size_corpus
from research.size_design import MODELS, SEEDS, check, locked_design, protocol
from research.study_results import matched
from studio.datasets import digest

SEED_FILES = (
    "config.json",
    "metrics.jsonl",
    "environment.json",
    "lengths.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
)


def prefix(seed):
    return "" if seed == SEEDS[0] else f"seeds/{seed}/"


def expected_files():
    return {"evaluation.json"} | {prefix(s) + n for s in SEEDS for n in SEED_FILES}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def verify_predictions(evaluation, data):
    if evaluation.get("protocol") != protocol() or set(
        evaluation.get("partitions", {})
    ) != {"development", "locked"}:
        raise ValueError("Size evaluation protocol mismatch.")
    for partition, split in (("development", "validation"), ("locked", "test")):
        arms = evaluation["partitions"][partition]
        if set(arms) != set(protocol()["arms"]):
            raise ValueError("Incomplete size evaluation arms.")
        for group in arms.values():
            rows = group["examples"]
            if len(rows) != len(data[split]) or group["total"] != len(rows):
                raise ValueError("Missing size evaluation rows.")
            hits = 0
            for row, original in zip(rows, data[split]):
                if (
                    any(
                        row.get(k) != original[k]
                        for k in (
                            "id",
                            "prompt",
                            "template",
                            "difficulty",
                            "unseen_template",
                            "unseen_pair",
                        )
                    )
                    or row["expected"] != original["completion"]
                ):
                    raise ValueError("Prediction differs from locked note identity.")
                if not (
                    math.isfinite(row["latency_ms"])
                    and row["latency_ms"] > 0
                    and 0 < row["generated_tokens"] <= 96
                    and row["input_tokens"] > 0
                    and math.isfinite(row["generation_peak_mib"])
                    and row["generation_peak_mib"] > 0
                ):
                    raise ValueError("Invalid generation measurement.")
                correct = score(row["prediction"], row["expected"])["exact"]
                if row["correct"] is not correct:
                    raise ValueError("Prediction score mismatch.")
                hits += int(correct)
            if hits != group["correct"]:
                raise ValueError("Group score mismatch.")


def training(folder):
    events = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text(encoding="utf8").splitlines()
        if line
    ]
    metrics = next((r for r in reversed(events) if "train_runtime" in r), {})
    env = read(folder / "environment.json")
    return {
        "trainable": env["trainable"],
        "adapter_bytes": (folder / "adapter/adapter_model.safetensors").stat().st_size,
        "train_seconds": metrics["train_runtime"],
        "train_tps": metrics["train_tokens_per_second"],
        "peak_mib": max(
            r["peak_allocated_mib"] for r in events if "peak_allocated_mib" in r
        ),
        "validation_loss": next(
            r["eval_loss"] for r in reversed(events) if "eval_loss" in r
        ),
    }


def verify_archive(doc, folder, evaluation, names):
    from research.export_results import adapter_header

    cell = check(doc)
    cfg = read(folder / "config.json")
    design = locked_design()
    if (
        set(names) != expected_files()
        or cfg["worker_sha256"] != design["workers"]
        or cfg["size_study"] != digest(design)
    ):
        raise ValueError("Incomplete or mismatched size study archive.")
    if (
        any(cfg.get(k) != v for k, v in cell["config"].items())
        or cfg["revision"] != cell["revision"]
        or cfg["protocol"] != protocol()
    ):
        raise ValueError(
            "Archived model configuration differs from the approved design."
        )
    for seed in SEEDS:
        sub = folder / prefix(seed)
        if read(sub / "config.json") != dict(cfg, seed=seed):
            raise ValueError("Per-seed configuration mismatch.")
        adapter = read(sub / "adapter/adapter_config.json")
        t = training(sub)
        if (
            adapter["r"] != 4
            or adapter["lora_alpha"] != 8
            or set(adapter["target_modules"]) != {"q_proj", "v_proj"}
        ):
            raise ValueError("Adapter geometry mismatch.")
        header = adapter_header(sub / "adapter/adapter_model.safetensors")
        if (
            t["trainable"] != cell["expected_trainable"]
            or header["serialized_parameters"] != t["trainable"]
        ):
            raise ValueError("Trainable and serialized adapter counts disagree.")
        if (
            any(not math.isfinite(v) or v < 0 for v in t.values())
            or t["train_seconds"] <= 0
        ):
            raise ValueError("Training telemetry missing.")
    verify_predictions(evaluation, size_corpus.locked_bundle())


def measurements(examples):
    summary = summarize(examples)
    groups = defaultdict(list)
    for e in examples:
        for label in (
            e.get("difficulty", e["template"]),
            "Unseen item-site pair" if e["unseen_pair"] else "Familiar item-site pair",
        ):
            groups[label].append(e["correct"])
    return {
        "n": len(examples),
        "correct": summary["exact"],
        "accuracy": summary["exact"] / len(examples),
        "fields": summary["fields"],
        "schema": summary["schema"],
        "latency": summary["latency_p50_ms"],
        "p95": summary["latency_p95_ms"],
        "generation_peak_mib": max(
            (e["generation_peak_mib"] for e in examples if "generation_peak_mib" in e),
            default=None,
        ),
        "generation_tps": 1000
        * sum(e["generated_tokens"] for e in examples)
        / sum(e["latency_ms"] for e in examples),
        "groups": {k: {"n": len(v), "correct": sum(v)} for k, v in groups.items()},
    }


def frontier(points, axis):
    """Maximize accuracy and minimize each selected resource. Retain exact ties."""
    axes = (axis,) if isinstance(axis, str) else tuple(axis)
    eligible = [p for p in points if all(p.get(k) is not None for k in axes)]
    return [
        p["id"]
        for p in eligible
        if not any(
            q["accuracy"] >= p["accuracy"]
            and all(q[k] <= p[k] for k in axes)
            and (q["accuracy"] > p["accuracy"] or any(q[k] < p[k] for k in axes))
            for q in eligible
        )
    ]


def sessions(documents, benchmark):
    if benchmark == "mission3":
        return [d for d in documents if d.get("size_study")]
    if benchmark == "historical64":
        return [
            d
            for d in documents
            if d.get("research_design") and d["config"]["rank"] == 4
        ]
    raise ValueError("Unknown size benchmark.")


def arms_for(doc, partition):
    result = doc["evaluation"]["partitions"][partition]
    if doc.get("size_study"):
        return {k: v["examples"] for k, v in result.items()}
    return {
        "base": result["base_constrained"]["examples"],
        "seed_42": result["tuned_constrained"]["examples"],
    }


def comparison(reference, examples):
    result = matched(reference, examples)
    left = {r["id"]: r for r in reference}
    by_family = defaultdict(list)
    for e in examples:
        by_family[e["template"]].append(
            int(e["correct"]) - int(left[e["id"]]["correct"])
        )
    total = result["gains"] - result["losses"]
    omitted = [
        (total - sum(values)) / (len(examples) - len(values))
        for values in by_family.values()
        if len(values) < len(examples)
    ]
    result["leave_family_out"] = [min(omitted), max(omitted)] if omitted else None
    return result


def view(documents, benchmark="mission3", partition="locked", condition="tuned"):
    if partition not in ("locked", "development") or condition not in ("tuned", "base"):
        raise ValueError("Unknown comparison.")
    docs = sessions(documents, benchmark)
    points = []
    pairs = []
    completed = [
        d
        for d in docs
        if d["status"] == "completed" and d.get("archived") and d.get("terminated")
    ]
    reference = next(
        (
            arms_for(d, partition)["base"]
            for d in completed
            if d["config"]["model"] == MODELS[3]["model"]
        ),
        None,
    )
    for d in completed:
        model = next(m for m in MODELS if m["model"] == d["config"]["model"])
        arms = arms_for(d, partition)
        rows = []
        for arm, examples in arms.items():
            if (condition == "base") != (arm == "base"):
                continue
            seed = None if arm == "base" else int(arm.split("_")[1])
            m = measurements(examples)
            resource = (
                (d["seeds"][str(seed)]["measurements"] if "public_resources" in d else training(Path(d["archive"]) / prefix(seed)))
                if seed
                else {
                    "trainable": 0,
                    "adapter_bytes": 0,
                    "train_seconds": None,
                    "train_tps": None,
                    "peak_mib": None,
                    "validation_loss": None,
                }
            )
            rows.append(
                dict(
                    arm=arm,
                    seed=seed,
                    own_base=comparison(arms["base"], examples) if seed else None,
                    **m,
                    **resource,
                )
            )
            if reference is not None:
                pairs.append(
                    dict(
                        model=model["model"], arm=arm, **comparison(reference, examples)
                    )
                )
        if not rows:
            continue
        center = mean(r["accuracy"] for r in rows)
        trainable = mean(r["trainable"] for r in rows)
        base = arms["base"]
        points.append(
            {
                "id": d["id"],
                "model": model["model"],
                "parameters": model["total"],
                "accuracy": center,
                "seed_rows": rows,
                "seed_sd": stdev(r["accuracy"] for r in rows)
                if len(rows) > 1
                else None,
                "adapter_bytes": mean(r["adapter_bytes"] for r in rows),
                "latency": mean(r["latency"] for r in rows),
                "trainable": trainable,
                "improvement_per_million": (
                    center - sum(r["correct"] for r in base) / len(base)
                )
                * 100
                * 1e6
                / trainable
                if trainable
                else None,
                "cost": (d.get("actual") or {}).get("totalAmount"),
            }
        )
    return {
        "benchmark": benchmark,
        "partition": partition,
        "condition": condition,
        "points": sorted(points, key=lambda p: p["parameters"]),
        "pairs": pairs,
        "frontiers": {
            axis: frontier(points, axis)
            for axis in ("parameters", "adapter_bytes", "latency", "cost")
        },
        "joint_frontier": frontier(points, ("latency", "cost")),
        "sessions": [
            {"id": d["id"], "model": d["config"]["model"], "status": d["status"]}
            for d in docs
        ],
    }
