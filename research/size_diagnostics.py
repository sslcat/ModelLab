"""Supplementary measured tables for the completed size study. No model execution."""

import json
from itertools import combinations
from pathlib import Path

from remote.receipt_scores import score
from research.size_design import MODELS, SEEDS
from research.size_results import arms_for, comparison, frontier, prefix


def adapter_inventory(doc, model):
    """Validate every LoRA matrix against pinned query/value geometry and file length."""
    width, layers, heads, kv = model["geometry"]
    rows = []
    for seed in SEEDS:
        path = (
            Path(doc["archive"])
            / prefix(seed)
            / "adapter"
            / "adapter_model.safetensors"
        )
        with path.open("rb") as stream:
            n = int.from_bytes(stream.read(8), "little")
            if not 0 < n < 4 * 1024**2:
                raise ValueError("Invalid adapter header.")
            header = json.loads(stream.read(n))
        keys = {k: v for k, v in header.items() if k != "__metadata__"}
        offsets = []
        parameters = 0
        types = set()
        for layer in range(layers):
            for target, out in (("q_proj", width), ("v_proj", width * kv // heads)):
                for letter, shape in (("A", [4, width]), ("B", [out, 4])):
                    name = f"base_model.model.model.layers.{layer}.self_attn.{target}.lora_{letter}.weight"
                    entry = keys.pop(name)
                    if entry["shape"] != shape:
                        raise ValueError("LoRA tensor shape mismatch.")
                    dtype = entry["dtype"]
                    bytes_per = {"F32": 4, "F16": 2, "BF16": 2}[dtype]
                    lo, hi = entry["data_offsets"]
                    if hi - lo != shape[0] * shape[1] * bytes_per:
                        raise ValueError("Invalid LoRA tensor extent.")
                    offsets.append((lo, hi))
                    parameters += shape[0] * shape[1]
                    types.add(dtype)
        end = 0
        for lo, hi in sorted(offsets):
            if lo != end:
                raise ValueError("Gap or overlap in serialized adapter.")
            end = hi
        if keys or path.stat().st_size != 8 + n + end:
            raise ValueError("Unexpected adapter tensors or trailing bytes.")
        rows.append(
            {
                "run": doc["id"],
                "model": model["model"],
                "seed": seed,
                "tensors": len(offsets),
                "tensor_parameters": parameters,
                "adapter_bytes": path.stat().st_size,
                "dtypes": ",".join(sorted(types)),
            }
        )
    return rows


def extra_tables(docs, studies):
    index = {d["config"]["model"]: d for d in docs}
    result = {
        name: []
        for name in (
            "adapter-inventory",
            "training-history",
            "frontiers",
            "seed-agreement",
            "matched-size",
            "primary-notes",
        )
    }
    for m in MODELS[:4]:
        doc = index[m["model"]]
        result["adapter-inventory"].extend(adapter_inventory(doc, m))
        for seed in SEEDS:
            path = Path(doc["archive"]) / prefix(seed) / "metrics.jsonl"
            for line in path.read_text(encoding="utf8").splitlines():
                if line:
                    result["training-history"].append(
                        {"model": m["model"], "seed": seed, **json.loads(line)}
                    )
    for partition in ("development", "locked"):
        big = arms_for(index[MODELS[3]["model"]], partition)
        for model in MODELS[:4]:
            arms = arms_for(index[model["model"]], partition)
            seed_rows = [arms[f"seed_{s}"] for s in SEEDS]
            same_hits = sum(
                len({row["correct"] for row in group}) == 1
                for group in zip(*seed_rows, strict=True)
            )
            same_text = sum(
                len({row["prediction"] for row in group}) == 1
                for group in zip(*seed_rows, strict=True)
            )
            for (s1, a), (s2, b) in combinations(zip(SEEDS, seed_rows, strict=True), 2):
                diff = comparison(a, b)
                result["seed-agreement"].append(
                    dict(
                        partition=partition,
                        model=model["model"],
                        reference_seed=s1,
                        candidate_seed=s2,
                        all_three_same_correctness=same_hits,
                        all_three_same_text=same_text,
                        **diff,
                    )
                )
            if model != MODELS[3]:
                for seed in SEEDS:
                    result["matched-size"].append(
                        dict(
                            partition=partition,
                            model=model["model"],
                            reference=MODELS[3]["model"],
                            seed=seed,
                            **comparison(big[f"seed_{seed}"], arms[f"seed_{seed}"]),
                        )
                    )
            if model == MODELS[1]:
                for a, b in zip(big["base"], arms["seed_42"], strict=True):
                    if (a["id"], a["expected"], a["prompt"]) != (
                        b["id"],
                        b["expected"],
                        b["prompt"],
                    ):
                        raise ValueError("Primary pair identity mismatch.")
                    result["primary-notes"].append(
                        {
                            "partition": partition,
                            "note_id": a["id"],
                            "template": a["template"],
                            "difficulty": a["difficulty"],
                            "unseen_pair": a["unseen_pair"],
                            "expected": a["expected"],
                            "base_7b": a["prediction"],
                            "specialized_1_5b": b["prediction"],
                            "base_correct": a["correct"],
                            "specialized_correct": b["correct"],
                            "base_errors": " | ".join(
                                score(a["prediction"], a["expected"])["errors"]
                            ),
                            "specialized_errors": " | ".join(
                                score(b["prediction"], b["expected"])["errors"]
                            ),
                        }
                    )
        together = []
        for condition in ("base", "tuned"):
            study = studies[partition + "-" + condition]
            together.extend(
                dict(p, id=condition + ":" + p["id"], condition=condition)
                for p in study["points"]
            )
        for condition in ("base", "tuned", "combined"):
            subset = [
                p
                for p in together
                if condition == "combined" or p["condition"] == condition
            ]
            for label, axis in (
                ("parameters", "parameters"),
                ("adapter_bytes", "adapter_bytes"),
                ("latency_ms", "latency"),
                ("session_cost_usd", "cost"),
                ("latency_and_session_cost", ("latency", "cost")),
            ):
                members = frontier(subset, axis)
                for p in subset:
                    result["frontiers"].append(
                        {
                            "partition": partition,
                            "comparison_set": condition,
                            "resource": label,
                            "model": p["model"],
                            "weights": p["condition"],
                            "accuracy": p["accuracy"],
                            "latency_ms": p["latency"],
                            "session_cost_usd": p["cost"],
                            "on_frontier": p["id"] in members,
                        }
                    )
    return result


def export():
    from research.export_results import atomic, csv_bytes, json_bytes
    from research.size_results import sessions
    from studio import archive, store
    from studio.cloud import one_drive

    folder = (one_drive() or store.home()) / "ModelLab" / "research" / "receipt-size-v1-results"
    docs = sessions(store.runs(), "mission3")
    evidence = json.loads((folder / "evidence.json").read_text(encoding="utf8"))
    if len(docs) != 4 or {d["id"] for d in docs} != {r["run"] for r in evidence}:
        raise ValueError(
            "Supplementary comparisons require the complete measured package."
        )
    for doc in docs:
        if (
            doc["status"] != "completed"
            or not doc.get("terminated")
            or not archive.verified(doc)
        ):
            raise ValueError("Unverified study archive.")
    studies = json.loads((folder / "comparisons.json").read_text(encoding="utf8"))
    tables = extra_tables(docs, studies)
    hashes = {}
    for name, rows in tables.items():
        target = folder / (name + ".csv")
        atomic(target, csv_bytes(rows))
        hashes[target.name] = archive.sumfile(target)
    atomic(folder / "diagnostic-checksums.json", json_bytes(hashes))
    return folder


if __name__ == "__main__":
    print(export())
