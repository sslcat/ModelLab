"""Derived comparison views. Never write derived scores into an archived experiment."""

import json
import math
from remote.controlled import ARMS
from remote.receipt_scores import FENCE, FIELDS, score, summarize
from studio.cloud import CloudError

PILOT_BUNDLE = "8e0f32c1c3586747a0225eb9fe7c3be4674a015423576d1204b55a5d5e351ce7"


def verify_evaluation(config, evaluation):
    from research.receipts import locked_bundle

    expected = locked_bundle()
    if not evaluation or config["protocol"] != evaluation.get("protocol"):
        raise CloudError("Research evaluation protocol missing or changed.")
    for partition, part in (("development", "validation"), ("locked", "test")):
        output = evaluation.get("partitions", {}).get(partition, {})
        if set(output) != set(ARMS):
            raise CloudError("Research evaluation arm missing.")
        for arm in ARMS:
            result = output[arm]
            examples = result.get("examples", [])
            if result.get("total") != len(expected[part]) or len(examples) != len(
                expected[part]
            ):
                raise CloudError("Incomplete research partition.")
            for example, row in zip(examples, expected[part]):
                if (
                    any(
                        example.get(k) != row[k]
                        for k in (
                            "id",
                            "prompt",
                            "template",
                            "unseen_template",
                            "unseen_pair",
                        )
                    )
                    or example.get("expected") != row["completion"]
                ):
                    raise CloudError("Research prediction identity mismatch.")
                latency = example.get("latency_ms")
                if (
                    not isinstance(latency, (int, float))
                    or not math.isfinite(latency)
                    or latency < 0
                ):
                    raise CloudError("Research latency missing.")
            measured = summarize(examples)
            if (
                result.get("metrics") != measured
                or result.get("correct") != measured["exact"]
            ):
                raise CloudError(
                    "Research scores differ from the recorded predictions."
                )


def analysis(doc, partition=None, arm=None):
    ev = doc.get("evaluation") or {}
    partitions = ev.get("partitions")
    if partitions is None:
        name = ev.get("split", "evaluation")
        partitions = (
            {name: {k: ev[k] for k in ("base", "tuned") if ev.get(k)}} if ev else {}
        )
    available = list(partitions)
    selected = (
        partition
        if partition in available
        else ("locked" if "locked" in available else next(iter(available), None))
    )
    arms = partitions.get(selected, {})
    chosen = (
        arm if arm in arms else ("tuned" if "tuned" in arms else next(iter(arms), None))
    )
    result = arms.get(chosen, {})
    raw = result.get("examples", [])
    research = bool(ev.get("protocol"))
    rows = []
    receipt = bool(raw)
    for record in raw:
        try:
            ref = json.loads(record["expected"])
            receipt = (
                receipt
                and isinstance(ref, dict)
                and set(ref) == set(FIELDS)
                and type(ref["quantity"]) is int
                and all(isinstance(ref[k], str) for k in ("item", "site"))
            )
        except (ValueError, TypeError):
            receipt = False
    for record in raw:
        row = dict(record)
        if doc.get("bundle_hash") == PILOT_BUNDLE:
            row["unseen_template"] = True
            row["unseen_pair"] = False
        row["scores"] = score(row["prediction"], row["expected"]) if receipt else None
        if receipt:
            fence = FENCE.fullmatch(row["prediction"].strip())
            row["content_scores"] = (
                score(fence[1], row["expected"]) if fence else row["scores"]
            )
        rows.append(row)
    summary = summarize(rows) if receipt else None
    if summary:
        summary["content_fields"] = {
            k: sum(row["content_scores"]["fields"][k] for row in rows) for k in FIELDS
        }
    groups = {}
    if research and rows:
        for template in (False, True):
            for pair in (False, True):
                subset = [
                    r
                    for r in rows
                    if r["unseen_template"] == template and r["unseen_pair"] == pair
                ]
                if subset:
                    label = (
                        ("New" if template else "Familiar")
                        + " template / "
                        + ("new" if pair else "familiar")
                        + " pair"
                    )
                    groups[label] = dict(
                        n=len(subset), exact=sum(r["scores"]["exact"] for r in subset),
                        content=sum(r["content_scores"]["exact"] for r in subset),
                        valid=sum(r["scores"]["valid"] for r in subset),
                        schema=sum(r["scores"]["schema"] for r in subset),
                    )
    return dict(
        partitions=available,
        partition=selected,
        arms=list(arms),
        arm=chosen,
        summary=summary,
        examples=rows,
        groups=groups,
        bundle_hash=doc.get("bundle_hash"),
        correct=result.get("correct"),
        total=result.get("total"),
    )
