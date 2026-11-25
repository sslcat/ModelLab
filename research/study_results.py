"""Paired receipt comparisons and measured per-rank resources. No cloud calls."""

import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median
from remote.controlled import ARMS
from remote.receipt_scores import score


def binomial_upper(k, n, tail):
    if k == n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(60):
        p = (low + high) / 2
        cdf = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))
        if cdf > tail:
            low = p
        else:
            high = p
    return (low + high) / 2


def paired_counts(gains, losses, n):
    """Two exact 97.5% marginal intervals yield a conservative ≥95% difference interval."""
    if not (n and 0 <= gains and 0 <= losses and gains + losses <= n):
        raise ValueError("Invalid matched counts.")
    changed = gains + losses
    p_value = min(
        1.0,
        2
        * sum(math.comb(changed, j) for j in range(min(gains, losses) + 1))
        / 2**changed,
    )
    gain_upper = binomial_upper(gains, n, 0.0125)
    loss_upper = binomial_upper(losses, n, 0.0125)
    gain_lower = 1 - binomial_upper(n - gains, n, 0.0125)
    loss_lower = 1 - binomial_upper(n - losses, n, 0.0125)
    return dict(
        n=n,
        gains=gains,
        losses=losses,
        ties=n - changed,
        delta=(gains - losses) / n,
        lower=gain_lower - loss_upper,
        upper=gain_upper - loss_lower,
        exact_p=p_value,
    )


def matched(reference, candidate, metric="exact"):
    left = {r["id"]: r for r in reference}
    right = {r["id"]: r for r in candidate}
    if (
        not left
        or left.keys() != right.keys()
        or len(left) != len(reference)
        or len(right) != len(candidate)
    ):
        raise ValueError("Each arm must contain the same unique note IDs.")
    gains = losses = 0
    for key, a in left.items():
        b = right[key]
        if (a["prompt"], a["expected"]) != (b["prompt"], b["expected"]):
            raise ValueError("Paired reference or note differs.")
        av = score(a["prediction"], a["expected"])[metric]
        bv = score(b["prediction"], b["expected"])[metric]
        gains += int(bv and not av)
        losses += int(av and not bv)
    return paired_counts(gains, losses, len(left))


def resource_metrics(doc):
    if "public_resources" in doc:
        return doc["public_resources"]
    folder = Path(doc["archive"]) if doc.get("archive") else None

    def artifact(name):
        path = folder / name if folder else None
        return (
            json.loads(path.read_text(encoding="utf8"))
            if path and path.is_file()
            else {}
        )

    env = artifact("environment.json")
    manifest = artifact("manifest.json")
    weights = manifest.get("files", {}).get("adapter/adapter_model.safetensors", {})
    rows = doc.get("metrics", [])
    losses = [r["loss"] for r in rows if "loss" in r]

    def latest(k):
        return next((r[k] for r in reversed(rows) if k in r), None)

    training_gpu = [
        r["gpu_percent"] for r in rows if "loss" in r and "gpu_percent" in r
    ]
    peak = max(
        (r["peak_allocated_mib"] for r in rows if "peak_allocated_mib" in r),
        default=None,
    )
    vram = max((r["vram_mib"] for r in rows if "vram_mib" in r), default=None)
    q = doc.get("quote") or {}
    cost = None
    pod_seconds = None
    if doc.get("ended") and doc.get("pod"):
        created = doc.get("provider_created_at")
        start = (
            datetime.fromisoformat(created).timestamp()
            if created
            else doc["approved_at"]
        )
        pod_seconds = max(0, doc["ended"] - start)
        stop = min(doc.get("stopped_at", doc["ended"]), doc["ended"])
        cost = (
            max(0, stop - start) / 3600 * (q["gpu_hourly"] + q["storage_hourly"])
            + (doc["ended"] - stop) / 3600 * q["stopped_hourly"]
        )
    actual = (doc.get("actual") or {}).get("totalAmount")
    return dict(
        gpu=env.get("gpu"),
        trainable=env.get("trainable"),
        parameters=env.get("parameters"),
        adapter_bytes=weights.get("size"),
        adapter_sha256=weights.get("sha256"),
        steps=max((r["step"] for r in rows if "loss" in r), default=None),
        first_loss=losses[0] if losses else None,
        last_loss=losses[-1] if losses else None,
        train_loss=latest("train_loss"),
        validation_loss=latest("eval_loss"),
        training_seconds=latest("train_runtime"),
        tokens_per_second=latest("train_tokens_per_second"),
        cuda_peak_mib=peak,
        sampled_vram_mib=vram,
        median_training_gpu_percent=median(training_gpu) if training_gpu else None,
        pod_seconds=pod_seconds,
        estimated_usd=cost,
        posted_usd=actual,
    )


def rank_view(doc, partition, arm, reference):
    result = dict(
        id=doc["id"],
        rank=doc["config"]["rank"],
        alpha=doc["config"]["alpha"],
        status=doc["status"],
        resources=resource_metrics(doc),
        summary=None,
        groups={},
        paired=None,
    )
    partitions = (doc.get("evaluation") or {}).get("partitions", {})
    arms = partitions.get(partition, {})
    if arm not in arms:
        return result
    from studio.research import analysis

    candidate = analysis(doc, partition, arm)
    result.update(summary=candidate["summary"], groups=candidate["groups"])
    if reference in arms and doc.get("status") == "completed":
        result["paired"] = matched(arms[reference]["examples"], arms[arm]["examples"])
    return result


def study_view(
    documents, partition="locked", arm="tuned_constrained", reference="base_constrained"
):
    if (
        partition not in ("development", "locked")
        or arm not in ARMS
        or reference not in ARMS
    ):
        raise ValueError("Unknown controlled study partition or arm.")
    docs = sorted(
        (d for d in documents if d.get("research_design")),
        key=lambda d: d["config"]["rank"],
    )
    return dict(
        partition=partition,
        arm=arm,
        reference=reference,
        arms=list(ARMS),
        ranks=[rank_view(d, partition, arm, reference) for d in docs],
    )
