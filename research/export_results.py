"""Export audit-ready derived tables after all frozen ranks finish. Existing run archives are read-only."""

import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
from statistics import median
from remote.controlled import ARMS, protocol
from research.receipts import locked_bundle
from research.study_results import matched, resource_metrics
from studio import archive, lifecycle, store
from studio.cloud import artifact_root, one_drive
from studio.datasets import digest
from studio.research import analysis, verify_evaluation


def csv_bytes(rows):
    stream = StringIO(newline="")
    columns = dict.fromkeys(key for row in rows for key in row)
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf8")


def json_bytes(obj):
    return (
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def atomic(dest, payload):
    tmp = dest.with_name(dest.name + ".partial")
    try:
        tmp.write_bytes(payload)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def adapter_header(path):
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        if not 0 < size < 4 * 1024**2:
            raise ValueError("Invalid adapter header length.")
        header = json.loads(handle.read(size))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    return dict(
        serialized_parameters=sum(math.prod(t["shape"]) for t in tensors),
        adapter_tensors=len(tensors),
        adapter_dtypes=",".join(sorted({t["dtype"] for t in tensors})),
    )


def collect_results():
    docs = sorted(
        [r for r in store.runs() if r.get("research_design")],
        key=lambda r: r["config"]["rank"],
    )
    if [r["config"]["rank"] for r in docs] != [4, 8, 16, 32]:
        raise ValueError("Expected all four frozen rank records.")
    collection = {
        "resources": [],
        "training_history": [],
        "arms": [],
        "strata": [],
        "paired": [],
        "repeatability": [],
        "rank_contrasts": [],
        "failure_index": [],
        "samples": [],
    }
    bundles = {}
    worker_hashes = []
    settings = []
    evidence = []
    data_hash = digest(locked_bundle())
    for doc in docs:
        rank = doc["config"]["rank"]
        if (
            doc["status"] != "completed"
            or not doc["terminated"]
            or not archive.verified(doc)
        ):
            raise ValueError(f"Rank {rank} has no verified completed archive.")
        folder = Path(doc["archive"])
        if folder != artifact_root() / doc["id"]:
            raise ValueError("Unrecognized archive root.")
        config = json.loads((folder / "config.json").read_text(encoding="utf8"))
        ev = json.loads((folder / "evaluation.json").read_text(encoding="utf8"))
        verify_evaluation(config, ev)
        lifecycle.check_research_lock(doc)
        if (
            config["bundle_hash"] != data_hash
            or config["protocol"] != protocol()
            or config["alpha"] != 2 * rank
        ):
            raise ValueError("Frozen design mismatch.")
        worker_hashes.append(config["worker_sha256"])
        settings.append(
            {
                k: v
                for k, v in config.items()
                if k
                not in {
                    "rank",
                    "alpha",
                    "experiment",
                    "name",
                    "deadline",
                    "gpu",
                    "minutes",
                }
            }
        )
        if config["worker_sha256"] != doc["research_worker_sha256"]:
            raise ValueError(
                "Executed worker hash differs from the preregistered worker."
            )
        bundle = ev["partitions"]
        bundles[rank] = bundle
        history = [
            json.loads(line)
            for line in (folder / "metrics.jsonl")
            .read_text(encoding="utf8")
            .splitlines()
            if line.strip()
        ]
        if history != doc["metrics"]:
            raise ValueError(
                "Local training telemetry differs from its verified remote archive."
            )
        collection["training_history"].extend(
            dict(rank=rank, **entry) for entry in history
        )
        resources = resource_metrics(doc)
        resources.update(
            adapter_header(folder / "adapter" / "adapter_model.safetensors")
        )
        if (
            resources["serialized_parameters"] != resources["trainable"]
            or resources["trainable"] != 315392 * rank
        ):
            raise ValueError(
                "Exported adapter disagrees with the measured trainable count or pinned Qwen geometry."
            )
        measurements = []
        samplefile = folder / "observer-metrics.jsonl"
        if not samplefile.is_file():
            samplefile = store.home() / ("research-samples-" + doc["id"] + ".jsonl")
        if samplefile.is_file():
            for line in samplefile.read_text(encoding="utf8").splitlines():
                s = json.loads(line)
                if s["pod"] != doc["pod"]:
                    raise ValueError("Telemetry pod identity mismatch.")
                measurements.append(s)
                collection["samples"].append(dict(rank=rank, **s))
        resources.update(
            external_samples=len(measurements),
            max_observed_pod_vram_mib=max(
                (s["vram_mib"] for s in measurements), default=None
            ),
        )
        collection["resources"].append(
            dict(
                rank=rank,
                alpha=config["alpha"],
                run=doc["id"],
                pod=doc["pod"],
                **resources,
            )
        )
        evidence.append(
            dict(
                rank=rank,
                experiment=doc["id"],
                manifest_sha256=archive.sumfile(folder / "manifest.json"),
                adapter_sha256=resources["adapter_sha256"],
                archive=str(folder),
                bundle_sha256=data_hash,
                model_revision=config["revision"],
                cloud_image=config["cloud_image"],
                gpu_rate=doc["quote"]["gpu_hourly"],
                approved_max=doc["quote"]["max_usd"],
                archive_sync=archive.sync_state(folder),
            )
        )
        for part, arms in bundle.items():
            for arm in ARMS:
                examples = arms[arm]["examples"]
                v = analysis(dict(doc, evaluation=ev), part, arm)
                s = v["summary"]
                for example in v["examples"]:
                    strict = example["scores"]
                    if strict["exact"] != example["correct"]:
                        raise ValueError(
                            "Stored prediction classification and independent score disagree."
                        )
                    if not strict["exact"]:
                        collection["failure_index"].append(
                            dict(
                                rank=rank,
                                partition=part,
                                arm=arm,
                                note_id=example["id"],
                                template=example["template"],
                                unseen_template=example["unseen_template"],
                                unseen_pair=example["unseen_pair"],
                                errors=" | ".join(strict["errors"]),
                                content_errors=" | ".join(
                                    example["content_scores"]["errors"]
                                ),
                                content_correct=example["content_scores"]["exact"],
                                hit_token_limit=example["hit_token_limit"],
                            )
                        )
                collection["arms"].append(
                    dict(
                        rank=rank,
                        partition=part,
                        arm=arm,
                        n=len(examples),
                        json_valid=s["valid"],
                        schema=s["schema"],
                        exact=s["exact"],
                        content=s["content"],
                        **{"raw_" + f: s["fields"][f] for f in s["fields"]},
                        **{
                            "content_" + f: s["content_fields"][f]
                            for f in s["content_fields"]
                        },
                        latency_p50_ms=s["latency_p50_ms"],
                        latency_p95_ms=s["latency_p95_ms"],
                        generated_tokens_median=median(
                            x["generated_tokens"] for x in examples
                        ),
                        hit_token_limit=sum(x["hit_token_limit"] for x in examples),
                    )
                )
                for label, g in v["groups"].items():
                    collection["strata"].append(
                        dict(
                            rank=rank,
                            partition=part,
                            arm=arm,
                            stratum=label,
                            **g,
                        )
                    )
            for before, after in (
                ("base", "prompted"),
                ("base", "base_constrained"),
                ("base", "tuned"),
                ("tuned", "tuned_constrained"),
                ("base_constrained", "tuned_constrained"),
            ):
                for metric in ("exact", "content"):
                    collection["paired"].append(
                        dict(
                            rank=rank,
                            partition=part,
                            reference=before,
                            candidate=after,
                            metric=metric,
                            **matched(
                                arms[before]["examples"],
                                arms[after]["examples"],
                                metric,
                            ),
                        )
                    )
    if any(s != settings[0] for s in settings) or any(
        h != worker_hashes[0] for h in worker_hashes
    ):
        raise ValueError("Non-rank scientific settings or workers differ across ranks.")
    for part in ("development", "locked"):
        for rank in (8, 16, 32):
            for arm in ARMS[:3]:
                a = bundles[4][part][arm]["examples"]
                b = bundles[rank][part][arm]["examples"]
                index = {r["id"]: r for r in b}
                same = sum(r["prediction"] == index[r["id"]]["prediction"] for r in a)
                collection["repeatability"].append(
                    dict(
                        partition=part,
                        arm=arm,
                        reference_rank=4,
                        rank=rank,
                        n=len(a),
                        identical_text=same,
                        **{"score_" + k: v for k, v in matched(a, b).items()},
                    )
                )
        for rank in (4, 16, 32):
            for arm in ("tuned", "tuned_constrained"):
                collection["rank_contrasts"].append(
                    dict(
                        partition=part,
                        arm=arm,
                        reference_rank=8,
                        rank=rank,
                        **matched(
                            bundles[8][part][arm]["examples"],
                            bundles[rank][part][arm]["examples"],
                        ),
                    )
                )
    return collection, evidence


def measured_report(tables):
    arms = tables["arms"]
    primary = next(
        r
        for r in tables["paired"]
        if r["rank"] == 8
        and r["partition"] == "locked"
        and r["reference"] == "base_constrained"
        and r["candidate"] == "tuned_constrained"
        and r["metric"] == "exact"
    )
    lines = [
        "# Controlled receipts · measured results",
        "",
        f"The preregistered rank-8 comparison gained {primary['gains']} notes, lost {primary['losses']} and tied on {primary['ties']} of {primary['n']}. "
        f"The paired strict accuracy change was {100 * primary['delta']:+.2f} percentage points "
        f"(conservative 95% interval {100 * primary['lower']:+.2f} to {100 * primary['upper']:+.2f}, exact McNemar p = {primary['exact_p']:.5g}).",
        "",
        "## Locked test",
        "",
        "| Rank | Arm | JSON valid | Schema | Strict exact | After one fence | Content item / quantity / site | Latency p50 / p95 (ms) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in arms:
        if r["partition"] != "locked":
            continue
        lines.append(
            f"| {r['rank']} | {r['arm']} | {r['json_valid']}/{r['n']} | {r['schema']}/{r['n']} | {r['exact']}/{r['n']} | "
            f"{r['content']}/{r['n']} | {r['content_item']} / {r['content_quantity']} / {r['content_site']} | "
            f"{r['latency_p50_ms']:.0f} / {r['latency_p95_ms']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Training measurements",
            "",
            "| Rank / alpha | Trainable parameters | Adapter bytes | Steps | Mean training loss | Validation loss | Train seconds | Tokens/s | CUDA peak (MiB) | Estimated USD | Posted USD |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in tables["resources"]:
        bill = f"{r['posted_usd']:.5f}" if r["posted_usd"] is not None else "Pending"
        lines.append(
            f"| {r['rank']} / {r['alpha']} | {r['trainable']:,} | {r['adapter_bytes']:,} | {r['steps']} | {r['train_loss']:.6f} | "
            f"{r['validation_loss']:.6f} | {r['training_seconds']:.2f} | {r['tokens_per_second']:.2f} | {r['cuda_peak_mib']:.1f} | "
            f"{r['estimated_usd']:.5f} | {bill} |"
        )
    lines.extend(
        [
            "",
            "## Generalization under the shared grammar",
            "",
            "| Rank | Partition | Template and pair stratum | Base exact | Tuned exact |",
            "| ---: | --- | --- | ---: | ---: |",
        ]
    )
    baseline = {
        (r["rank"], r["partition"], r["stratum"]): r
        for r in tables["strata"]
        if r["arm"] == "base_constrained"
    }
    for r in tables["strata"]:
        if r["arm"] != "tuned_constrained":
            continue
        b = baseline[r["rank"], r["partition"], r["stratum"]]
        lines.append(
            f"| {r['rank']} | {r['partition']} | {r['stratum']} | {b['exact']}/{b['n']} | {r['exact']}/{r['n']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation of the measurements",
            "",
            "Strict record match requires parseable, schema-compliant JSON with three correct typed fields. The content diagnostic removes at most one enclosing Markdown fence. The primary comparison uses the same grammar on both sides. Latency is synchronized generation time, excluding tokenization and parser construction.",
            "",
            "The paired gain-minus-loss estimate divides discordant gains minus losses by the number of matched notes. Each direction receives an exact 97.5% binomial proportion interval. Subtracting opposite endpoints gives a conservative simultaneous 95% interval under independent sampled notes. The two-sided exact McNemar test conditions on the number of discordant notes. Shared synthetic templates and one training seed limit interpretation. Secondary comparisons are exploratory.",
            "",
            "All ranks used QLoRA with NF4 base weights, query and value adapters, and alpha/rank = 2. The pinned Qwen geometry gives 315,392 trainable parameters per unit of rank. Serialized tensor shapes and remote PEFT counts were checked separately.",
            "",
            "The CSV files include development results, every arm, all four crossed strata, paired comparisons, base-control repeatability, resource measurements and sampled GPU telemetry. Each run's verified archive contains every prompt, reference, prediction, timing, environment and adapter. The evidence file identifies each archive and hash.",
            "",
        ]
    )
    return "\n".join(lines)


def result_directory():
    return (
        (one_drive() or store.home())
        / "ModelLab"
        / "research"
        / "controlled-receipts-results"
    )


def export(destination=None):
    tables, evidence = collect_results()
    if destination is None:
        destination = result_directory()
    destination.mkdir(parents=True, exist_ok=True)
    payloads = {name + ".csv": csv_bytes(rows) for name, rows in tables.items() if rows}
    payloads["measurements.md"] = measured_report(tables).encode("utf8")
    payloads["evidence.json"] = json_bytes(
        {
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_source": "5665c05",
            "paired_analysis_source": "27a5145",
            "protocol": protocol(),
            "runs": evidence,
            "cost_status": "Posted amounts come from RunPod per-pod billing records and can be revised.",
        }
    )
    for name, payload in payloads.items():
        atomic(destination / name, payload)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    if any(
        archive.sumfile(destination / name) != expected
        for name, expected in hashes.items()
    ):
        raise ValueError("Derived report export failed checksum verification.")
    atomic(destination / "table-checksums.json", json_bytes(hashes))
    return destination


if __name__ == "__main__":
    print(export())
