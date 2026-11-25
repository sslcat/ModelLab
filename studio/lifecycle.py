"""Durable state transitions for one approved experiment."""

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time

from . import archive, store
from .cloud import CloudError, CreateRejected, Provider, Transport, identity
from .config import BUDGET, IMAGE, MODELS
from .datasets import canon, digest

ACTIVE = {
    "authorized",
    "creating",
    "allocated",
    "running",
    "archiving",
    "deleting",
    "stopping",
    "recovery",
    "uncertain",
}


def committed(doc):
    return max(doc.get("reserved", 0), (doc.get("actual") or {}).get("totalAmount", 0))


def quote(key, p):
    doc = store.run(key)
    offer = p.quote(doc["config"])
    inventory = p.inventory()
    offer.update(
        token=secrets.token_urlsafe(24),
        expires=int(time.time() + 300),
        account={
            "pods": len(inventory["pods"]),
            "volumes": len(inventory["volumes"]),
            "endpoints": len(inventory["endpoints"]),
            "balance": inventory["balance"],
        },
    )
    with store.db(True) as con:
        doc = store.run(key, con)
        if doc["status"] != "draft":
            raise ValueError("Only drafts can be quoted.")
        doc["quote"] = offer
        store.put(con, doc)
    return offer


def authorize(key, token, p):
    current = store.run(key)
    check_research_lock(current)
    offer = current["quote"]
    if (
        not offer
        or not secrets.compare_digest(token, offer["token"])
        or time.time() > offer["expires"]
    ):
        raise ValueError("The quote expired or changed. Request a new quote.")
    _, bundle = store.dataset(current["config"]["dataset"])
    if digest(bundle) != current["bundle_hash"]:
        raise ValueError("Dataset fingerprint changed. Recreate the draft.")
    live = p.quote(current["config"])
    if current.get("size_study"):
        from research.size_design import CEILING, check
        if live["gpu_hourly"] > CEILING or offer["max_usd"] > check(current)["max_usd"]:
            raise ValueError("The size study price exceeds its registered session allowance.")
    inventory = p.inventory()
    account = offer["account"]
    if any(
        len(inventory[key]) != account[key] for key in ("pods", "volumes", "endpoints")
    ):
        raise ValueError("Account inventory changed. Refresh the quote.")
    balance = inventory["balance"]
    if (
        balance.get("billingMode") == "PREPAID"
        and balance.get("balanceAmount", 0) < offer["max_usd"]
    ):
        raise ValueError("Available prepaid balance is below the estimate.")
    if live["max_usd"] > offer["max_usd"] or live["gpu_hourly"] > offer["gpu_hourly"]:
        raise ValueError("The live rate increased. Refresh the quote.")
    identity()
    with store.db(True) as con:
        doc = store.run(key, con)
        if (
            doc["status"] != "draft"
            or doc["quote"] != offer
            or time.time() > offer["expires"]
        ):
            raise ValueError("Quote already used or expired.")
        others = store.runs(con)
        if any(r["status"] in ACTIVE for r in others):
            raise ValueError(
                "Resolve the existing active allocation before starting another."
            )
        if sum(committed(r) for r in others) + offer["max_usd"] > BUDGET:
            raise ValueError("Insufficient remaining mission budget.")
        now = time.time()
        doc.update(
            status="authorized",
            approved_at=now,
            deadline=now + doc["config"]["minutes"] * 60,
            reserved=offer["max_usd"],
        )
        store.put(con, doc)
    return doc


def historical_worker_paths():
    return {p.name: p for p in (store.ROOT / "remote").iterdir() if p.is_file()}


def worker_sources():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in historical_worker_paths().items()}


def current_worker_paths():
    paths = historical_worker_paths()
    for name in ("train.py", "requirements.txt"):
        paths[name] = store.ROOT / "remote" / "current" / name
    return paths


def check_research_lock(doc):
    if doc.get("pilot_snapshot"):
        from research.reproduce import pilot_sources
        hashes = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n,p in pilot_sources().items()}
        if hashes != doc.get("research_worker_sha256"):
            raise ValueError("Pilot worker differs from its frozen snapshot.")
        return
    if doc.get("size_study"):
        from research.size_design import check
        check(doc)
        return
    frozen = doc.get("research_worker_sha256")
    if frozen and frozen != worker_sources():
        raise ValueError("The worker changed after the research design was frozen.")


def worker_config(doc):
    dataset = store.dataset_meta(doc["config"]["dataset"])
    provenance = {
        k: dataset[k]
        for k in (
            "id",
            "name",
            "license",
            "seed",
            "fraction",
            "fingerprint",
            "split_id",
            "train",
            "validation",
            "test",
        )
    }
    sources = doc.get("research_worker_sha256")
    extra = {}
    if not sources and not doc.get("size_study"):
        sources = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in current_worker_paths().items()}
        extra["dependency_profile"] = "current"
    if doc.get("size_study"):
        from research.size_design import locked_design
        frozen = locked_design()
        sources = frozen["workers"]
        extra = {"protocol": frozen["protocol"], "size_study": doc["size_study"]}
    elif dataset.get("protocol"):
        from remote.controlled import protocol
        extra["protocol"] = protocol()
    return dict(
        doc["config"], **extra,
        dataset_provenance=provenance,
        cloud_image=IMAGE,
        worker_sha256=sources,
        revision=MODELS[doc["config"]["model"]],
        experiment=doc["id"],
        bundle_hash=doc["bundle_hash"],
        deadline=doc.get("deadline"),
    )


def stage(doc, ssh):
    check_research_lock(doc)
    base = "/workspace/modellab/"
    ssh.command(
        'kill -0 "$(cat /run/modellab-guard.pid)" && test -d /workspace/modellab'
    )
    _, data = store.dataset(doc["config"]["dataset"])
    if digest(data) != doc["bundle_hash"]:
        raise ValueError("Staged data changed.")
    payload = {
        "config.json": canon(worker_config(doc)).encode(),
        "data.json": canon(data).encode(),
    }
    names = ["requirements.txt", "launch.py", "train.py"]
    if "protocol" in json.loads(payload["config.json"]):
        names.extend(["controlled.py", "receipt_scores.py"])
    if doc.get("size_study"):
        from research.size_design import sources
        files = {n: path for n, path in sources().items() if n != "boot.sh"}
    elif doc.get("pilot_snapshot"):
        from research.reproduce import pilot_sources
        files = {n: path for n,path in pilot_sources().items() if n != "boot.sh"}
    else:
        paths = (
            current_worker_paths()
            if not doc.get("research_worker_sha256")
            else historical_worker_paths()
        )
        files = {name: paths[name] for name in names}
    for name, path in files.items():
        payload[name] = path.read_bytes()
    for name, content in payload.items():
        ssh.command(
            f"umask 077 && cat > {base}{name}.tmp && mv {base}{name}.tmp {base}{name}",
            content,
        )
    observed = ssh.command("sha256sum /workspace/modellab/data.json").decode().split()
    if not observed or observed[0] != doc["bundle_hash"]:
        raise CloudError(
            "The remote dataset checksum does not match the approved split."
        )
    # Persist intent first. A lost reply is retried with a remote process lock.
    store.update(doc["id"], status="running", launched=True)
    launch(ssh)
    store.clear_staging(doc["config"]["dataset"])


def launch(ssh):
    ssh.command(
        "cd /workspace/modellab && (nohup /run/modellab-python -u launch.py </dev/null >>supervisor.log 2>&1 &)"
    )


def metrics(doc, ssh):
    data = ssh.command(
        "cd /workspace/modellab && if test -f metrics.jsonl; then tail -n 4000 metrics.jsonl; fi"
    )
    rows = []
    for line in data.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if rows != doc["metrics"]:
        store.update(doc["id"], metrics=rows)


def tick(key, p, transport=Transport):
    doc = store.run(key)
    status = doc["status"]
    if status not in ACTIVE:
        return
    now = time.time()
    if status == "authorized":
        if doc["cancel"] or now >= min(doc["deadline"] - 300, doc["quote"]["expires"]):
            store.update(key, status="cancelled", reserved=0)
            return
        # A crash/timeout after this write can reconcile, but can never repeat POST.
        _, pub = identity()
        store.update(key, status="creating")
        try:
            pid = p.create(doc, pub)
        except CreateRejected as exc:
            store.update(key, status="failed", detail=str(exc), reserved=0)
            return
        store.update(key, pod=pid, status="allocated")
        return
    if not doc["pod"]:
        matches = [
            r for r in p.pages("pods", "pods") if r["name"] == doc["resource_name"]
        ]
        if len(matches) == 1:
            store.update(key, pod=matches[0]["id"], status="allocated")
        else:
            store.update(
                key,
                status="uncertain",
                detail="Create response uncertain. Checking inventory; no second pod will be requested.",
            )
        return
    pid = doc["pod"]
    pod = p.pod(pid)
    if not pod or pod["status"] == "TERMINATED":
        outcome = (
            doc.get("outcome", "failed")
            if doc.get("archived") or not doc.get("launched")
            else "lost"
        )
        store.update(
            key,
            status=outcome,
            ended=now,
            terminated=True,
            detail=doc["detail"]
            if status == "deleting"
            else "Pod is absent from provider inventory.",
        )
        return
    if pod["name"] != doc["resource_name"]:
        raise CloudError("Provider resource identity mismatch.")
    if not doc.get("provider_created_at") and pod.get("createdAt"):
        store.update(key, provider_created_at=pod["createdAt"])
    if status == "deleting":
        if doc.get("launched") and not archive.verified(doc):
            store.update(
                key,
                status="stopping",
                detail="Archive changed; preserving the remote volume.",
            )
            return
        p.delete(pid)
        return
    if (
        status == "stopping"
        or (now >= doc["deadline"] and status != "recovery")
        or pod["cost"] > doc["quote"]["gpu_hourly"] + 0.00001
    ):
        if pod["status"] == "EXITED":
            store.update(
                key,
                status="recovery" if doc.get("launched") else "deleting",
                stopped_at=doc.get("stopped_at", now),
            )
        elif "stop" in pod.get("actions", []):
            p.stop(pid)
            store.update(key, status="stopping")
        return
    if status == "recovery" and pod["status"] != "RUNNING":
        return
    if pod["status"] in {"ERROR", "EXITED"}:
        store.update(
            key,
            status="recovery" if doc.get("launched") else "deleting",
            stopped_at=now,
            detail="Pod exited. Persistent disk may require recovery.",
        )
        return
    if not doc.get("launched") and (doc["cancel"] or now - doc["approved_at"] > 480):
        store.update(
            key,
            status="deleting",
            outcome="cancelled" if doc["cancel"] else "failed",
            detail="Stopped before worker launch.",
        )
        return
    if pod["status"] != "RUNNING" or not pod["ssh"].get("direct"):
        return
    ssh = transport(pod)
    if status == "allocated":
        archive.recovery_record(doc)
        stage(doc, ssh)
        return
    if status == "recovery":
        ssh.command(
            "cd /workspace/modellab && if ! test -f manifest.json; then flock -n worker.lock /run/modellab-python -c \"from launch import finish; finish('failed', 'Recovered after interruption')\"; fi"
        )
    if doc["cancel"]:
        ssh.command("touch /workspace/modellab/cancel")
    metrics(doc, ssh)
    manifest = ssh.command(
        "cd /workspace/modellab && if test -f manifest.json; then cat manifest.json; fi"
    )
    if manifest:
        store.update(key, status="archiving")
        result = archive.retrieve(doc, json.loads(manifest), ssh)
        store.update(key, status="deleting", **result)
    elif status == "running":
        launch(ssh)


def supervisor(key):
    # OS releases this exclusive lease on crash. Startup can safely reattach.
    file = (store.home() / ("lease-" + key)).open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        file.close()
        return
    p = Provider()
    try:
        while store.run(key)["status"] in ACTIVE:
            try:
                tick(key, p)
            except (ValueError, KeyError, OSError, CloudError) as exc:
                store.update(key, detail=str(exc)[:300])
            time.sleep(
                5 if store.run(key)["status"] not in {"recovery", "uncertain"} else 30
            )
        doc = store.run(key)
        if doc.get("pod"):
            try:
                store.update(key, actual=p.bill(doc["pod"], doc["approved_at"]))
            except CloudError:
                pass
    finally:
        file.close()


def spawn(key):
    subprocess.Popen(
        [sys.executable, "-m", "studio.lifecycle", key],
        cwd=store.ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


if __name__ == "__main__":
    supervisor(sys.argv[1])
