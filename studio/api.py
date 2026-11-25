import asyncio
import csv
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import archive, lifecycle, store
from .cloud import CloudError, Provider, artifact_root, key, one_drive
from .config import BUDGET, DISK, MODELS, VOLUME, Config
from .datasets import LIMIT, SECRET, parse


@asynccontextmanager
async def lifespan(app):
    store.init()
    for doc in store.runs():
        if doc["status"] in lifecycle.ACTIVE:
            lifecycle.spawn(doc["id"])
    yield


app = FastAPI(title="ModelLab", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
)


@app.middleware("http")
async def local_writes(req: Request, call_next):
    if req.method not in ("GET", "HEAD", "OPTIONS"):
        origin = req.headers.get("origin")
        if req.headers.get("x-modellab") != "studio" or (
            origin and urlsplit(origin).netloc != req.headers.get("host")
        ):
            return JSONResponse({"detail": "Local studio request required."}, 403)
        length = req.headers.get("content-length")
        if not length or not length.isdecimal() or int(length) > LIMIT * 3 + 10000:
            return JSONResponse({"detail": "Request exceeds import limit."}, 413)
    return await call_next(req)


@app.exception_handler(CloudError)
async def cloud_error(req, exc):
    return JSONResponse({"detail": str(exc)}, 503)


@app.exception_handler(ValueError)
@app.exception_handler(csv.Error)
async def bad_input(req, exc):
    return JSONResponse({"detail": str(exc)}, 400)


def display_run(doc):
    # Consider billed amounts provisional until the provider finalizes its ledger.
    d = dict(doc)
    if (d.get("evaluation") or {}).get("partitions"):
        ev = d["evaluation"]
        # Prediction details have a dedicated endpoint; overview polls stay small.
        d["evaluation"] = {"split": ev["split"], "limitations": ev["limitations"]}
    if d.get("pod") and d.get("approved_at"):
        until = d.get("ended", time.time())
        created = d.get("provider_created_at")
        start = datetime.fromisoformat(created).timestamp() if created else d["approved_at"]
        hours = max(0, (min(d.get("stopped_at", until), until) - start) / 3600)
        stopped = max(0, (until - d.get("stopped_at", until)) / 3600)
        q = d["quote"]
        d["estimated_usage"] = round(
            hours * (q["gpu_hourly"] + q["storage_hourly"])
            + stopped * q["stopped_hourly"],
            5,
        )
        d["stopped_storage_estimate"] = round(stopped * q["stopped_hourly"], 5)
    if d.get("archived") and d.get("sync") == "OneDrive sync unconfirmed":
        folder = Path(d["archive"])
        if folder.is_dir():
            updated = archive.sync_state(folder)
            if updated != d["sync"]:
                store.update(d["id"], sync=updated)
                d["sync"] = updated
    return d


@app.get("/api/studio")
def overview():
    docs = store.runs()
    return {
        "datasets": store.datasets(),
        "runs": [display_run(d) for d in docs],
        "models": MODELS,
        "budget": BUDGET,
        "reserved": sum(lifecycle.committed(d) for d in docs),
        "credential": bool(key()),
        "one_drive": str(one_drive() or ""),
        "artifact_root": str(artifact_root()),
        "disk": DISK,
        "volume": VOLUME,
        "active": sum(d["status"] in lifecycle.ACTIVE for d in docs),
    }


class Import(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    license: str = Field(min_length=2, max_length=120)
    format: Literal["csv", "jsonl"]
    content: str = Field(max_length=LIMIT)
    seed: int = Field(default=42, ge=0, le=2147483647)
    fraction: float = Field(default=0.15, ge=0.05, le=0.45)
    public_data: Literal[True]


@app.post("/api/datasets")
def import_data(body: Import):
    if SECRET.search(body.name + " " + body.license):
        raise ValueError("Credential-like metadata cannot be imported.")
    return store.save_dataset(
        body.name,
        body.license,
        parse(body.content, body.format),
        body.seed,
        body.fraction,
    )


@app.post("/api/runs")
def create_draft(cfg: Config):
    if SECRET.search(cfg.name + " " + cfg.gpu):
        raise ValueError("Credential-like experiment metadata.")
    return store.draft(cfg.model_dump())


@app.get("/api/gpus")
def gpus():
    if os.getenv("MODELLAB_OFFLINE") == "1":
        raise CloudError("Cloud access is disabled for offline verification.")
    if key():
        return {
            "source": "RunPod Secure Cloud · CUDA ≥12.8",
            "gpus": Provider().gpus(),
            "at": time.time(),
        }
    query = "{gpuTypes{id displayName memoryInGb securePrice lowestPrice(input:{gpuCount:1,secureCloud:true}){stockStatus uninterruptablePrice}}}"
    try:
        r = httpx.post(
            "https://api.runpod.io/graphql", json={"query": query}, timeout=12
        )
        r.raise_for_status()
        items = r.json()["data"]["gpuTypes"]
    except (httpx.HTTPError, KeyError, TypeError) as exc:
        raise CloudError("Public catalog unavailable. Try refreshing.") from exc
    rows = []
    for g in items:
        if g["memoryInGb"] < 24 or not g["securePrice"]:
            continue
        price = g.get("lowestPrice") or {}
        rows.append(
            {
                "id": g["id"],
                "name": g["displayName"],
                "vram": g["memoryInGb"],
                "hourly": price.get("uninterruptablePrice") or g["securePrice"],
                "stock": price.get("stockStatus") or "Unconfirmed",
            }
        )
    return {
        "source": "RunPod public Secure Cloud catalog",
        "gpus": sorted(rows, key=lambda g: g["hourly"]),
        "at": time.time(),
    }


@app.get("/api/inventory")
def inventory():
    info = Provider().inventory()
    docs = store.runs()
    with store.db() as con:
        namespace = con.execute("SELECT v FROM kv WHERE k='namespace'").fetchone()[0]
    live = [
        r for r in info["pods"] if r["name"].startswith("modellab-" + namespace + "-")
    ]
    active = {d["pod"] for d in docs if d["status"] in lifecycle.ACTIVE}
    info["abandoned"] = [
        {"id": p["id"], "name": p["name"]} for p in live if p["id"] not in active
    ]
    return info


@app.post("/api/runs/{rid}/quote")
def get_quote(rid: str):
    return lifecycle.quote(rid, Provider())


class Approval(BaseModel):
    token: str
    approve: Literal[True]


@app.post("/api/runs/{rid}/approve")
def approve(rid: str, body: Approval):
    doc = lifecycle.authorize(rid, body.token, Provider())
    lifecycle.spawn(rid)
    return doc


@app.post("/api/runs/{rid}/cancel")
def cancel(rid: str):
    with store.db(True) as con:
        doc = store.run(rid, con)
        if doc["status"] not in lifecycle.ACTIVE:
            raise ValueError("Experiment has no active allocation.")
        doc["cancel"] = True
        store.put(con, doc)
    return {"requested": True}


@app.post("/api/runs/{rid}/billing")
def billing(rid: str):
    doc = store.run(rid)
    if not doc.get("pod"):
        raise ValueError("No provider pod assigned.")
    amounts = Provider().bill(doc["pod"], doc["approved_at"])
    if doc.get("archive") and amounts:
        dest = Path(doc["archive"]) / "billing.json"
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(amounts), encoding="utf8")
        tmp.replace(dest)
        store.update(rid, sync=archive.sync_state(dest.parent))
    return store.update(rid, actual=amounts or doc.get("actual"))


@app.get("/api/runs/{rid}/events")
async def events(rid: str, req: Request):
    store.run(rid)

    async def stream():
        previous = None
        while not await req.is_disconnected():
            doc = store.run(rid)
            event = json.dumps(
                {
                    "status": doc["status"],
                    "metrics": doc["metrics"],
                    "detail": doc["detail"],
                }
            )
            if event != previous:
                yield "data: " + event + "\n\n"
                previous = event
            else:
                yield ": heartbeat\n\n"
            if doc["status"] not in lifecycle.ACTIVE:
                break
            await asyncio.sleep(3)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


STUDY_FILES = {
    "findings": "findings.md", "tables": "measurements.md", "figure": "rank-comparison.png",
    "scores": "arms.csv", "paired": "paired.csv", "strata": "strata.csv",
    "failures": "failure_index.csv", "costs": "cloud-closeout.json",
    "training": "training_history.csv", "resources": "resources.csv",
    "rank-contrasts": "rank_contrasts.csv", "repeatability": "repeatability.csv",
}


def published_study(docs):
    from research.export_results import result_directory
    runs = [d for d in docs if d.get("research_design")]
    if len(runs) != 4 or any(d["status"] != "completed" for d in runs):
        return None
    folder = result_directory()
    try:
        evidence = json.loads((folder / "evidence.json").read_text(encoding="utf8"))
        if {r["experiment"] for r in evidence["runs"]} == {r["id"] for r in runs}:
            return folder
    except (OSError, KeyError, ValueError):
        pass
    return None


SIZE_FILES = {
    "findings": "findings.md", "figure": "size-comparison.png", "seeds": "seeds.csv",
    "difficulty": "difficulty.csv", "paired": "paired.csv", "frontiers": "frontiers.csv",
    "billing": "cloud-closeout.json", "history": "training-history.csv",
    "adapters": "adapter-inventory.csv", "agreement": "seed-agreement.csv",
    "matched": "matched-size.csv", "primary": "primary-notes.csv",
    "checksums": "release-sha256.json",
}


def size_report_folder(documents):
    drive = one_drive() or store.home()
    runs = [d for d in documents if d.get("size_study")]
    if not drive or len(runs) != 4 or any(d["status"] != "completed" or not d.get("terminated") for d in runs):
        return None
    folder = drive / "ModelLab" / "research" / "receipt-size-v1-results"
    try:
        evidence = json.loads((folder / "evidence.json").read_text(encoding="utf8"))
        if {d["id"] for d in runs} == {r["run"] for r in evidence}:
            return folder
    except (OSError, ValueError, KeyError):
        pass
    return None


@app.get("/api/research/size-files/{kind}")
def size_report_file(kind: str):
    folder = size_report_folder(store.runs())
    name = SIZE_FILES.get(kind)
    if not folder or not name or not (folder / name).is_file():
        raise HTTPException(404, "Verified study export unavailable.")
    return FileResponse(folder / name, filename=name)


@app.get("/api/research/model-size")
def model_size(benchmark: str = "mission3", partition: str = "locked", condition: str = "tuned"):
    from research.size_results import view
    documents = store.runs()
    result = view(documents, benchmark, partition, condition)
    folder = size_report_folder(documents) if benchmark == "mission3" else None
    result["exports"] = [k for k,name in SIZE_FILES.items() if folder and (folder / name).is_file()]
    return result


@app.get("/api/research/model-size/{rid}/cases")
def size_cases(rid: str, partition: Literal["locked", "development"], arm: str, failures: bool = False,
               offset: int = 0, difficulty: str | None = None):
    from research.size_results import arms_for
    from remote.receipt_scores import score
    d = store.run(rid)
    if not (d.get("size_study") or d.get("research_design") and d["config"]["rank"] == 4) or d["status"] != "completed":
        raise HTTPException(404, "Completed size benchmark unavailable.")
    arms = arms_for(d, partition)
    if arm not in arms or offset < 0:
        raise ValueError("Invalid prediction selection.")
    rows = [e for e in arms[arm] if (not failures or not e["correct"]) and
            (not difficulty or e.get("difficulty", e["template"]) == difficulty)]
    return {"total": len(rows), "examples": [dict(e, errors=score(e["prediction"], e["expected"])["errors"])
            for e in rows[offset:offset+20]]}


@app.get("/api/research/model-size/{rid}/adapter/{seed}/{kind}")
def size_adapter(rid: str, seed: int, kind: Literal["weights", "config"]):
    from research.size_results import SEEDS, prefix
    d = store.run(rid)
    if not d.get("size_study") or d["status"] != "completed" or seed not in SEEDS:
        raise HTTPException(404, "Verified seed adapter unavailable.")
    name = "adapter_model.safetensors" if kind == "weights" else "adapter_config.json"
    return FileResponse(Path(d["archive"]) / prefix(seed) / "adapter" / name, filename=name)


@app.get("/api/research/study")
def controlled_study(partition: str = "locked", arm: str = "tuned_constrained", reference: str = "base_constrained"):
    from research.study_results import study_view
    docs = store.runs()
    data = study_view(docs, partition, arm, reference)
    folder = published_study(docs)
    data["exports"] = [k for k,name in STUDY_FILES.items() if folder and (folder/name).is_file()]
    return data


@app.get("/api/research/files/{kind}")
def study_file(kind: str):
    folder = published_study(store.runs())
    name = STUDY_FILES.get(kind)
    if not name or not folder or not (folder/name).is_file():
        raise HTTPException(404, "This study export is unavailable.")
    return FileResponse(folder/name, filename=name)


@app.get("/api/runs/{rid}/analysis")
def research_analysis(rid: str, partition: str | None = None, arm: str | None = None):
    from .research import analysis
    return analysis(store.run(rid), partition, arm)


@app.get("/api/runs/{rid}/configuration")
def export_configuration(rid: str):
    doc = store.run(rid)
    if doc.get("archived"):
        return FileResponse(Path(doc["archive"]) / "config.json", filename=rid+"-config.json")
    config = lifecycle.worker_config(doc)
    return JSONResponse(config, headers={"Content-Disposition": f'attachment; filename="{rid}-config.json"'})


@app.get("/api/runs/{rid}/export/{kind}")
def export(
    rid: str,
    kind: Literal["report", "adapter", "adapter-config", "metrics", "manifest"],
):
    doc = store.run(rid)
    if not doc.get("archived"):
        raise HTTPException(404, "Verified archive unavailable.")
    folder = Path(doc["archive"])
    names = {
        "report": "report.json",
        "manifest": "manifest.json",
        "metrics": "metrics.jsonl",
    }
    if kind in ("adapter", "adapter-config"):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
        ending = (
            "adapter_config.json"
            if kind == "adapter-config"
            else "adapter_model.safetensors"
        )
        name = next(
            (
                n
                for n in manifest["files"]
                if archive.ADAPTER.fullmatch(n) and n.endswith(ending)
            ),
            "",
        )
    else:
        name = names[kind]
    if not name or not (folder / name).is_file():
        raise HTTPException(404, "This run did not produce the selected file.")
    return FileResponse(folder / name, filename=Path(name).name)


if (store.ROOT / "out").is_dir():
    app.mount("/", StaticFiles(directory=store.ROOT / "out", html=True), name="studio")
