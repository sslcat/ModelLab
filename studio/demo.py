"""Public read-only application. It has no database or cloud lifecycle startup hooks."""

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from remote.receipt_scores import score
from research import size_results, study_results
from studio.config import BUDGET, MODELS
from studio.research import analysis

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
app = FastAPI(title="ModelLab · historical research", version="1.0")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
)


@lru_cache(maxsize=1)
def records():
    result = {}
    order = [
        "size-7b",
        "size-3b",
        "size-1.5b",
        "size-0.5b",
        "rank-32",
        "rank-16",
        "rank-8",
        "rank-4",
        "pilot",
        "pilot-attempt",
    ]
    for name in order:
        path = EVIDENCE / (name + ".json.gz")
        with gzip.open(path, "rt", encoding="utf8") as stream:
            record = json.load(stream)
        if record["id"] + ".json.gz" != path.name:
            raise ValueError("Public evidence identifier mismatch")
        result[record["id"]] = record
    return result


def lookup(rid):
    if rid not in records():
        raise HTTPException(404, "Unknown public experiment")
    return records()[rid]


@app.middleware("http")
async def read_only(req: Request, call_next):
    if req.method not in ("HEAD", "GET", "OPTIONS"):
        return JSONResponse({"detail": "Historical workspace is read-only."}, 405)
    return await call_next(req)


@app.exception_handler(ValueError)
async def invalid(req, exc):
    return JSONResponse({"detail": str(exc)}, 400)


@app.get("/api/studio")
def overview():
    runs = []
    for r in records().values():
        row = {
            k: r[k]
            for k in (
                "id",
                "status",
                "config",
                "dataset_name",
                "metrics",
                "actual",
                "archived",
                "bundle_hash",
            )
        }
        if r.get("size_study"):
            row["size_study"] = True
        if r.get("research_design"):
            row["research_design"] = True
        row.update(
            evaluation={k: r["evaluation"][k] for k in ("split", "limitations")}
            if r["evaluation"]
            else None,
            estimated_usage=r["public_resources"]["estimated_usd"],
            pod=None,
            quote=None,
            detail="",
            adapter_available=False,
            historical=True,
        )
        runs.append(row)
    adapters = [
        dict(
            run=r["id"],
            model=r["config"]["model"],
            rank=r["config"]["rank"],
            seed=int(seed),
            bytes=a["adapter_bytes"],
            trainable=a["trainable"],
            sha256=a["adapter_sha256"],
        )
        for r in records().values()
        for seed, a in r["seeds"].items()
    ]
    return dict(
        mode="demo",
        runs=runs,
        recorded_adapters=adapters,
        datasets=json.loads((EVIDENCE / "datasets.json").read_text("utf8")),
        models=MODELS,
        budget=BUDGET,
        reserved=sum(r["actual"]["totalAmount"] for r in runs),
        active=0,
        credential=False,
        one_drive="",
        artifact_root="Bundled public research evidence",
    )


@app.get("/api/runs/{rid}/analysis")
def analysis_view(rid: str, partition: str | None = None, arm: str | None = None):
    return analysis(lookup(rid), partition, arm)


@app.get("/api/runs/{rid}/configuration")
def configuration(rid: str):
    record = lookup(rid)
    return JSONResponse(
        dict(record["config"], **record["execution"]),
        headers={"Content-Disposition": f'attachment; filename="{rid}-config.json"'},
    )


@app.get("/api/runs/{rid}/export/{kind}")
def export(rid: str, kind: Literal["report", "metrics", "manifest"]):
    r = lookup(rid)
    data = (
        r
        if kind == "report"
        else r["metrics"]
        if kind == "metrics"
        else {
            "public_file": rid + ".json.gz",
            "sha256": json.loads((EVIDENCE / "sha256.json").read_text())[
                rid + ".json.gz"
            ],
            "adapters": {
                s: {
                    k: v
                    for k, v in x.items()
                    if k in ("adapter_bytes", "adapter_sha256", "trainable")
                }
                for s, x in r["seeds"].items()
            },
        }
    )
    return JSONResponse(
        data,
        headers={"Content-Disposition": f'attachment; filename="{rid}-{kind}.json"'},
    )


@app.get("/api/research/study")
def ranks(
    partition: str = "locked",
    arm: str = "tuned_constrained",
    reference: str = "base_constrained",
):
    return study_results.study_view(records().values(), partition, arm, reference)


@app.get("/api/research/model-size")
def sizes(
    benchmark: str = "mission3", partition: str = "locked", condition: str = "tuned"
):
    return size_results.view(list(records().values()), benchmark, partition, condition)


@app.get("/api/research/model-size/{rid}/cases")
def cases(
    rid: str,
    partition: Literal["locked", "development"],
    arm: str,
    failures: bool = False,
    offset: int = 0,
    difficulty: str | None = None,
):
    doc = lookup(rid)
    if not (
        doc.get("size_study")
        or doc.get("research_design")
        and doc["config"]["rank"] == 4
    ):
        raise HTTPException(404, "Unknown benchmark")
    arms = size_results.arms_for(doc, partition)
    if arm not in arms or offset < 0:
        raise ValueError("Invalid prediction selection")
    rows = [
        r
        for r in arms[arm]
        if (not failures or not r["correct"])
        and (not difficulty or r.get("difficulty", r["template"]) == difficulty)
    ]
    return {
        "total": len(rows),
        "examples": [
            dict(r, errors=score(r["prediction"], r["expected"])["errors"])
            for r in rows[offset : offset + 20]
        ],
    }


@app.get("/api/evidence/{rid}")
def evidence(rid: str):
    lookup(rid)
    return FileResponse(EVIDENCE / (rid + ".json.gz"), filename=rid + ".json.gz")


if (ROOT / "out").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "out", html=True), name="studio")
