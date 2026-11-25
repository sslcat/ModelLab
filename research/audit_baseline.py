"""Read-only audit of the original paired pilot. Run as python -m research.audit_baseline."""

import json
from collections import Counter
from pathlib import Path
from studio import archive
from studio.cloud import artifact_root
from studio.datasets import dispatch_rows, validate, digest
from remote.receipt_scores import FENCE, score, summarize

LOCK = Path(__file__).with_name("baseline-lock.json")


def audit(folder):
    files = {
        f.relative_to(folder).as_posix(): archive.sumfile(f)
        for f in sorted(folder.rglob("*"))
        if f.is_file()
    }
    if LOCK.exists() and any(
        files.get(k) != v for k, v in json.loads(LOCK.read_text()).items()
    ):
        raise ValueError("An original baseline artifact differs from its lock.")
    ev = json.loads((folder / "evaluation.json").read_text(encoding="utf8"))
    cfg = json.loads((folder / "config.json").read_text(encoding="utf8"))
    _, bundle = validate(dispatch_rows())
    bundle["test"] = dispatch_rows(True)
    assert digest(bundle) == cfg["bundle_hash"]
    index = {row["prompt"]: i for i, row in enumerate(dispatch_rows())}

    def template(row):
        i = index[row["prompt"]]
        return (i // 24 + i % 3) % 4

    splits = {}
    for key, rows in bundle.items():
        values = [json.loads(r["completion"]) for r in rows]
        counts = Counter(
            i % 2 if key == "test" else template(r) for i, r in enumerate(rows)
        )
        splits[key] = dict(
            rows=len(rows),
            templates=dict(counts),
            items=len({v["item"] for v in values}),
            sites=len({v["site"] for v in values}),
            pairs=len({(v["item"], v["site"]) for v in values}),
            q_range=[
                min(v["quantity"] for v in values),
                max(v["quantity"] for v in values),
            ],
        )
    metrics = {}
    by_item = {}
    errors = []
    cases = []
    paired = Counter()
    for arm in ("base", "tuned"):
        rows = ev[arm]["examples"]
        assert [(r["prompt"], r["expected"]) for r in rows] == [
            (v["prompt"], v["completion"]) for v in bundle["test"]
        ]
        metrics[arm] = summarize([dict(r, unseen_template=True) for r in rows])
        by_item[arm] = {}
        for i, r in enumerate(rows):
            fence = FENCE.fullmatch(r["prediction"].strip())
            s = score(fence[1] if fence else r["prediction"], r["expected"])
            cases.append(
                dict(
                    arm=arm,
                    row=i + 1,
                    reference=json.loads(r["expected"]),
                    strict=score(r["prediction"], r["expected"]),
                    unfenced=s,
                    had_fence=bool(fence),
                )
            )
            target = json.loads(r["expected"])
            item = target["item"]
            group = by_item[arm].setdefault(item, Counter())
            group.update(
                {
                    "n": 1,
                    "item": s["fields"]["item"],
                    "quantity": s["fields"]["quantity"],
                    "site": s["fields"]["site"],
                    "content": s["exact"],
                }
            )
            if not s["exact"]:
                errors.append(
                    dict(
                        arm=arm, row=i + 1, reference=target, prediction=r["prediction"]
                    )
                )
        assert metrics[arm]["exact"] == ev[arm]["correct"]
    for a, b in zip(ev["base"]["examples"], ev["tuned"]["examples"]):
        paired[
            str(
                (
                    score(a["prediction"], a["expected"])["content"],
                    score(b["prediction"], b["expected"])["content"],
                )
            )
        ] += 1
    repeats = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in ("prompt", "completion"):
            repeats[a + "/" + b + "/" + field] = len(
                {r[field] for r in bundle[a]} & {r[field] for r in bundle[b]}
            )
    return dict(
        files=files,
        splits=splits,
        overlap=repeats,
        metrics=metrics,
        by_item=by_item,
        paired_content=dict(paired),
        failures=errors,
        cases=cases,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify the original private pilot archive against its frozen digest list.")
    parser.add_argument("folder", type=Path)
    print(json.dumps(audit(parser.parse_args().folder), indent=2, ensure_ascii=True))
