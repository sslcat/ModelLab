"""Small, normalized local staging records. The GPU owns the training copy."""

import csv
import hashlib
import io
import json
import math
import re
import unicodedata

LIMIT = 2 * 1024 * 1024
SECRET = re.compile(
    r"-----BEGIN [\w ]*PRIVATE KEY-----|\b(?:rpa_|hf_|gh[pousr]_|github_pat_|sk-(?:proj-)?)[a-zA-Z0-9_-]{16,}|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
)


def canon(obj):
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def digest(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def parse(content, fmt):
    if len(content.encode()) > LIMIT:
        raise ValueError("Imports must fit within 2 MiB.")
    with io.StringIO(content.lstrip("\ufeff"), newline="") as source:
        if fmt == "csv":
            reader = csv.DictReader(source, strict=True)
            if sorted(reader.fieldnames or []) != ["completion", "prompt"]:
                raise ValueError("CSV headers must be prompt and completion.")
            return list(reader)
        if fmt != "jsonl":
            raise ValueError("Choose CSV or JSONL.")
        rows = []
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON on line {number}.") from exc
        return rows


def stats(lengths):
    ordered = sorted(lengths)
    return {
        "min": min(ordered),
        "mean": round(sum(ordered) / len(ordered), 1),
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "max": max(ordered),
    }


def validate(rows, seed=42, fraction=0.15):
    if not 0.05 <= fraction <= 0.45:
        raise ValueError("Validation fraction must be between 5% and 45%.")
    by_prompt = {}
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"prompt", "completion"}:
            raise ValueError(f"Row {i} must contain exactly prompt and completion.")
        if not all(
            isinstance(v, str) and v.strip() and len(v) <= 20000 for v in row.values()
        ):
            raise ValueError(
                f"Row {i} needs two nonempty strings of at most 20,000 characters."
            )
        item = {
            k: unicodedata.normalize("NFC", v.strip().replace("\r\n", "\n"))
            for k, v in row.items()
        }
        if any(SECRET.search(v) or "\x00" in v for v in item.values()):
            raise ValueError(
                f"Row {i} contains a credential-like value or binary data."
            )
        previous = by_prompt.setdefault(item["prompt"], item)
        if item != previous:
            raise ValueError(
                f"Row {i} conflicts with an earlier answer to the same prompt."
            )
    unique = sorted(by_prompt.values(), key=canon)
    if len(unique) < 3:
        raise ValueError("Provide at least three distinct prompts.")
    ranked = sorted(unique, key=lambda row: digest([seed, row["prompt"]]))
    n = max(1, round(len(unique) * fraction))
    splits = {"train": ranked[n:], "validation": ranked[:n]}
    meta = {
        "fingerprint": digest(unique),
        "split_id": digest(splits),
        "seed": seed,
        "fraction": fraction,
        "rows": len(unique),
        "duplicates": len(rows) - len(unique),
        "train": len(unique) - n,
        "validation": n,
        "preview": unique[:3],
        "tokens": stats(
            [
                len(re.findall(r"\w+|[^\w\s]", r["prompt"] + " " + r["completion"]))
                for r in unique
            ]
        ),
    }
    return meta, splits


def dispatch_rows(test=False):
    """Original synthetic stockroom notes, with a distinct test-only paraphrase family."""
    templates = [
        "At {site}, {n} {item} came in from receiving.",
        "Receiving log: delivered {item} to {site}; count {n}.",
        "Put {n} units of {item} into the inventory at {site}.",
        "{site} checked in a delivery of {item}: {n} in all.",
    ]
    sites = ["North Bay", "Cedar Annex", "East Depot", "Lake Store"]
    products = ["washers", "hinges", "cable ties", "latches", "brackets", "gaskets"]
    rows = []
    for i in range(48 if test else 240):
        site, item = sites[i % 4], products[(i // 4) % 6]
        n = (80 if test else 1) + (i * 17 + i // 24) % 79
        test_forms = [
            "Counted at unloading — {site} received {n} pieces ({item}).",
            "Intake slip: {item} x {n}; handoff location was {site}.",
        ]
        template = test_forms[i % 2] if test else templates[(i // 24 + i % 3) % 4]
        text = template.format(site=site, item=item, n=n)
        prompt = (
            "Extract the receipt as JSON with keys item, quantity (integer), and site. Return only JSON.\nNote: "
            + text
        )
        rows.append(
            {
                "prompt": prompt,
                "completion": canon({"item": item, "quantity": n, "site": site}),
            }
        )
    return rows
