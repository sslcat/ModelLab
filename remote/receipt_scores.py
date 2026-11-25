"""Receipt extraction scores, shared by the studio, offline audit and GPU worker."""

import json
import math
import re
from collections import Counter

FIELDS = ("item", "quantity", "site")
SCHEMA = {
    "type": "object",
    "properties": {
        "item": {"type": "string"},
        "quantity": {"type": "integer"},
        "site": {"type": "string"},
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}
FENCE = re.compile(r"\x60{3}(?:json)?[ \t]*\r?\n(.*?)\r?\n\x60{3}", re.I | re.S)


def reject(value):
    raise ValueError(value)


def parse(text):
    duplicates = []

    def obj(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=obj, parse_constant=reject)
    except (ValueError, RecursionError):
        return False, None, False
    return True, value, bool(duplicates)


def score(prediction, expected):
    gold = json.loads(expected)
    valid, value, duplicate = parse(prediction)
    obj = value if isinstance(value, dict) and not duplicate else {}
    fields = {
        k: type(obj.get(k)) is type(gold[k]) and obj[k] == gold[k] for k in FIELDS
    }
    schema = (
        valid
        and not duplicate
        and set(obj) == set(FIELDS)
        and all(type(obj[k]) is (int if k == "quantity" else str) for k in FIELDS)
    )
    exact = schema and all(fields.values())
    fence = FENCE.fullmatch(prediction.strip())
    if fence:
        content = score_unwrapped(fence[1], gold)
    else:
        content = exact
    errors = []
    if not valid:
        errors.append("Markdown fence" if fence else "Invalid JSON")
    elif not schema:
        errors.append("Schema violation")
    if valid and isinstance(value, dict) and not duplicate:
        errors.extend("Wrong " + k for k in FIELDS if not fields[k])
    return dict(
        valid=valid,
        schema=schema,
        fields=fields,
        exact=exact,
        content=content,
        errors=errors,
    )


def score_unwrapped(text, gold):
    ok, obj, dup = parse(text)
    return bool(
        ok
        and not dup
        and isinstance(obj, dict)
        and set(obj) == set(FIELDS)
        and all(type(obj[k]) is type(gold[k]) and obj[k] == gold[k] for k in FIELDS)
    )


def summarize(examples):
    rows = [score(x["prediction"], x["expected"]) for x in examples]
    latency = sorted(
        x["latency_ms"]
        for x in examples
        if isinstance(x.get("latency_ms"), (float, int))
        and x["latency_ms"] >= 0
        and math.isfinite(x["latency_ms"])
    )
    errors = Counter(e for r in rows for e in r["errors"])
    unseen = [r for r, x in zip(rows, examples) if x.get("unseen_template") is True]
    return {
        "n": len(rows),
        **{k: sum(r[k] for r in rows) for k in ("valid", "schema", "exact", "content")},
        "fields": {k: sum(r["fields"][k] for r in rows) for k in FIELDS},
        "unseen_n": len(unseen),
        "unseen_exact": sum(r["exact"] for r in unseen),
        "errors": dict(errors),
        "latency_n": len(latency),
        "latency_p50_ms": latency[math.ceil(len(latency) * 0.5) - 1]
        if latency
        else None,
        "latency_p95_ms": latency[math.ceil(len(latency) * 0.95) - 1]
        if latency
        else None,
    }
