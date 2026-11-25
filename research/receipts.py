"""Original MIT-licensed receipt corpus. Template families and pair assignments are versioned."""

import json
import random
from collections import Counter
from pathlib import Path
from studio.datasets import canon, digest, stats
import re

NAME = "Receiving notes · controlled v2"
INSTRUCTION = "Extract the receipt as JSON with keys item, quantity (integer), and site. Return only JSON.\nNote: "
ITEMS = (
    "brackets",
    "washers",
    "cable ties",
    "hex bolts",
    "brass shims",
    "retaining rings",
    "nitrile gloves",
    "safety glasses",
    "paper towels",
    "floor cleaner",
    "packing tape",
    "bubble mailers",
    "USB cables",
    "patch panels",
    "bearing grease",
    "drill bits",
)
SITES = (
    "North Bay",
    "Cedar Annex",
    "East Depot",
    "Lake Store",
    "Dock 7",
    "Mill Yard",
    "Annex B",
    "West Hub",
)
TRAIN = {
    "clerk": "At {site} I counted {qty} {item} into stock. The order called for {plan}.",
    "table": "Receiving sheet\nfacility | product | accepted\n{site} | {item} | {qty}\nPO {ref}",
    "correction": "Correction to ticket {ref}: replace the earlier figure of {plan}. Final receipt at {site}: {qty} {item}.",
    "dispatch": "{source} sent {item} to {site}. This end signed for {qty} units. Transfer reference {ref}.",
    "exception": "Exception log: {other} stayed on backorder. What arrived at {site} was {item}, count {qty}.",
    "labels": "SKU description={item}\nordered={plan}\ncounted in={qty}\nship-to={site}\nfrom={source}",
    "email": "To stores: please post {qty} units of {item} against the delivery into {site}. Cheers, Receiving.",
    "handwritten": "{site} / inbound / {qty} pcs / {item}. Margin note: {other} due tomorrow.",
}
DEV = {
    "call": "Radio transcript\nQ: Delivery point?\nA: {site}.\nQ: What did you accept?\nA: {item}, {qty}.",
    "recount": "Manifest {ref} estimated {plan}. After a recount, {site} booked {qty} of {item}.",
    "mark": "Mark the completed line: [ ] {other} for {source}; [x] {qty} {item} taken in at {site}.",
}
TEST = {
    "ledger": "Dock ledger #{ref}\narticle :: posted :: requested :: destination\n{item} :: {qty} :: {plan} :: {site}",
    "handover": "Night shift handover — the carton marked {other} is still pending. I left the received {item} ({qty} units) with {site}.",
    "amendment": "Cross out “{source}, {plan}” on the provisional entry. The signed line for {item} reads “{site}, {qty} received”.",
    "waybill": "Waybill {ref} moved from {source} to {site}. Contents: {item}. The sender listed {plan}; the receiver tallied {qty}.",
}
LOCK = Path(__file__).with_name("receipt-lock.json")


def seed(*parts):
    return int(digest(parts)[:16], 16)


def build():
    families = {**TRAIN, **DEV, **TEST}
    pairs = [(i, s) for i in range(len(ITEMS)) for s in range(len(SITES))]
    seen = [p for p in pairs if (p[1] - p[0]) % 8 < 4]
    counts = Counter()

    def row(part, family, pair, novel):
        i, s = pair
        ordinal = counts[pair]
        counts[pair] += 1
        # A permutation per pair prevents any gold triple crossing partitions.
        qty = random.Random(seed("qty", i, s)).sample(range(1, 241), 32)[ordinal]
        rng = random.Random(seed(part, family, i, s, ordinal))
        data = dict(
            item=ITEMS[i],
            site=SITES[s],
            qty=qty,
            plan=qty + rng.randint(1, 17),
            other=ITEMS[(i + rng.randrange(1, len(ITEMS))) % len(ITEMS)],
            source=SITES[(s + rng.randrange(1, len(SITES))) % len(SITES)],
            ref=rng.randint(3001, 9988),
        )
        prompt = INSTRUCTION + families[family].format(**data)
        return dict(
            id=digest([part, prompt])[:16],
            prompt=prompt,
            completion=canon(
                {"item": data["item"], "site": data["site"], "quantity": qty}
            ),
            template=family,
            unseen_template=family not in TRAIN,
            unseen_pair=novel,
        )

    forms = list(TRAIN)
    train = [
        row("train", forms[(j + i + s) % len(forms)], (i, s), False)
        for i, s in seen
        for j in range(6)
    ]
    result = {"train": train}
    for split, held, offset in (("validation", list(DEV), 4), ("test", list(TEST), 6)):
        examples = []
        for novel in (False, True):
            choices = (
                [p for p in pairs if offset <= (p[1] - p[0]) % 8 < offset + 2]
                if novel
                else list(seen)
            )
            rng = random.Random(seed(split, novel))
            rng.shuffle(choices)
            for j in range(32):
                novel_form = j >= 16
                options = held if novel_form else forms
                examples.append(
                    row(split, options[j % len(options)], choices[j], novel)
                )
        result[split] = examples
    for split, rows in result.items():
        random.Random(seed("order", split)).shuffle(rows)
    return result


def manifest(bundle):
    return {
        "version": "receipt-v2.1",
        "license": "MIT",
        "bundle_sha256": digest(bundle),
        "splits": {
            s: {
                "count": len(rows),
                "sha256": digest(rows),
                "templates": dict(sorted(Counter(r["template"] for r in rows).items())),
            }
            for s, rows in bundle.items()
        },
    }


def locked_bundle():
    data = build()
    if manifest(data) != json.loads(LOCK.read_text(encoding="utf8")):
        raise ValueError("Receipt research split differs from its committed manifest.")
    return data


def metadata(data):
    train = data["train"]
    key = digest(["controlled-v2", digest(data)])
    return dict(
        id=key,
        name=NAME,
        license="Original · MIT",
        builtin=True,
        protocol="receipt-v2.1",
        seed=None,
        fraction=len(data["validation"]) / (len(train) + len(data["validation"])),
        train=len(train),
        validation=len(data["validation"]),
        test=len(data["test"]),
        rows=len(train) + len(data["validation"]),
        duplicates=0,
        preview=[{k: r[k] for k in ("prompt", "completion")} for r in train[:3]],
        fingerprint=digest(data),
        split_id=digest({s: [r["id"] for r in v] for s, v in data.items()}),
        bundle_hash=digest(data),
        tokens=stats(
            [
                len(re.findall(r"\w+|[^\w\s]", r["prompt"] + " " + r["completion"]))
                for r in train
            ]
        ),
    )
