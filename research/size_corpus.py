"""Independent original receipt evaluation. Historical training rows are reused verbatim."""

import json
import random
from collections import Counter
from pathlib import Path

from research import receipts as old
from studio.datasets import canon, digest

LOCK = Path(__file__).with_name("size-lock.json")
VERSION = "size-v1"
# Each family has a unique adjudication rule in its text and one unambiguous final receipt.
DEV = {
    "scanner_retry": (
        "amendment",
        "Scanner retry {ref}. The first capture showed {plan} {other} in {source}. VOID first capture. Confirmed on the second scan: {site} took in {qty} {item}.",
    ),
    "shift_vote": (
        "ambiguity",
        "Two shifts disagree on bay {ref}. One recalls {other} at {source}; the other recalls {qty} of {item} at {site}. Use the latter account, countersigned by both. The requisition listed {plan}.",
    ),
    "dispatch_stub": (
        "distractor",
        "The truck left {source} at 07:40. Manifest {ref} called for {plan}. The receipt stub lists {item} / {site} / {qty} accepted. A separate pallet of {other} remains sealed.",
    ),
    "floor_walk": (
        "long_context",
        "Walk log {ref}. {memo} At the end of the walk the receiver posted {qty} units of {item} to {site}. {source} had requested {plan} of {other} during the planning call.",
    ),
}
TEST = {
    "carbon_copy": (
        "amendment",
        "Carbon set {ref}. Office copy (superseded): {source}, {plan} {other}. Dock copy (signed amendment): {site}, {qty} {item}. File the dock copy. Its number is {ref}.",
    ),
    "strike_sheet": (
        "crossed_out",
        "Receiving desk / sheet {ref}\n~~{other} | {plan} | {source}~~\nReplacement immediately below the strike: {item} | {qty} | {site}. The crossed line stays visible for audit.",
    ),
    "reverse_correction": (
        "crossed_out",
        "The binding entry comes FIRST: received {qty} {item} for {site}. Below is the cancelled draft which must still be legible: {plan} {other}, consignee {source}. File {ref}.",
    ),
    "two_recipients": (
        "ambiguity",
        "Both desks appear on ticket {ref}. {source} signed only for the keys. The goods receipt bears the other desk's stamp, {site}. The inscription reads {qty} {item}. Expected before arrival: {plan}.",
    ),
    "piece_count": (
        "multiple_numbers",
        "Load {ref}, scheduled {plan}, cartons {boxes}, seals {seal}. The piece column is the posted inventory unit: {qty}. Article {item}. Booked at {site}. Pickup at {source}.",
    ),
    "other_transaction": (
        "distractor",
        "Two transactions were stapled together. Issue voucher {ref}: {plan} {other} left {source}. Goods received voucher: {qty} {item} entered {site}. Only the inbound voucher increases inventory.",
    ),
    "archive_packet": (
        "long_context",
        "Archive packet {ref}. The front slip estimates {plan} {other} for {source}. {memo} On the last page the signed receipt gives product {item}, tally {qty}, and receiving site {site}.",
    ),
    "early_evidence": (
        "long_context",
        "Controlling receipt: {site} accepted {qty} {item}. {memo} Appendix {ref} repeats the obsolete planned entry, {plan} {other} at {source}. The appendix is included for audit only.",
    ),
}
MEMOS = (
    "Staff initial the register even when the load has yet to arrive. That initial records a visit and conveys no authority to post the expected quantity. Keep the outer pouch attached to the transfer paperwork until end of shift.",
    "A courier can sign the dispatch stub at the gate before a receiver has inspected the contents. Dispatch slips accompany returned packaging as well as ordinary stock. Treat the word dispatched as a movement record.",
    "The printed requisition carries the buying office address for correspondence. The delivery crew sometimes annotates the route in the margin. Accounting uses the receiving clerk's signed site entry when writing the stock movement.",
    "During the handover the earlier copy stays in the folder for reference. The filing clerk retains even an unsigned estimate and a withdrawn voucher so a reviewer can reconstruct the sequence the following morning.",
    "Route labels and pallet tags travel with the vehicle after a split delivery. Staff keep a photocopy under the route tab. That tab can mention an earlier destination served by the same vehicle.",
    "The buyer's file also holds advance requests for the following week. The requests share a load reference because the carrier invoices by trip. Only a receiver can confirm goods taken into the store.",
)


def build():
    historical = old.locked_bundle()
    train = historical["train"]
    banned = {r["completion"] for rows in historical.values() for r in rows}
    seen = {
        (x["item"], x["site"]) for x in (json.loads(r["completion"]) for r in train)
    }
    unseen = [(i, s) for i in old.ITEMS for s in old.SITES if (i, s) not in seen]
    random.Random(31991).shuffle(unseen)
    held = {"validation": unseen[:32], "test": unseen[32:]}
    data = {"train": train}
    for split, families, per in (("validation", DEV, 16), ("test", TEST, 32)):
        # Balanced blocks independently shuffle pairs for every family.
        rows = []
        for family, (category, pattern) in families.items():
            rng = random.Random(old.seed(VERSION, split, family))
            for novel, pool in ((False, sorted(seen)), (True, held[split])):
                for item, site in rng.sample(pool, per // 2):
                    while True:
                        qty = rng.randint(1, 240)
                        completion = canon(
                            {"item": item, "quantity": qty, "site": site}
                        )
                        if completion not in banned:
                            banned.add(completion)
                            break
                    other = rng.choice([i for i in old.ITEMS if i != item])
                    source = rng.choice([s for s in old.SITES if s != site])
                    memo = (
                        " ".join(rng.sample(MEMOS, 3))
                        if category == "long_context"
                        else ""
                    )
                    body = pattern.format(
                        item=item,
                        qty=qty,
                        site=site,
                        other=other,
                        source=source,
                        ref=rng.randint(12000, 99500),
                        plan=qty + rng.randint(2, 25),
                        boxes=rng.randint(2, 18),
                        seal=rng.randint(300, 900),
                        memo=memo,
                    )
                    prompt = old.INSTRUCTION + body
                    rows.append(
                        {
                            "id": digest([VERSION, split, prompt])[:20],
                            "prompt": prompt,
                            "completion": completion,
                            "template": family,
                            "difficulty": category,
                            "unseen_template": True,
                            "unseen_pair": novel,
                        }
                    )
        random.Random(old.seed(VERSION, split, "order")).shuffle(rows)
        data[split] = rows
    return data


def manifest(data):
    return {
        "version": VERSION,
        "license": "MIT",
        "bundle_sha256": digest(data),
        "splits": {
            s: {
                "count": len(rows),
                "sha256": digest(rows),
                "templates": dict(sorted(Counter(x["template"] for x in rows).items())),
                "difficulty": dict(
                    sorted(
                        Counter(
                            x["difficulty"] for x in rows if "difficulty" in x
                        ).items()
                    )
                ),
                "unseen_pairs": sum(x["unseen_pair"] for x in rows),
            }
            for s, rows in data.items()
        },
    }


def locked_bundle():
    data = build()
    if manifest(data) != json.loads(LOCK.read_text(encoding="utf8")):
        raise ValueError("Size study corpus differs from its preregistered lock.")
    return data


def metadata(data):
    m = old.metadata(data)
    key = digest([VERSION, digest(data)])
    return dict(
        m, id=key, name="Receiving notes · size and specialization", protocol=VERSION
    )
