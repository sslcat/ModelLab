"""Render the consolidated report directly from the bundled public evidence."""

import argparse
from pathlib import Path
from research.public_results import recompute
from research.study_results import matched, paired_counts
from studio.demo import records, sizes

ROOT = Path(__file__).resolve().parents[1]


def table(head, rows):
    return "\n".join(
        ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def report():
    s, cells = recompute()
    data = records()
    lookup = {(r["session"], r["partition"], r["arm"]): r for r in cells}
    bill = lambda group: sum(
        r["actual"]["totalAmount"] for r in data.values() if r["study"] == group
    )
    base = lookup["pilot", "locked", "base"]
    fit = lookup["pilot", "locked", "tuned"]
    primary = s["primary"]["locked"]
    dev = s["primary"]["development"]
    rankstats = []
    for k in (4, 8, 16, 32):
        rid = "rank-" + str(k)
        t = lookup[rid, "locked", "tuned_constrained"]
        a = data[rid]["seeds"]["42"]
        rankstats.append(
            [
                k,
                2 * k,
                lookup[rid, "locked", "base_constrained"]["exact"],
                t["exact"],
                lookup[rid, "development", "tuned_constrained"]["exact"],
                f"{a['trainable']:,}",
                f"{a['adapter_bytes']:,}",
            ]
        )
    ranked = table(
        [
            "Rank",
            "Alpha",
            "Locked base / 64",
            "Locked tuned / 64",
            "Development tuned / 64",
            "Trainable parameters",
            "Adapter bytes",
        ],
        rankstats,
    )
    sizes_table = table(
        [
            "Size",
            "Base / 256",
            "Seed 42",
            "Seed 73",
            "Seed 101",
            "Mean",
            "Seed SD (pp)",
        ],
        [
            [
                r["size"].upper(),
                r["base"],
                r["seed_42"],
                r["seed_73"],
                r["seed_101"],
                f"{100 * r['mean']:.2f}%",
                f"{100 * r['sample_sd']:.2f}",
            ]
            for r in s["size"]
        ],
    )
    comparison = sizes()
    small = next(p for p in comparison["points"] if "1.5B-" in p["model"])
    left = next(a for a in small["seed_rows"] if a["seed"] == 42)
    large = next(p for p in sizes(condition="base")["points"] if "7B-" in p["model"])[
        "seed_rows"
    ][0]
    groups = table(
        ["Locked group", "Notes", "7B base correct", "1.5B seed 42 correct"],
        [
            [k.replace("_", " "), v["n"], large["groups"][k]["correct"], v["correct"]]
            for k, v in sorted(left["groups"].items())
        ],
    )
    p = paired_counts(7, 0, 64)
    arm8 = data["rank-8"]["evaluation"]["partitions"]["locked"]
    r = matched(
        arm8["base_constrained"]["examples"], arm8["tuned_constrained"]["examples"]
    )
    assert r == p
    return f"""# What changed when I fine-tuned the extractor?

I ran three studies of the same task. Given an inventory receipt, extract the item, integer quantity and receiving site. I kept the base generations, failed outputs, adapter predictions, training signals and bill together so I could distinguish a usable JSON response from a correct inventory record.

## Study 1 · Output formatting

I trained one rank-8 Qwen2.5-7B-Instruct adapter on 204 receipts and validated on 36. The two models faced the same 48 separate receipts with greedy decoding. The base produced **{base["exact"]}/48** strict exact matches and the adapter **{fit["exact"]}/48**. All 48 base responses had enclosing Markdown fences. As a post-hoc diagnostic, I stripped one enclosing fence and rescored the same outputs. The counts became **{base["normalized"]}/48 and {fit["normalized"]}/48**. Only one additional extraction became correct. All seven remaining errors were item substitutions on receipts for brackets. The low validation loss measured next-token prediction on familiar wording. It offered little protection against that recurring field error. The pilot includes a failed initial allocation with zero training steps and a successful retry.

## Study 2 · Rank and parameter efficiency

I compared ranks 4, 8, 16 and 32 with the same 384 training receipts, 64 development receipts, 64 locked receipts, seed 42, query/value targets and alpha-to-rank ratio of two. Each session recorded original base, improved-prompt base, constrained base, tuned and constrained tuned controls on both partitions. That is {s["counts"]["rank"]:,} predictions. The primary contrast was rank 8 under identical JSON constraints. Both models passed the schema on all 64 locked notes.

{ranked}

For the rank-8 locked controls, original base prompting yielded {lookup["rank-8", "locked", "base"]["exact"]}/64 strict complete records. The improved base prompt yielded {lookup["rank-8", "locked", "prompted"]["exact"]}/64, and the tuned model without the grammar yielded {lookup["rank-8", "locked", "tuned"]["exact"]}/64. Better prompting alone recovered much of the initial formatting loss.

The constrained base scored 56/64 in every session. Every constrained adapter scored 63/64 and missed the same amendment. For rank 8 there were {r["gains"]} gains and {r["losses"]} regressions, a {100 * r["delta"]:.2f}-point change. The exact two-sided McNemar p-value was {r["exact_p"]:.5f}. The conservative paired 95% interval was {100 * r["lower"]:.2f} to {100 * r["upper"]:.2f} points. These two procedures use different constructions. Rank 4 achieved the observed score with the fewest trained parameters and smallest adapter file. I used it for the size study. There was one training seed, 64 test notes and no registered equivalence margin. Identical observed scores leave equality of population performance unresolved.

## Study 3 · Model size and specialization

Each size had one constrained quantized base and three freshly trained rank-4 adapters, using seeds 42, 73 and 101. Each arm scored 64 development and 256 independently locked receipts. This produced {s["counts"]["size"]:,} generations. Training data stayed fixed. Every locked family was unseen during training and development. Half the locked notes used unseen item–site combinations. Item and site vocabulary remained familiar.

{sizes_table}

The preregistered specialist, 1.5B at seed 42, extracted **{left["correct"]}/256 ({100 * left["accuracy"]:.2f}%)**, versus **{large["correct"]}/256 ({100 * large["accuracy"]:.2f}%)** for 7B base. There were {primary["gains"]} gains, {primary["losses"]} regressions and {primary["ties"]} ties. The paired difference was **{100 * primary["delta"]:.2f} percentage points**, conservative 95% interval **{100 * primary["lower"]:.2f} to +{100 * primary["upper"]:.2f}**, exact two-sided p = {primary["exact_p"]:.5f}. On development the direction reversed. The specialist scored 32/64 and 7B base 62/64, with {dev["losses"]} regressions and {dev["gains"]} gains. These data do not establish superiority or equivalence.

{groups}

The primary specialist recovered more superseded entries and signed amendments. On competing-transaction distractors it correctly extracted just one of 32 records. The larger base recovered 23. The six difficulty categories partition the 256 notes. The two item–site strata partition the same notes a second way. All three tuned 3B models fell below their own 130/256 base, even though their development scores improved. The tuned 7B had the highest locked accuracy at each matched seed. Three seeds quantify some optimizer variation. They do not multiply the number of independent receipts.

## Costs, measurements and uncertainty

| Study | Scored generations | Posted GPU, CPU and storage charges (USD) |
| --- | ---: | ---: |
| Formatting | {s["counts"]["pilot"]} | {bill("pilot"):.5f} |
| Rank | {s["counts"]["rank"]:,} | {bill("rank"):.5f} |
| Size | {s["counts"]["size"]:,} | {bill("size"):.5f} |
| Total | {sum(s["counts"].values()):,} | **{s["actual_usd"]:.5f}** |

Costs are posted per pod session, including setup, shared base generation and disk. The pilot's two postings came from the historical ledger. The later sessions also have archived billing files. On the locked set, the observed joint accuracy, latency and cost frontier comprised the 3B base, tuned 3B and tuned 7B. The tuned 3B was fastest on these hosts and its session cost the least. The tuned 7B had the highest average accuracy. All systems shared the RTX 4090 SKU, with a different physical host per model size. Generated output length and host performance affect these measurements.

All three corpora consist of authored synthetic English receipts. Family mix, finite test size, fixed vocabulary and exact string matching limit transfer to real documents. Multiple notes share templates. The note-level intervals can understate uncertainty across template families. Both arms of the constrained controls used the same four-bit backbone. This design cannot identify the effect of quantization relative to full precision. Model size also varies with architecture and pretraining. One rank, optimizer recipe and short training schedule cannot determine each size’s best achievable accuracy. Exploring secondary seeds, ranks and subgroups introduced multiple comparisons. Only the registered contrast in each controlled study was primary.

## Trace the results

[Every arm’s split, denominator, schema, field and exact hits](../evidence/scores.csv) · [Paired statistics, fingerprints, adapter sizes and per-session charges](../evidence/summary.json) · [Full public observations](../evidence/README.md) · [Mathematical definitions](mathematics.md)

From the repository root, run `python -m research.public_results` to verify the released tables. Run `python -m research.report --check` to regenerate this report in memory and compare it with the checked-in text.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    file = ROOT / "docs/research.md"
    value = report()
    if args.check:
        if file.read_text("utf8") != value:
            raise SystemExit("Report differs from measured evidence")
    else:
        file.write_text(value, encoding="utf8", newline="\n")
