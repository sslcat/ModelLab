"""Render a standalone figure from the local, verified size-study measurements."""

import argparse
import json
from pathlib import Path


def render(folder):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    data = json.loads((folder / "comparisons.json").read_text(encoding="utf8"))
    base = data["locked-base"]["points"]
    tuned = data["locked-tuned"]["points"]
    if (
        len(base) != 4
        or len(tuned) != 4
        or [r["id"] for r in base] != [r["id"] for r in tuned]
    ):
        raise ValueError("Four matched model sessions are required.")
    if any(r["cost"] is None for r in base):
        raise ValueError("Wait for all four posted charges.")
    colors = ["#28615c", "#3d62a6", "#a1522c", "#794b85"]
    names = [r["model"].split("2.5-")[1].replace("-Instruct", "") for r in base]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4), layout="constrained")
    fig.suptitle(
        "Specialization and model size on 256 locked receipts",
        fontweight="bold",
        fontsize=17,
    )
    a, b, c, d = axes.flat
    for i, (raw, fit, color) in enumerate(zip(base, tuned, colors, strict=True)):
        acc = raw["accuracy"]
        a.scatter(
            i - 0.12,
            acc,
            s=65,
            edgecolor=color,
            facecolor="white",
            linewidth=2,
            zorder=4,
        )
        for j, seed in enumerate(fit["seed_rows"]):
            a.scatter(
                i + 0.11 + (j - 1) * 0.04,
                seed["accuracy"],
                s=17,
                color=color,
                alpha=0.6,
                zorder=3,
            )
        a.scatter(
            i + 0.11,
            fit["accuracy"],
            s=75,
            marker="_",
            linewidth=2.8,
            color=color,
            zorder=4,
        )
        for ax, key in ((c, "latency"), (d, "cost")):
            ax.plot(
                [raw[key], fit[key]], [acc, fit["accuracy"]], color=color, alpha=0.45
            )
            ax.scatter(
                raw[key],
                acc,
                marker="o",
                s=48,
                facecolor="white",
                edgecolor=color,
                linewidth=2,
                zorder=3,
            )
            ax.scatter(
                fit[key], fit["accuracy"], marker="D", s=36, color=color, zorder=3
            )
    a.set_xticks(range(4), names)
    a.set(
        xlim=(-0.4, 3.4),
        xlabel="Qwen2.5 Instruct size",
        title="Complete-record accuracy",
    )
    a.legend(
        handles=[
            Line2D(
                [], [], marker="o", mfc="white", color="#424858", ls="", label="Base"
            ),
            Line2D([], [], marker=".", color="#424858", ls="", label="Adapter seed"),
            Line2D([], [], marker="_", color="#424858", ls="", label="Seed mean"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    candidates = next(p for p in tuned if "1.5B-" in p["model"])
    primary = next(s for s in candidates["seed_rows"] if s["seed"] == 42)["groups"]
    control = base[-1]["seed_rows"][0]["groups"]
    cats = [k for k in sorted(primary) if "pair" not in k]
    for j, k in enumerate(cats):
        for rows, delta, color, label in (
            (control, -0.18, colors[-1], "7B base"),
            (primary, 0.18, colors[1], "1.5B tuned, seed 42"),
        ):
            b.barh(
                j + delta,
                rows[k]["correct"] / rows[k]["n"],
                height=0.32,
                color=color,
                label=label if j == 0 else None,
            )
    b.set_yticks(range(len(cats)), [s.replace("_", " ").capitalize() for s in cats])
    b.set(
        xlim=(0, 1.03),
        xlabel="Complete records correct",
        title="Primary comparison by difficulty",
    )
    b.invert_yaxis()
    b.xaxis.set_major_formatter(PercentFormatter(1))
    b.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
    )
    c.set(xlabel="Median generation latency (ms)", title="Latency and locked accuracy")
    d.set(xlabel="Whole-session posted cost (USD)", title="Cost and locked accuracy")
    for ax in (a, c, d):
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Complete records correct")
    for ax in axes.flat:
        ax.grid(alpha=0.16)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    legend = [
        Line2D([], [], marker="s", color=col, ls="", label=name)
        for col, name in zip(colors, names, strict=True)
    ]
    c.legend(handles=legend, ncol=2, loc="lower right", fontsize=8, frameon=False)
    d.legend(
        handles=[
            Line2D(
                [], [], marker="o", mfc="white", color="#424858", ls="", label="Base"
            ),
            Line2D(
                [], [], marker="D", color="#424858", ls="", label="Adapter seed mean"
            ),
        ],
        loc="lower right",
        fontsize=8,
        frameon=False,
    )
    out = folder / "size-comparison.png"
    tmp = out.with_suffix(".partial")
    try:
        fig.savefig(tmp, format="png", dpi=165)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
        plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    print(render(parser.parse_args().folder))
