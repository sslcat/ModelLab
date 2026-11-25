"""Render the controlled study results figure from the verified CSV exports."""

import argparse
import csv
from pathlib import Path

LABELS = {
    "base": "Base",
    "prompted": "Improved prompt",
    "base_constrained": "Constrained base",
    "tuned": "Tuned",
    "tuned_constrained": "Constrained tuned",
}


def render(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    def read(name):
        with (folder / name).open(encoding="utf8", newline="") as f:
            return list(csv.DictReader(f))

    arms = [
        r for r in read("arms.csv") if r["rank"] == "8" and r["partition"] == "locked"
    ]
    paired = [
        r
        for r in read("paired.csv")
        if r["reference"] == "base_constrained"
        and r["candidate"] == "tuned_constrained"
        and r["partition"] == "locked"
        and r["metric"] == "exact"
    ]
    if len(arms) != 5 or len(paired) != 4:
        raise ValueError("A complete verified study export is required.")
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, (ax, bx) = plt.subplots(
            1,
            2,
            figsize=(11.5, 4.8),
            layout="constrained",
            gridspec_kw={"width_ratios": [1.3, 1]},
        )
        fig.suptitle(
            "Receipt extraction on 64 locked notes", fontsize=16, fontweight="bold"
        )
        for i, r in enumerate(arms):
            n = int(r["n"])
            strict = int(r["exact"]) / n
            content = int(r["content"]) / n
            ax.barh(
                i - 0.17,
                strict,
                height=0.32,
                color="#344fa6",
                label="Raw JSON record" if not i else None,
            )
            ax.barh(
                i + 0.17,
                content,
                height=0.32,
                color="#66a4a3",
                label="After fence removal" if not i else None,
            )
        ax.set(
            yticks=range(5),
            yticklabels=[LABELS[r["arm"]] for r in arms],
            xlim=(0, 1.02),
            xlabel="Complete records correct",
        )
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title("Rank 8 across inference conditions", loc="left", fontsize=12)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=2,
            frameon=False,
            fontsize=9,
        )
        bx.axvline(0, color="#747b87", linestyle="--", linewidth=1)
        for i, r in enumerate(paired):
            point, lo, hi = [100 * float(r[k]) for k in ("delta", "lower", "upper")]
            bx.errorbar(
                point,
                i,
                xerr=[[point - lo], [hi - point]],
                fmt="o",
                color="#344fa6" if r["rank"] == "8" else "#588082",
                capsize=4,
                linewidth=2,
                markersize=7,
            )
        bx.set(
            yticks=range(4),
            yticklabels=["Rank " + r["rank"] for r in paired],
            xlabel="Paired difference (percentage points)",
        )
        bx.invert_yaxis()
        bx.set_title("Tuned minus base under the same grammar", loc="left", fontsize=12)
        for a in [ax, bx]:
            a.set_axisbelow(True)
            a.grid(axis="x", alpha=0.18)
        bx.text(
            0.5,
            -0.23,
            "Conservative 95% note-level intervals · one training seed",
            fontsize=8.5,
            ha="center",
            transform=bx.transAxes,
        )
        dest = folder / "rank-comparison.png"
        tmp = folder / "rank-comparison.partial"
        try:
            fig.savefig(tmp, format="png", dpi=175, bbox_inches="tight")
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
            plt.close(fig)
        return dest


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("folder", type=Path)
    print(render(arg.parse_args().folder))
