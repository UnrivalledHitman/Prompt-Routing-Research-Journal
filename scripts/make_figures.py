"""Draw the results figures from an analysis directory. CPU only.

Reads the CSVs written by ``scripts/run_analysis.py`` and writes one PDF per figure and
unit, sized for one column of an IEEE two-column page. A figure whose input is missing
or empty is skipped with a warning, so the script also runs on partial results.

Usage::

    python scripts/make_figures.py results/processed/analysis/main --out paper/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from arbiter.eval.labels import CELL_NAMES  # noqa: E402

LOGGER = logging.getLogger("arbiter.make_figures")
COLUMN_IN = 3.45
SIZES = ("0.6b", "1.7b", "4b", "8b")
SCOPE_TEXT = {"all": "All", "math": "Math", "code": "Code", "mcq": "MCQ", "open": "Open-ended"}
UNIT_TEXT = {
    "energy_j": ("dynamic GPU energy", "J"),
    "board_j": ("GPU board energy", "J"),
    "tokens": ("tokens", "tokens"),
    "tokens_output_heavy": ("tokens, output priced 4x", "tokens"),
    "latency_s": ("latency", "s"),
}
POLICIES = ("P1", "P2", "P3", "P4-uniform", "random")
ROUTERS = {
    "R1a-tfidf-lr": ("TF-IDF + LR", "C0"),
    "R1b-features-lightgbm": ("Features + LightGBM", "C1"),
    "R1b-features-xgboost": ("Features + XGBoost", "C2"),
    "R1c-minilm-lr": ("MiniLM + LR", "C3"),
    "R2-distilbert": ("DistilBERT", "C7"),
    "R2-deberta-v3-small": ("DeBERTa-v3-small", "C8"),
    "R3-qwen3-0.6b-zero-shot": ("Qwen3-0.6B, zero-shot", "C4"),
    "R3-qwen3-0.6b-few-shot": ("Qwen3-0.6B, few-shot", "C5"),
    "R3-qwen3-0.6b-lora": ("Qwen3-0.6B, LoRA", "C9"),
    "cascade-automix": ("AutoMix-style cascade", "C6"),
}
ERROR_KW = {"elinewidth": 0.6, "capsize": 1.5}
LABEL_OFFSETS = ((2.5, 2.5), (2.5, -8.0), (-2.5, 2.5), (-2.5, -8.0), (2.5, 8.5))
BELOW_FIRST = ((2.5, -8.0), (-2.5, -8.0), (2.5, 2.5), (-2.5, 2.5), (2.5, -15.0))
PER_THOUSAND = 1000.0  # the public arm prices a query in dollars; the figure uses 1,000 queries


def style() -> None:
    """Set fonts and sizes for an IEEE column."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def read(analysis: Path, name: str) -> pd.DataFrame | None:
    """Return one analysis table, or None with a warning if it was not written."""
    path = analysis / f"{name}.csv"
    if not path.exists():
        LOGGER.warning("%s not found; dependent figures skipped", path.name)
        return None
    return pd.read_csv(path)


def usable(frame: pd.DataFrame | None, what: str) -> bool:
    """Return whether a filtered table has rows, warning if not."""
    if frame is None or frame.empty:
        LOGGER.warning("no rows for %s; skipped", what)
        return False
    return True


def save(fig: plt.Figure, out: Path, name: str) -> None:
    """Write a figure as PDF and close it."""
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s.pdf", name)


def marker_boxes(ax: plt.Axes, points: Sequence[tuple[float, float]], pad: float = 3.0) -> list:
    """Return a small box around each marker, so a label is not placed on a data point."""
    boxes = []
    for x, y in points:
        px, py = ax.transData.transform((x, y))
        boxes.append(Bbox.from_extents(px - pad, py - pad, px + pad, py + pad))
    return boxes


def label_points(
    ax: plt.Axes,
    points: Sequence[tuple[float, float]],
    texts: Sequence[str],
    *,
    placed: list,
    fontsize: float = 6,
    color: str = "black",
    offsets: Sequence[tuple[float, float]] = LABEL_OFFSETS,
) -> None:
    """Annotate points, taking the first offset whose text clears everything placed already.

    Matplotlib has no collision logic, and the frontier crowds several policies into one
    corner, where the default offset prints one label over another. Trying a short fixed
    list of offsets in order keeps the placement deterministic -- the same table always
    draws the same figure -- while keeping the labels legible. A candidate is taken only
    if it also falls inside the axes, so a label near an edge turns inwards instead of
    printing over a spine. A point with no clear offset keeps the first one rather than
    wandering far from its marker.
    """
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for (x, y), text in zip(points, texts, strict=True):
        for dx, dy in offsets:
            note = ax.annotate(
                text,
                (x, y),
                fontsize=fontsize,
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                color=color,
                zorder=6,
            )
            box = note.get_window_extent(renderer=renderer)
            inside = ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
            if inside and not any(box.overlaps(other) for other in placed):
                break
            note.remove()
        else:
            dx, dy = offsets[0]
            note = ax.annotate(
                text, (x, y), fontsize=fontsize, xytext=(dx, dy), textcoords="offset points",
                color=color, zorder=6,
            )  # fmt: skip
            box = note.get_window_extent(renderer=renderer)
        placed.append(box)


def cost_label(unit: str, prefix: str) -> str:
    """Return an axis label such as ``Net mean cost (J)``."""
    return f"{prefix} ({UNIT_TEXT.get(unit, (unit, unit))[1]})"


def cell_text(cell: str) -> str:
    """Return ``$c_i$`` for a cell name and ``none`` for unservable prompts."""
    return f"$c_{cell[1:]}$" if cell in CELL_NAMES else "none"


def scopes_in(frame: pd.DataFrame) -> list[str]:
    """Return ``all`` followed by the families present, in a fixed order."""
    present = set(frame["scope"])
    return [scope for scope in SCOPE_TEXT if scope in present]


def yerr(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Return asymmetric error-bar lengths from ``name_lo`` and ``name_hi``."""
    low = (frame[name] - frame[f"{name}_lo"]).to_numpy(dtype=float)
    high = (frame[f"{name}_hi"] - frame[name]).to_numpy(dtype=float)
    return np.nan_to_num(np.vstack([low, high]).clip(min=0.0))


def fig_cstar(frame: pd.DataFrame, unit: str, out: Path) -> None:
    """Share of prompts by minimum sufficient cell, overall and per family."""
    sub = frame[frame["unit"] == unit]
    if not usable(sub, f"c* distribution in {unit}"):
        return
    order = [*CELL_NAMES, "unservable"]
    scopes = scopes_in(sub)
    shares = sub.pivot_table(index="scope", columns="cstar", values="share", aggfunc="sum")
    shares = shares.reindex(index=scopes, columns=order).fillna(0.0)
    colors = [*plt.cm.viridis(np.linspace(0.0, 0.95, len(CELL_NAMES))), (0.78, 0.78, 0.78, 1.0)]
    y = np.arange(len(scopes))
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 0.3 * len(scopes) + 0.9))
    left = np.zeros(len(scopes))
    for column, color in zip(order, colors, strict=True):
        values = shares[column].to_numpy()
        ax.barh(y, values, left=left, color=color, height=0.7, label=cell_text(column))
        left += values
    ax.set_yticks(y, [SCOPE_TEXT[s] for s in scopes])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share of prompts")
    ax.legend(
        ncol=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        handlelength=0.8,
        columnspacing=0.5,
        handletextpad=0.3,
    )
    save(fig, out, f"cstar_{unit}")


def fig_tax(frame: pd.DataFrame, unit: str, out: Path) -> None:
    """Escalation and stickiness terms of the tax, per policy and detection regime."""
    sub = frame[(frame["scope"] == "all") & (frame["unit"] == unit)]
    sub = sub[sub["policy"].isin(POLICIES)]
    if not usable(sub, f"tax in {unit}"):
        return
    detections = [d for d in ("oracle", "illustrative") if d in set(sub["detection"])]
    x = np.arange(len(POLICIES))
    width = 0.8 / len(detections)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.0))
    for k, detection in enumerate(detections):
        rows = sub[sub["detection"] == detection].set_index("policy").reindex(POLICIES)
        pos = x - 0.4 + width * (k + 0.5)
        hatch = None if detection == "oracle" else "////"
        suffix = "oracle detection" if detection == "oracle" else "imperfect detection"
        ax.bar(
            pos,
            rows["escalation"],
            width,
            color="C0",
            hatch=hatch,
            edgecolor="white",
            linewidth=0.0,
            label=f"Escalation, {suffix}",
        )
        ax.bar(
            pos,
            rows["stickiness"],
            width,
            bottom=rows["escalation"],
            color="C1",
            hatch=hatch,
            edgecolor="white",
            linewidth=0.0,
            label=f"Stickiness, {suffix}",
        )
        ax.errorbar(
            pos, rows["tax"], yerr=yerr(rows, "tax"), fmt="none", ecolor="black", **ERROR_KW
        )
    ax.set_xticks(x, [p.removesuffix("-uniform") for p in POLICIES])
    ax.set_ylabel(cost_label(unit, "Escalation tax per prompt"))
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, f"tax_{unit}")


def fig_frontier(sweeps: pd.DataFrame, baselines: pd.DataFrame, unit: str, out: Path) -> None:
    """Solved rate against net mean cost for router threshold sweeps and fixed policies."""
    sub = sweeps[sweeps["unit"] == unit]
    base = baselines[(baselines["unit"] == unit) & (baselines["detection"] == "oracle")]
    if not usable(sub, f"frontier in {unit}") or not usable(base, f"baselines in {unit}"):
        return
    sub = sub[sub["seed"] == sub["seed"].min()]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.6))
    for router, rows in sub.groupby("router", sort=True):
        name = str(router).removesuffix("-cs")
        if name not in ROUTERS:
            continue
        label, color = ROUTERS[name]
        dashed = str(router).endswith("-cs")
        rows = rows.sort_values("tau")
        ax.plot(
            rows["net_mean_cost"],
            rows["solved_rate"],
            color=color,
            lw=0.9,
            ls="--" if dashed else "-",
            label=label + (", cost-sensitive" if dashed else ""),
        )
    direct = sub[sub["router"].str.startswith("R1-direct")]
    if not direct.empty:
        ax.scatter(
            direct["net_mean_cost"],
            direct["solved_rate"],
            marker="D",
            s=14,
            color="C3",
            label="Direct classifier",
            zorder=3,
        )
    static = base[base["policy"].str.startswith("always-")]
    ax.scatter(
        static["mean_cost"],
        static["solved_rate"],
        marker="o",
        s=10,
        color="0.55",
        label="Static",
        zorder=3,
    )
    manual = base[base["policy"].isin(POLICIES)]
    ax.scatter(
        manual["mean_cost"],
        manual["solved_rate"],
        marker="s",
        s=12,
        color="black",
        label="Manual policies",
        zorder=3,
    )
    oracle = base[base["policy"] == "oracle"]
    ax.scatter(
        oracle["mean_cost"],
        oracle["solved_rate"],
        marker="*",
        s=45,
        color="C4",
        label="Oracle",
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_xlabel(cost_label(unit, "Net mean cost per prompt"))
    ax.set_ylabel("Solved rate")
    ax.margins(y=0.08)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    spots = [(row["mean_cost"], row["solved_rate"]) for _, row in base.iterrows()]
    spots += list(zip(direct["net_mean_cost"], direct["solved_rate"], strict=True))
    boxes = marker_boxes(ax, spots)
    label_points(
        ax,
        [(row["mean_cost"], row["solved_rate"]) for _, row in manual.iterrows()],
        [str(row["policy"]).removesuffix("-uniform") for _, row in manual.iterrows()],
        placed=boxes,
    )
    label_points(
        ax,
        [(row["mean_cost"], row["solved_rate"]) for _, row in static.iterrows()],
        [cell_text(str(row["policy"]).removeprefix("always-")) for _, row in static.iterrows()],
        placed=boxes,
    )
    save(fig, out, f"frontier_{unit}")


def fig_detection(frame: pd.DataFrame, unit: str, out: Path) -> None:
    """Net cost per solved prompt and solved rate as detection errors grow."""
    policies = POLICIES[:-1]
    sub = frame[(frame["unit"] == unit) & frame["policy"].isin(policies)]
    if not usable(sub, f"detection sweep in {unit}"):
        return
    rejects = sorted(sub["p_reject_right"].unique())
    dashes = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.0), sharex=True)
    for c, policy in enumerate(policies):
        for k, p_reject in enumerate(rejects):
            rows = sub[(sub["policy"] == policy) & (sub["p_reject_right"] == p_reject)]
            rows = rows.sort_values("p_accept_wrong")
            kwargs = {"color": f"C{c}", "ls": dashes[k % len(dashes)], "lw": 0.9}
            label = policy.removesuffix("-uniform") if k == 0 else None
            axes[0].plot(rows["p_accept_wrong"], rows["net_cost_per_solved"], label=label, **kwargs)
            axes[1].plot(rows["p_accept_wrong"], rows["solved_rate"], **kwargs)
    axes[0].set_ylabel(cost_label(unit, "Cost per solved"))
    axes[0].legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axes[1].set_ylabel("Solved rate")
    axes[1].set_xlabel(r"$p_{\mathrm{acc}}$, probability of accepting a wrong answer")
    handles = [
        Line2D(
            [],
            [],
            color="black",
            lw=0.9,
            ls=dashes[k % len(dashes)],
            label=rf"$p_{{\mathrm{{rej}}}} = {p:g}$",
        )
        for k, p in enumerate(rejects)
    ]
    axes[1].legend(handles=handles, frameon=False, ncol=len(handles))
    save(fig, out, f"detection_{unit}")


def fig_mode(frame: pd.DataFrame, unit: str, out: Path) -> None:
    """Cost of turning thinking on, against the cost of moving up one size."""
    sub = frame[(frame["scope"] == "all") & (frame["unit"] == unit)].set_index("size")
    sizes = [s for s in SIZES if s in sub.index]
    if not sizes:
        LOGGER.warning("no rows for mode signature in %s; skipped", unit)
        return
    rows = sub.loc[sizes]
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.9))
    ax.bar(
        x - 0.2,
        rows["think_over_no_think"],
        0.4,
        color="C0",
        yerr=yerr(rows, "think_over_no_think"),
        error_kw=ERROR_KW,
        label="Thinking vs. same size without",
    )
    ax.bar(
        x + 0.2,
        rows["think_over_size_step"],
        0.4,
        color="C1",
        yerr=yerr(rows, "think_over_size_step"),
        error_kw=ERROR_KW,
        label="Thinking vs. next size without",
    )
    ax.axhline(1.0, color="black", lw=0.6, ls="--")
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(x, [s.upper() for s in sizes])
    ax.set_ylabel("Ratio of cost per solved")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, f"mode_{unit}")


def fig_asymmetry(frame: pd.DataFrame, unit: str, out: Path) -> None:
    """Extra cost of routing one tier too low against one tier too high."""
    sub = frame[frame["unit"] == unit]
    if not usable(sub, f"asymmetry in {unit}"):
        return
    scopes = scopes_in(sub)
    rows = sub.set_index("scope").loc[scopes]
    x = np.arange(len(scopes))
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.9))
    ax.bar(
        x - 0.2,
        rows["too_low_premium"],
        0.4,
        color="C3",
        yerr=yerr(rows, "too_low_premium"),
        error_kw=ERROR_KW,
        label="One tier too low",
    )
    ax.bar(
        x + 0.2,
        rows["too_high_premium"],
        0.4,
        color="C2",
        yerr=yerr(rows, "too_high_premium"),
        error_kw=ERROR_KW,
        label="One tier too high",
    )
    ax.set_xticks(x, [SCOPE_TEXT[s] for s in scopes])
    ax.set_ylabel(cost_label(unit, "Extra cost per prompt"))
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, f"asymmetry_{unit}")


def fig_public(pool: pd.DataFrame, taxes: pd.DataFrame, out: Path) -> None:
    """Accuracy against cost for the public pool, with its frontier and the policies."""
    models = pool[pool["scope"] == "all"]
    policies = taxes[(taxes["scope"] == "all") & (taxes["unit"] == "dollars")].set_index("policy")
    if not usable(models, "public pool") or not usable(policies, "public policies"):
        return
    cost = PER_THOUSAND * models["mean_dollars"].to_numpy()
    accuracy = 100.0 * models["accuracy"].to_numpy()
    ladder = models["on_ladder"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.3))
    ax.scatter(cost[~ladder], accuracy[~ladder], s=12, color="0.65", label="Pool member")
    ax.plot(
        cost[ladder],
        accuracy[ladder],
        marker="o",
        markersize=4,
        color="C0",
        linewidth=0.9,
        label="Escalation ladder",
    )
    rungs = list(zip(models["model"][ladder], cost[ladder], accuracy[ladder], strict=True))
    for index, (name, x, y) in enumerate(rungs):
        last = index == len(rungs) - 1  # the top rung would otherwise run off the axis
        ax.annotate(
            name.replace("-235b-a22b", ""),
            (x, y),
            textcoords="offset points",
            xytext=(-5, 5) if last else (3, -8),
            ha="right" if last else "left",
            va="bottom" if last else "top",
            fontsize=5.5,
            color="C0",
        )
    for policy, marker, color in (("P1", "s", "C1"), ("P3", "v", "C3"), ("oracle", "*", "C2")):
        if policy not in policies.index:
            continue
        row = policies.loc[policy]
        ax.scatter(
            PER_THOUSAND * row["mean_cost"],
            100.0 * row["solved_rate"],
            marker=marker,
            s=26 if marker != "*" else 50,
            color=color,
            zorder=3,
            label="Oracle" if policy == "oracle" else policy,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cost (US\\$ per 1000 queries)")
    ax.set_ylabel("Queries answered (\\%)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handletextpad=0.3, columnspacing=0.8)  # fmt: skip
    save(fig, out, "public_frontier")


HOSTED_SHORT = {"gpt-5-nano": "nano", "gpt-5-mini": "mini", "gpt-5": "gpt-5"}
HOSTED_EFFORT = {"minimal": ("o", "none"), "high": ("^", "C0")}


def fig_hosted(cells: pd.DataFrame, pool: pd.DataFrame, taxes: pd.DataFrame, out: Path) -> None:
    """Accuracy against dollars on the hosted ladder, with the reasoning-effort axis marked.

    One point per hosted cell, filled for high reasoning effort and open for minimal, so the
    mode axis reads as a horizontal displacement at fixed model size. The grey line is the
    cost-ordered frontier the escalation policies climb.
    """
    spec = cells.rename(columns={"model": "family", "cell": "model"})
    rows = pool[pool["scope"] == "all"].merge(
        spec[["model", "family", "effort"]], on="model", how="left"
    )
    policies = taxes[(taxes["scope"] == "all") & (taxes["unit"] == "dollars")].set_index("policy")
    if not usable(rows, "hosted cells") or not usable(policies, "hosted policies"):
        return
    rows = rows.sort_values("mean_dollars")
    cost = PER_THOUSAND * rows["mean_dollars"].to_numpy()
    accuracy = 100.0 * rows["accuracy"].to_numpy()
    ladder = rows["on_ladder"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.3))
    ax.plot(cost[ladder], accuracy[ladder], color="0.6", linewidth=0.9, zorder=1,
            label="Escalation ladder")  # fmt: skip
    for effort, (marker, fill) in HOSTED_EFFORT.items():
        mask = (rows["effort"] == effort).to_numpy()
        if not mask.any():
            continue
        ax.scatter(cost[mask], accuracy[mask], marker=marker, s=26, zorder=5,
                   facecolors=fill, edgecolors="C0", linewidths=0.9,
                   label=f"{effort.capitalize()} effort")  # fmt: skip
    marks = list(zip(cost, accuracy, strict=True))
    for policy, marker, color in (("P1", "s", "C1"), ("P3", "v", "C3"), ("oracle", "*", "C2")):
        if policy not in policies.index:
            continue
        row = policies.loc[policy]
        ax.scatter(PER_THOUSAND * row["mean_cost"], 100.0 * row["solved_rate"], marker=marker,
                   s=50 if marker == "*" else 26, color=color, zorder=3,
                   label="Oracle" if policy == "oracle" else policy)  # fmt: skip
        marks.append((PER_THOUSAND * row["mean_cost"], 100.0 * row["solved_rate"]))
    ax.set_xscale("log")
    ax.set_xlim(cost.min() / 1.9, cost.max() * 1.9)  # room for the end labels
    ax.set_xlabel("Cost (US\\$ per 1000 queries)")
    ax.set_ylabel("Queries answered (\\%)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handletextpad=0.3, columnspacing=0.8)  # fmt: skip
    label_points(
        ax,
        list(zip(cost, accuracy, strict=True)),
        [HOSTED_SHORT.get(family, str(family)) for family in rows["family"]],
        placed=marker_boxes(ax, marks),
        fontsize=5.5,
        color="C0",
        offsets=BELOW_FIRST,
    )
    save(fig, out, "hosted_frontier")


def main() -> None:
    """Draw every figure for which the analysis wrote its input."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "figures")
    parser.add_argument("--units", nargs="+", default=["energy_j", "tokens"])
    parser.add_argument(
        "--public", type=Path, default=None, help="public-benchmark arm results directory"
    )
    parser.add_argument(
        "--hosted", type=Path, default=None, help="hosted-frontier arm results directory"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    style()
    names = (
        "labels_cstar",
        "tax_policies",
        "tax_detection_sweep",
        "mode_signature",
        "asymmetry",
        "routers_sweeps",
        "baselines_test",
    )
    tables = {name: read(args.analysis, name) for name in names}
    for unit in args.units:
        if tables["labels_cstar"] is not None:
            fig_cstar(tables["labels_cstar"], unit, args.out)
        if tables["tax_policies"] is not None:
            fig_tax(tables["tax_policies"], unit, args.out)
        if tables["tax_detection_sweep"] is not None:
            fig_detection(tables["tax_detection_sweep"], unit, args.out)
        if tables["mode_signature"] is not None:
            fig_mode(tables["mode_signature"], unit, args.out)
        if tables["asymmetry"] is not None:
            fig_asymmetry(tables["asymmetry"], unit, args.out)
        if tables["routers_sweeps"] is not None and tables["baselines_test"] is not None:
            fig_frontier(tables["routers_sweeps"], tables["baselines_test"], unit, args.out)
    if args.public is not None:
        pool, taxes = read(args.public, "pool"), read(args.public, "tax_policies")
        if pool is not None and taxes is not None:
            fig_public(pool, taxes, args.out)
    if args.hosted is not None:
        h = {n: read(args.hosted, n) for n in ("cells", "pool", "tax_policies")}
        if all(frame is not None for frame in h.values()):
            fig_hosted(h["cells"], h["pool"], h["tax_policies"], args.out)


if __name__ == "__main__":
    main()
