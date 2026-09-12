r"""Write the results tables as LaTeX fragments from an analysis directory. CPU only.

Each table is written to ``<out>/<name>.tex`` as a booktabs ``tabular`` without a float
or caption, so the paper places it with ``\input`` inside its own table environment.
A table whose input is missing is skipped with a warning.

Usage::

    python scripts/make_tables.py results/processed/analysis/main --out paper/tables
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("arbiter.make_tables")
REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"all": "All", "math": "Math", "code": "Code", "mcq": "MCQ", "open": "Open"}
POLICIES = {
    "P1": "P1, frugal climb",
    "P2": "P2, think first",
    "P3": "P3, sticky high",
    "P4-uniform": "P4, uniform start",
    "P4-skewed_c8": "P4, start skewed to $c_8$",
    "random": "Random, then $c_8$",
    "oracle": "Oracle",
}
ROUTERS = {
    "R1a-tfidf-lr": "TF-IDF + LR",
    "R1b-features-lightgbm": "Features + LightGBM",
    "R1b-features-xgboost": "Features + XGBoost",
    "R1-direct-tfidf-lr": "Direct: TF-IDF + LR",
    "R3-qwen3-0.6b-zero-shot": "Qwen3-0.6B, zero-shot",
    "R3-qwen3-0.6b-few-shot": "Qwen3-0.6B, few-shot",
    "cascade-automix": "AutoMix-style cascade",
}
PER_CHECK = {"cascade-automix"}  # overhead is reported per verification call
DIGITS = {"energy_j": 1, "board_j": 1, "tokens": 0, "tokens_output_heavy": 0, "latency_s": 2}
PUBLIC_POLICIES = {
    "P1": "P1, climb the frontier",
    "P3": "P3, sticky top",
    "P4-uniform": "P4, uniform start",
    "random": "Random, then top",
    "oracle": "Oracle",
}
# The hosted pool has a mode axis the public one lacks, so it also reports the
# costly-mode-first policy. Its P1 stays the frontier climb the public arm reports.
HOSTED_POLICIES = {
    "P1": PUBLIC_POLICIES["P1"],
    "P2": "P2, costly mode first",
    **{name: label for name, label in PUBLIC_POLICIES.items() if name != "P1"},
}
HOSTED_EFFORTS = {"minimal": "Minimal", "high": "High"}
PER_THOUSAND = 1000.0  # published costs are dollars per query; the tables report per 1,000


def finite(value: Any) -> bool:  # noqa: ANN401
    """Return whether a value is a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def num(value: Any, digits: int) -> str:  # noqa: ANN401
    """Format a number with LaTeX digit grouping, or ``--`` when missing."""
    if not finite(value):
        return "--"
    return f"{float(value):,.{digits}f}".replace(",", "{,}")


def pct(value: Any, digits: int = 1) -> str:  # noqa: ANN401
    """Format a share as a percentage."""
    return num(100.0 * float(value), digits) if finite(value) else "--"


def with_ci(row: pd.Series, name: str, digits: int, scale: float = 1.0) -> str:
    """Return ``value [lo, hi]`` from ``name``, ``name_lo`` and ``name_hi``."""
    parts = [row.get(name), row.get(f"{name}_lo"), row.get(f"{name}_hi")]
    point, lo, hi = (num(scale * float(v), digits) if finite(v) else "--" for v in parts)
    return f"{point} [{lo}, {hi}]"


def pvalue(value: Any) -> str:  # noqa: ANN401
    """Format a p-value, flooring the display at 0.001."""
    if not finite(value):
        return "--"
    return "$<$0.001" if float(value) < 0.001 else f"{float(value):.3f}"


def router_text(name: str) -> str:
    """Return the display name of a router, marking cost-sensitive variants."""
    base = name.removesuffix("-cs")
    text = ROUTERS.get(base, base)
    return f"{text} (cost-sens.)" if name.endswith("-cs") else text


def tabular(
    columns: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    breaks: Sequence[int] = (),
    colsep: float | None = None,
) -> str:
    """Return a booktabs tabular, with a midrule before each row index in ``breaks``.

    ``colsep`` narrows the space between columns, in points, for tables that would overrun.
    """
    lines = [] if colsep is None else [f"\\setlength{{\\tabcolsep}}{{{colsep:g}pt}}"]
    lines += [f"\\begin{{tabular}}{{{columns}}}", "\\toprule", " & ".join(header) + r" \\"]
    lines.append("\\midrule")
    for index, row in enumerate(rows):
        if index in breaks and index > 0:
            lines.append("\\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def read(analysis: Path, name: str) -> pd.DataFrame | None:
    """Return one analysis table, or None with a warning if it was not written."""
    path = analysis / f"{name}.csv"
    if not path.exists():
        LOGGER.warning("%s not found; dependent tables skipped", path.name)
        return None
    return pd.read_csv(path)


def write(out: Path, name: str, text: str | None) -> None:
    """Write one table fragment, unless there was nothing to write."""
    if text is None:
        LOGGER.warning("no rows for %s; skipped", name)
        return
    (out / f"{name}.tex").write_text(text, encoding="utf-8")
    LOGGER.info("wrote %s.tex", name)


def table_cells(frame: pd.DataFrame) -> str | None:
    """Accuracy, cap rate and mean cost of every cell over all prompts."""
    rows = frame[frame["scope"] == "all"]
    if rows.empty:
        return None
    header = ["Cell", "Size", "Mode", r"Acc.\ (\%)", r"Capped (\%)", "Tokens", "Latency (s)",
              "Energy (J)", "J per solved"]  # fmt: skip
    body = [
        [
            f"$c_{row['cell'][1:]}$",
            str(row["size"]).upper(),
            "think" if row["mode"] == "think" else "no-think",
            pct(row["accuracy"]),
            pct(row["cap_rate"]),
            num(row.get("mean_tokens"), 0),
            num(row.get("mean_latency_s"), 2),
            num(row.get("mean_energy_j"), 1),
            num(row.get("energy_j_per_solved"), 1),
        ]
        for _, row in rows.iterrows()
    ]
    return tabular("@{}lllrrrrrr@{}", header, body)


def table_family_accuracy(frame: pd.DataFrame) -> str | None:
    """Accuracy of every cell in each task family."""
    scopes = [s for s in FAMILIES if s in set(frame["scope"])]
    if not scopes:
        return None
    wide = frame.pivot_table(index="cell", columns="scope", values="accuracy").reindex(
        columns=scopes
    )
    body = [[f"$c_{cell[1:]}$", *(pct(wide.loc[cell, s]) for s in scopes)] for cell in wide.index]
    header = ["Cell", *(FAMILIES[s] for s in scopes)]
    return tabular("@{}l" + "r" * len(scopes) + "@{}", header, body)


def table_tax(frame: pd.DataFrame, unit: str) -> str | None:
    """Cost, success and escalation tax of the manual policies under both detection regimes."""
    rows = frame[(frame["scope"] == "all") & (frame["unit"] == unit)]
    if rows.empty:
        return None
    d = DIGITS.get(unit, 1)
    body, breaks = [], []
    for detection in ("oracle", "illustrative"):
        block = rows[rows["detection"] == detection].set_index("policy")
        breaks.append(len(body))
        for policy in POLICIES:
            if policy not in block.index:
                continue
            row = block.loc[policy]
            body.append(
                [
                    POLICIES[policy],
                    "oracle" if detection == "oracle" else "imperfect",
                    pct(row["solved_rate"]),
                    with_ci(row, "cost_per_solved", d),
                    with_ci(row, "tax", d),
                    num(row["escalation"], d),
                    num(row["stickiness"], d),
                    num(row["relative_tax"], 2),
                ]
            )
    header = ["Policy", "Detection", r"Solved (\%)", "Cost per solved [95\\% CI]",
              "Tax [95\\% CI]", "Esc.", "Stk.", r"Tax / $c^*$ cost"]  # fmt: skip
    return tabular("@{}llrrrrrr@{}", header, body, breaks)


def table_routers(
    tests: pd.DataFrame,
    baselines: pd.DataFrame,
    comparisons: pd.DataFrame,
    areas: pd.DataFrame,
    unit: str,
    recovery: str = "top",
    detection: str = "oracle",
) -> str | None:
    """Routers on the test split against fixed policies, net of router cost."""
    d = DIGITS.get(unit, 1)
    base = baselines[(baselines["unit"] == unit) & (baselines["detection"] == detection)]
    runs = tests[
        (tests["unit"] == unit)
        & (tests["recovery"] == recovery)
        & (tests["detection"] == detection)
    ]
    if base.empty or runs.empty:
        return None
    body = []
    for policy in ("always-c8", "P1", "random", "oracle"):
        match = base[base["policy"] == policy]
        if match.empty:
            continue
        row = match.iloc[0]
        name = "Always $c_8$" if policy == "always-c8" else POLICIES[policy]
        body.append([name, "--", pct(row["solved_rate"]), with_ci(row, "net_mean_cost", d),
                     "--", "--", "--", "--"])  # fmt: skip
    breaks = [len(body)]
    first_seed = runs["seed"].min()
    compared = comparisons[
        (comparisons["block"] == "routers_vs_baselines")
        & (comparisons["unit"] == unit)
        & (comparisons["recovery"] == recovery)
        & (comparisons["detection"] == detection)
        & (comparisons["baseline"] == "P1")
    ].set_index("candidate")
    in_unit = areas[areas["unit"] == unit]
    for router, group in runs.groupby("router", sort=True):
        row = group[group["seed"] == first_seed].iloc[0]
        spread = group["net_mean_cost"].std(ddof=1) if len(group) > 1 else float("nan")
        vs = compared.loc[router] if router in compared.index else pd.Series(dtype=float)
        area = in_unit.loc[in_unit["curve"] == router, "area"].mean()
        body.append(
            [
                router_text(str(router)),
                num(row["tau"], 2),
                pct(row["solved_rate"]),
                f"{with_ci(row, 'net_mean_cost', d)} ({num(spread, d)})",
                with_ci(vs, "saving", d) if not vs.empty else "--",
                with_ci(vs, "solved_difference", 1, 100.0) if not vs.empty else "--",
                pvalue(vs.get("saving_p_holm")) if not vs.empty else "--",
                num(area, 3),
            ]
        )
    header = ["Policy or router", r"$\tau$", r"Solved (\%)", "Net cost [95\\% CI] (s.d.)",
              "Saving vs.\\ P1 [95\\% CI]", "$\\Delta$ solved (pp)", "$p_{\\mathrm{Holm}}$",
              "Area"]  # fmt: skip
    return tabular("@{}lrrrrrrr@{}", header, body, breaks, colsep=4)


def table_overhead(frame: pd.DataFrame) -> str | None:
    """Decision latency and energy per router, averaged over seeds and both training objectives."""
    if frame.empty:
        return None
    frame = frame.assign(router=frame["router"].str.removesuffix("-cs"))
    means = frame.groupby(["router", "device"], sort=True)[
        ["p50_s", "p95_s", "joules_per_decision"]
    ].mean()
    body = [
        [
            router_text(str(router)) + (", per check" if router in PER_CHECK else ""),
            str(device).upper(),
            num(1000 * row["p50_s"], 2),
            num(1000 * row["p95_s"], 2),
            num(row["joules_per_decision"], 3),
        ]
        for (router, device), row in means.iterrows()
    ]
    header = ["Router", "Device", "p50 (ms)", "p95 (ms)", "J/decision"]
    return tabular("@{}llrrr@{}", header, body, colsep=4)


def table_robustness(
    noise: pd.DataFrame | None, cap: pd.DataFrame | None, apportion: pd.DataFrame | None
) -> str | None:
    """One-line summaries of the label robustness checks, over all prompts."""
    body = []
    if noise is not None and not noise[noise["scope"] == "all"].empty:
        row = noise[noise["scope"] == "all"].iloc[0]
        body += [
            ["Cell labels differing between two samples", pct(row["cell_pairwise_disagreement"])],
            ["$c^*$ equal between two samples", pct(row["cstar_pairwise_agreement"])],
            [
                "$c^*$ equal, first sample vs.\\ majority of three",
                pct(row["cstar_first_vs_majority"]),
            ],
        ]
    if cap is not None:
        rows = cap[(cap["family"] == "all") & (cap["size"] == "all")]
        if not rows.empty:
            row = rows.iloc[0]
            body += [
                ["Generations hitting the cap", pct(row["cap_rate"])],
                ["Capped reruns succeeding at 8{,}192 tokens", pct(row["rescued_rate"])],
                ["Capped reruns that move $c^*$", pct(row["cstar_moved_rate"])],
            ]
    if apportion is not None and not apportion[apportion["scope"] == "all"].empty:
        row = apportion[apportion["scope"] == "all"].iloc[0]
        body += [
            ["$c^*$ equal, wall-time vs.\\ token-share energy", pct(row["cstar_agreement"])],
            ["Rank correlation of per-generation energy", num(row["spearman"], 3)],
        ]
    if not body:
        return None
    return tabular("@{}lr@{}", ["Check", "Value"], body)


def table_public_pool(pool: pd.DataFrame, cstar: pd.DataFrame) -> str | None:
    """Accuracy, cost, $c^*$ share and ladder membership of every model in the public pool."""
    rows = pool[pool["scope"] == "all"].sort_values("rung")
    shares = cstar[(cstar["scope"] == "all") & (cstar["unit"] == "dollars")]
    if rows.empty or shares.empty:
        return None
    share = dict(zip(shares["cstar"], shares["share"], strict=True))
    body = [
        [
            rf"\texttt{{{row['model']}}}",
            pct(row["accuracy"]),
            num(PER_THOUSAND * row["mean_dollars"], 2),
            num(PER_THOUSAND * row["dollars_per_solved"], 2),
            pct(share.get(row["model"], float("nan"))),
            r"$\bullet$" if row["on_ladder"] else "",
        ]
        for _, row in rows.iterrows()
    ]
    body.append(["No model answers", "--", "--", "--", pct(share.get("unservable")), ""])
    header = ["Model", r"Acc.\ (\%)", r"Cost (\$/1{,}000)", r"\$ per solved (/1{,}000)",
              r"$c^*$ share (\%)", "Ladder"]  # fmt: skip
    return tabular("@{}lrrrrc@{}", header, body, breaks=[len(body) - 1], colsep=4)


def table_public_tax(
    frame: pd.DataFrame,
    static: Callable[[str], str] | None = None,
    labels: Mapping[str, str] | None = None,
) -> str | None:
    """Escalation tax on a priced pool, in dollars per 1,000 queries.

    ``labels`` names and orders the policies; it defaults to the public arm's set.
    """
    labels = labels or PUBLIC_POLICIES
    rows = frame[(frame["scope"] == "all") & (frame["unit"] == "dollars")].set_index("policy")
    if rows.empty:
        return None
    names = [p for p in labels if p != "oracle" and p in rows.index]
    statics = [p for p in rows.index if str(p).startswith("always-")]
    body, breaks = [], [len(names), len(names) + len(statics)]
    for policy in [*names, *statics, "oracle"]:
        if policy not in rows.index:
            continue
        row = rows.loc[policy]
        name = str(policy).removeprefix("always-")
        fallback = static(name) if static else rf"Always \texttt{{{name}}}"
        label = labels.get(policy, fallback)
        body.append(
            [
                label,
                pct(row["solved_rate"]),
                num(row["attempts"], 2),
                with_ci(row, "mean_cost", 2, PER_THOUSAND),
                with_ci(row, "tax", 2, PER_THOUSAND),
                num(row["relative_tax"], 2),
            ]
        )
    header = ["Policy", r"Solved (\%)", "Attempts", r"Cost (\$/1{,}000) [95\% CI]",
              r"Tax [95\% CI]", r"Tax / $c^*$ cost"]  # fmt: skip
    return tabular("@{}lrrrrr@{}", header, body, breaks, colsep=4)


def table_hosted_cells(cells: pd.DataFrame) -> str | None:
    """Accuracy, output and reasoning tokens, cap-hit rate and price of every hosted cell.

    Two accuracies: the headline one, in which a generation cut off at the cap is wrong, and
    the same figure over the generations that finished on their own.
    """
    if cells.empty:
        return None
    body = [
        [
            f"$h_{row['cell'][1:]}$",
            rf"\texttt{{{row['model']}}}",
            HOSTED_EFFORTS.get(row["effort"], str(row["effort"])),
            num(row["cap"], 0),
            pct(row["accuracy"]),
            pct(row["accuracy_finished"]),
            num(row["mean_completion_tokens"], 0),
            num(row["mean_reasoning_tokens"], 0),
            pct(row["cap_hit_rate"]),
            num(PER_THOUSAND * row["mean_cost_usd"], 2),
        ]
        for _, row in cells.iterrows()
    ]
    header = ["Cell", "Model", "Effort", "Cap", r"Acc.\ (\%)", r"Acc.\ fin.\ (\%)",
              "Out.\\ tok.", "Reas.\\ tok.", r"Cap hit (\%)",
              r"Cost (\$/1{,}000)"]  # fmt: skip
    return tabular("@{}lllrrrrrrr@{}", header, body, colsep=3)


def main() -> None:
    """Write every table for which the analysis wrote its inputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "tables")
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
    names = ("labels_cells", "tax_policies", "routers_test", "baselines_test", "comparisons",
             "frontier_area", "routers_overhead", "label_noise_cstar", "cap_sensitivity",
             "energy_apportioning")  # fmt: skip
    t = {name: read(args.analysis, name) for name in names}
    if t["labels_cells"] is not None:
        write(args.out, "cells", table_cells(t["labels_cells"]))
        write(args.out, "family_accuracy", table_family_accuracy(t["labels_cells"]))
    for unit in args.units:
        if t["tax_policies"] is not None:
            write(args.out, f"tax_{unit}", table_tax(t["tax_policies"], unit))
        router_inputs = [
            t[n] for n in ("routers_test", "baselines_test", "comparisons", "frontier_area")
        ]
        if all(frame is not None for frame in router_inputs):
            write(args.out, f"routers_{unit}", table_routers(*router_inputs, unit))
    if t["routers_overhead"] is not None:
        write(args.out, "overhead", table_overhead(t["routers_overhead"]))
    write(
        args.out,
        "robustness",
        table_robustness(t["label_noise_cstar"], t["cap_sensitivity"], t["energy_apportioning"]),
    )
    if args.public is not None:
        p = {name: read(args.public, name) for name in ("pool", "cstar", "tax_policies")}
        if p["pool"] is not None and p["cstar"] is not None:
            write(args.out, "public_pool", table_public_pool(p["pool"], p["cstar"]))
        if p["tax_policies"] is not None:
            write(args.out, "public_tax", table_public_tax(p["tax_policies"]))
    if args.hosted is not None:
        h = {n: read(args.hosted, n) for n in ("cells", "tax_policies")}
        if h["cells"] is not None:
            write(args.out, "hosted_cells", table_hosted_cells(h["cells"]))
        if h["tax_policies"] is not None:
            write(
                args.out,
                "hosted_tax",
                table_public_tax(
                    h["tax_policies"],
                    lambda name: f"Always $h_{{{name[1:]}}}$",
                    HOSTED_POLICIES,
                ),
            )


if __name__ == "__main__":
    main()
