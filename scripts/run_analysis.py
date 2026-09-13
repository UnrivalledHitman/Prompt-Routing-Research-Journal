"""Run the study analyses on graded, energy-attributed labels.

Writes one text-free CSV per results block, the split assignment, and ``summary.json``
to the output directory. Everything runs on the CPU except the training of R2 and
R3-LoRA, which waits for an otherwise idle GPU; ``--skip-neural`` leaves out those
routers and R1c.

Usage::

    python scripts/run_analysis.py --grades results/processed/grades_labelling.csv \
        --energy results/processed/energy_labelling.csv --out results/processed/analysis/pilot
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from arbiter.data.corpus import read_corpus  # noqa: E402
from arbiter.energy.telemetry import TelemetryRecorder  # noqa: E402
from arbiter.eval import analysis, probes, robustness  # noqa: E402
from arbiter.eval.labels import (  # noqa: E402
    CELL_NAMES,
    SPLIT_PATTERN,
    LabelTable,
    assign_splits,
    build_label_table,
    minimum_sufficient,
)
from arbiter.eval.policies import ORACLE, Detection  # noqa: E402
from arbiter.labeling.harness import GpuGuard  # noqa: E402
from arbiter.routers import neural  # noqa: E402
from arbiter.routers.classical import (  # noqa: E402
    DirectClassifier,
    FeatureBoostRouter,
    TfidfRouter,
)
from arbiter.routers.llm import LookupRouter  # noqa: E402

LOGGER = logging.getLogger("arbiter.run_analysis")
LLM_ROUTERS = {"route-zero": "R3-qwen3-0.6b-zero-shot", "route-few": "R3-qwen3-0.6b-few-shot"}


def load_table(
    grades: pd.DataFrame, energy: pd.DataFrame | None, prices: dict[str, dict[str, float]]
) -> LabelTable:
    """Build the primary label table, adding token costs under every other price sheet.

    Raises
    ------
    ValueError
        If two price sheets yield different prompt sets.
    """
    primary = build_label_table(grades, energy, **prices["primary"])
    costs = dict(primary.costs)
    for label, sheet in prices.items():
        if label == "primary":
            continue
        other = build_label_table(grades, energy, **sheet)
        if not other.prompts["prompt_id"].equals(primary.prompts["prompt_id"]):
            raise ValueError("price sheets produced different prompt sets")
        costs[f"tokens_{label}"] = other.costs["tokens"]
    return LabelTable(primary.prompts, primary.success, costs, primary.capped, primary.contaminated)


def usable(table: LabelTable, units: list[str]) -> tuple[LabelTable, list[str]]:
    """Keep units measured somewhere, then prompts measured in every kept unit."""
    kept = [unit for unit in units if np.isfinite(table.costs[unit]).any()]
    if set(units) - set(kept):
        LOGGER.warning("no measurements in %s; left out", sorted(set(units) - set(kept)))
    finite = np.all([np.isfinite(table.costs[unit]).all(axis=1) for unit in kept], axis=0)
    if (~finite).any():
        LOGGER.warning("dropping %d prompts with unmeasured cells", int((~finite).sum()))
    return table.take(finite), kept


def router_specs(cfg: dict[str, Any]) -> list[analysis.RouterSpec]:
    """Build the R1 family, its cost-sensitive variants and the direct ablation."""
    c_grid = tuple({"c": c} for c in cfg["tfidf_c_grid"])
    specs = [analysis.RouterSpec("R1a-tfidf-lr", "R1", TfidfRouter, c_grid)]
    for backend in cfg["boost_backends"]:
        make = partial(FeatureBoostRouter, backend=backend)
        specs.append(analysis.RouterSpec(f"R1b-features-{backend}", "R1", make))
    specs += [replace(spec, name=f"{spec.name}-cs", cost_sensitive=True) for spec in list(specs)]
    specs.append(
        analysis.RouterSpec(
            "R1-direct-tfidf-lr", "R1-direct", DirectClassifier, c_grid, direct=True
        )
    )
    return specs


def neural_router_specs(
    rcfg: dict[str, Any], route: pd.DataFrame | None, mean_cost: np.ndarray
) -> tuple[list[analysis.RouterSpec], set[str]]:
    """Build R1c, R2 and their cost-sensitive variants, and R3-LoRA when it can be charged.

    R3-LoRA decides with the prompt and base model of the zero-shot R3 probe, so it is
    charged that probe's metered cost per prompt. This is an upper bound, since the probe
    decodes several tokens where the adapter reads one distribution. Without the probe,
    R3-LoRA is left out. ``mean_cost`` orders its letters as for R3.

    Returns
    -------
    tuple
        The specs, and the names of those that train on the GPU.
    """
    ncfg = rcfg["neural"]
    c_grid = tuple({"c": c} for c in rcfg["tfidf_c_grid"])
    embedding = partial(neural.EmbeddingRouter, **ncfg["embedding"])
    specs = [analysis.RouterSpec("R1c-minilm-lr", "R1", embedding, c_grid)]
    encoder_grid = tuple(dict(point) for point in ncfg["encoder_grid"])
    for label, model in ncfg["encoders"].items():
        name = f"R2-{label}"
        make = partial(neural.EncoderRouter, **model, **ncfg["encoder_training"], name=name)
        specs.append(analysis.RouterSpec(name, "R2", make, encoder_grid))
    specs += [replace(spec, name=f"{spec.name}-cs", cost_sensitive=True) for spec in list(specs)]
    zero = None if route is None else route[route["variant"] == "route-zero"]
    if zero is None or zero.empty:
        LOGGER.info("no zero-shot R3 probe to charge R3-LoRA with; skipped")
    else:
        specs.append(
            analysis.RouterSpec(
                "R3-qwen3-0.6b-lora",
                "R3",
                partial(neural.LoraRouter, mean_cost, **ncfg["lora"]),
                tuple(dict(point) for point in ncfg["lora_grid"]),
                overhead=partial(probes.prompt_costs, zero),
                device="gpu",
            )
        )
    return specs, {spec.name for spec in specs if spec.family != "R1"}


def wait_for_gpu(gcfg: dict[str, Any], reason: str) -> None:
    """Wait, as the labelling harness does, for an otherwise idle GPU with free memory.

    Without CUDA, training runs on the CPU and nothing is waited for.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    recorder = TelemetryRecorder(REPO_ROOT / "results/logs/router_training_telemetry.csv", 0.5)
    recorder.start()
    try:
        guard = GpuGuard(recorder, gcfg)
        guard.wait_until_idle(float(gcfg["idle_required_s"]), reason)
        guard.wait_for_vram(float(gcfg["min_free_vram_mb"]), reason)
    finally:
        recorder.stop()


def load_probes(cfg: dict[str, Any], kind: str, units: list[str]) -> pd.DataFrame | None:
    """Load a probe table with costs in every unit; None if absent or missing a unit."""
    relative = cfg.get("probes", {}).get(kind)
    path = REPO_ROOT / relative if relative else None
    if path is None or not path.exists():
        LOGGER.info("no %s probe table (%s); skipped", kind, relative)
        return None
    frame = probes.unit_costs(pd.read_csv(path), cfg["token_prices"])
    unmeasured = [unit for unit in units if not np.isfinite(frame[unit]).any()]
    if unmeasured:
        LOGGER.warning("%s probes have no measurements in %s; skipped", kind, unmeasured)
        return None
    return frame


def llm_router_specs(
    frame: pd.DataFrame, texts: Mapping[str, str], mean_cost: np.ndarray
) -> list[analysis.RouterSpec]:
    """Build R3 from its measured label distributions, one spec per prompting variant.

    ``mean_cost`` orders the configurations when label mass is accumulated into per-cell
    success, in the label unit on the training split.
    """
    specs = []
    for variant, name in LLM_ROUTERS.items():
        sub = frame[frame["variant"] == variant]
        if sub.empty:
            continue
        parsed = sub[sub["parsed"].astype(bool)]
        LOGGER.info("%s: %d of %d label distributions parsed", name, len(parsed), len(sub))
        labels = parsed[list(probes.LABEL_COLUMNS)].to_numpy(dtype=float)
        distributions = {
            texts[pid]: row
            for pid, row in zip(parsed["prompt_id"], labels, strict=True)
            if pid in texts
        }
        specs.append(
            analysis.RouterSpec(
                name,
                "R3",
                partial(LookupRouter, distributions, mean_cost, name=name),
                overhead=partial(probes.prompt_costs, sub),
                device="gpu",
            )
        )
    return specs


def cascade_thresholds(ccfg: dict[str, Any]) -> list[float]:
    """Return the cascade's threshold grid, with its extra points near one."""
    grid = ccfg["thresholds"]
    values = np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"])
    return sorted({round(float(v), 6) for v in [*values, *ccfg.get("extra_thresholds", [])]})


def git_commit() -> str:
    """Return the current commit hash, or ``unknown``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def write(out: Path, name: str, frame: pd.DataFrame) -> None:
    """Write one results table."""
    path = out / f"{name}.csv"
    frame.to_csv(path, index=False, float_format="%.6g")
    LOGGER.info("wrote %s (%d rows)", path.name, len(frame))


def assign(table: LabelTable, units: list[str], split_unit: str, seed: int) -> pd.DataFrame:
    """Split prompts 60/20/20 within family and ``c*`` strata; record ``c*`` per unit."""
    cstar = minimum_sufficient(table.success, table.costs[split_unit])
    strata = [f"{f}|{c}" for f, c in zip(table.prompts["family"], cstar, strict=True)]
    frame = table.prompts.copy()
    frame["split"] = assign_splits(frame["prompt_id"].tolist(), strata, seed, SPLIT_PATTERN)
    for unit in units:
        choice = minimum_sufficient(table.success, table.costs[unit])
        frame[f"cstar_{unit}"] = [CELL_NAMES[c] if c >= 0 else "unservable" for c in choice]
    return frame


def label_analyses(
    out: Path, table: LabelTable, units: list[str], cfg: dict[str, Any], n_boot: int
) -> None:
    """Write corpus, tax, mode-axis and asymmetry tables."""
    seed, level = cfg["seed"], cfg["bootstrap"]["level"]
    detections = {name: Detection(**cfg["detection"][name]) for name in ("oracle", "illustrative")}
    p4 = cfg["p4_start_weights"]
    grid = cfg["detection"]["sweep"]
    write(out, "labels_cells", analysis.label_summary(table, units))
    write(out, "labels_cstar", analysis.cstar_distribution(table, units))
    started = time.time()
    policies = analysis.policy_table(table, units, detections, p4, n_boot, seed, level)
    write(out, "tax_policies", policies)
    LOGGER.info("policy table in %.0f s", time.time() - started)
    write(out, "carbon", analysis.carbon_table(policies, cfg["carbon"]["intensities"]))
    sweeps = [
        analysis.detection_sweep(table, unit, grid["p_accept_wrong"], grid["p_reject_right"], p4)
        for unit in units
    ]
    write(out, "tax_detection_sweep", pd.concat(sweeps, ignore_index=True))
    signature = [analysis.mode_signature(table, unit, n_boot, seed, level) for unit in units]
    write(out, "mode_signature", pd.concat(signature, ignore_index=True))
    asymmetry = [analysis.asymmetry(table, unit, n_boot, seed, level) for unit in units]
    write(out, "asymmetry", pd.concat(asymmetry, ignore_index=True))


def robustness_analyses(
    out: Path, grades: pd.DataFrame, table: LabelTable, cfg: dict[str, Any]
) -> None:
    """Write label-noise, cap-sensitivity and apportioning checks for the data that exist."""
    prices = cfg["token_prices"]["primary"]
    if {1, 2} <= set(grades["sample"].unique()):
        cells, cstar = robustness.label_noise(grades, (0, 1, 2), "tokens")
        write(out, "label_noise_cells", cells)
        write(out, "label_noise_cstar", cstar)
    else:
        LOGGER.info("no repeated samples; label noise skipped")
    variant = "cap8192"  # the cap stage in configs/labelling.yaml
    if (grades["variant"] == variant).any():
        write(
            out, "cap_sensitivity", robustness.cap_sensitivity(grades, variant, "tokens", **prices)
        )
    else:
        LOGGER.info("no %s reruns; cap sensitivity skipped", variant)
    if np.isfinite(table.costs.get("energy_token_j", np.array([np.nan]))).any():
        write(out, "energy_apportioning", robustness.apportioning_check(table))


def router_analyses(
    out: Path,
    table: LabelTable,
    splits: pd.DataFrame,
    units: list[str],
    primary: str,
    cfg: dict[str, Any],
    n_boot: int,
    neural_routers: bool,
) -> None:
    """Evaluate routers and the cascade, and write their results and comparisons."""
    items = {item.prompt_id: item.text for item in read_corpus(REPO_ROOT / cfg["corpus"])}
    parts = {}
    for name in ("train", "val", "test"):
        sub = table.take((splits["split"] == name).to_numpy())
        parts[name] = analysis.Split(sub, [items[p] for p in sub.prompts["prompt_id"]])
    rcfg = cfg["routers"]
    taus = np.round(
        np.arange(rcfg["taus"]["start"], rcfg["taus"]["stop"] + 1e-9, rcfg["taus"]["step"]), 6
    )
    detections = {name: Detection(**cfg["detection"][name]) for name in ("oracle", "illustrative")}
    oracle_label = next(label for label, d in detections.items() if d == ORACLE)
    settings = analysis.RouterSettings(
        units=tuple(units),
        label_unit=primary,
        taus=tuple(float(t) for t in taus),
        seeds=tuple(rcfg["seeds"]),
        detections=detections,
        recoveries=(cfg["recovery"]["primary"], cfg["recovery"]["sensitivity"]),
        objective=rcfg["selection_objective"],
        cpu_power_w=rcfg["cpu_power_w"],
    )
    test = parts["test"].table
    resamples = analysis.Resamples.draw(
        len(test), n_boot, cfg["seed"] + 1, cfg["bootstrap"]["level"]
    )
    baselines, base_eval = analysis.baseline_evaluations(
        test, units, detections, cfg["p4_start_weights"], resamples
    )
    write(out, "baselines_test", baselines)

    specs = router_specs(rcfg)
    route = load_probes(cfg, "route", units)
    label_costs = analysis.mean_cell_cost(parts["train"].table.costs[primary])
    if route is not None:
        specs += llm_router_specs(route, items, label_costs)
    gpu_trained: set[str] = set()
    if neural_routers:
        extra, gpu_trained = neural_router_specs(rcfg, route, label_costs)
        specs += extra
    runs: dict[str, analysis.RouterRun] = {}
    families: dict[str, str] = {}
    for spec in specs:
        if spec.name in gpu_trained:
            wait_for_gpu(rcfg["neural"]["gpu_guard"], spec.name)
        started = time.time()
        runs[spec.name] = analysis.run_router(
            spec, parts["train"], parts["val"], parts["test"], settings, resamples
        )
        families[spec.name] = spec.family
        LOGGER.info("%s in %.0f s", spec.name, time.time() - started)
    verify = load_probes(cfg, "verify", units)
    if verify is not None:
        checks = {
            name: probes.verification(verify, parts[name].table, units) for name in ("val", "test")
        }
        runs[analysis.CASCADE] = analysis.run_cascade(
            parts["val"],
            parts["test"],
            checks["val"],
            checks["test"],
            settings,
            cascade_thresholds(cfg["cascade"]),
            resamples,
        )
        families[analysis.CASCADE] = "cascade"
    write(
        out, "routers_selection", pd.DataFrame([r for run in runs.values() for r in run.selection])
    )
    write(out, "routers_test", pd.DataFrame([r for run in runs.values() for r in run.test]))
    write(out, "routers_overhead", pd.DataFrame([r for run in runs.values() for r in run.overhead]))
    sweeps = pd.concat([f for run in runs.values() for f in run.sweeps], ignore_index=True)
    write(out, "routers_sweeps", sweeps)
    write(out, "frontier_area", analysis.frontier_table(sweeps, baselines))

    frames = []
    per_cell = [name for name in runs if families[name] == "R1"]
    direct = [name for name in runs if families[name] == "R1-direct"]
    one_shot = [name for name in runs if families[name] in ("R1", "R2", "R3")]
    for unit in units:
        for recovery in settings.recoveries:
            for label in detections:
                candidates = {name: runs[name].evaluated[(unit, recovery, label)] for name in runs}
                group = {
                    name: base_eval[(unit, label, name)]
                    for name in analysis.COMPARED_BASELINES
                    if (unit, label, name) in base_eval
                }
                oracle = {"oracle": base_eval[(unit, oracle_label, "oracle")]}
                blocks = {
                    "routers_vs_baselines": analysis.compare(candidates, group, resamples),
                    "oracle_vs_baselines": analysis.compare(oracle, group, resamples),
                    "per_cell_vs_direct": analysis.compare(
                        {name: candidates[name] for name in per_cell},
                        {name: candidates[name] for name in direct},
                        resamples,
                    ),
                }
                if analysis.CASCADE in runs:
                    blocks["routers_vs_cascade"] = analysis.compare(
                        {name: candidates[name] for name in one_shot},
                        {analysis.CASCADE: candidates[analysis.CASCADE]},
                        resamples,
                    )
                for block, frame in blocks.items():
                    frames.append(
                        frame.assign(block=block, unit=unit, recovery=recovery, detection=label)
                    )
    write(out, "comparisons", pd.concat(frames, ignore_index=True))


def main() -> None:
    """Load labels, run every analysis and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--energy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/analysis.yaml")
    parser.add_argument("--n-boot", type=int, help="override the configured resample count")
    parser.add_argument("--skip-routers", action="store_true")
    parser.add_argument("--skip-neural", action="store_true", help="leave out R1c, R2 and R3-LoRA")
    args = parser.parse_args()

    log_path = REPO_ROOT / "results/logs" / f"analysis_{args.out.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    started = time.time()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    n_boot = args.n_boot or cfg["bootstrap"]["n_boot"]
    args.out.mkdir(parents=True, exist_ok=True)

    grades = pd.read_csv(args.grades)
    energy = pd.read_csv(args.energy) if args.energy else None
    table = load_table(grades, energy, cfg["token_prices"])
    extra = [f"tokens_{label}" for label in cfg["token_prices"] if label != "primary"]
    table, units = usable(table, [*cfg["units"], *extra])
    primary = cfg["primary_unit"] if cfg["primary_unit"] in units else "tokens"
    router_units = [unit for unit in cfg["units"] if unit in units]
    LOGGER.info("%d prompts; units %s; primary %s", len(table), units, primary)

    splits = assign(table, units, cfg["split_unit"], cfg["seed"])
    write(args.out, "splits", splits)
    label_analyses(args.out, table, units, cfg, n_boot)
    robustness_analyses(args.out, grades, table, cfg)
    neural_routers = cfg["routers"]["neural"]["enabled"] and not args.skip_neural
    if not args.skip_routers:
        router_analyses(args.out, table, splits, router_units, primary, cfg, n_boot, neural_routers)

    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "grades": str(args.grades),
        "energy": str(args.energy) if args.energy else None,
        "n_boot": n_boot,
        "n_prompts": len(table),
        "prompts_by_family_and_split": {
            family: group["split"].value_counts().to_dict()
            for family, group in splits.groupby("family")
        },
        "units": units,
        "primary_unit": primary,
        "unservable_share": float((~table.success.any(axis=1)).mean()),
        "capped_share": float(table.capped.mean()),
        "contaminated_share": float(table.contaminated.mean()),
        "routers": not args.skip_routers,
        "neural_routers": neural_routers and not args.skip_routers,
        "probes": {
            kind: (REPO_ROOT / path).exists() for kind, path in cfg.get("probes", {}).items()
        },
        "seconds": round(time.time() - started, 1),
        "config": cfg,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("done in %.0f s -> %s", time.time() - started, args.out)


if __name__ == "__main__":
    main()
