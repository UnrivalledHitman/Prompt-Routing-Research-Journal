"""Run one oracle-labelling stage on the local Ollama pool. Uses the GPU.

Stages are defined in ``configs/labelling.yaml``. A stage resumes from its recorded
generations if interrupted, and holds whenever another workload is using the GPU.

Usage::

    python scripts/run_labelling.py --stage smoke
    python scripts/run_labelling.py --stage pilot
    python scripts/run_labelling.py --verify-models      # metadata only, no GPU
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from arbiter.labeling.harness import load_config, run_stage, verify_models  # noqa: E402

LOGGER = logging.getLogger("arbiter.run_labelling")


def keep_system_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs.

    The request belongs to this process and ends when it exits; no power setting is
    changed. A closed lid or a manual sleep still suspends the machine.
    """
    if sys.platform != "win32":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> None:
    """Parse arguments and run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "labelling.yaml")
    parser.add_argument("--stage", help="stage name from the config")
    parser.add_argument("--verify-models", action="store_true", help="metadata only; no GPU")
    args = parser.parse_args()
    if not args.verify_models and not args.stage:
        parser.error("--stage is required unless --verify-models is given")

    log_dir = REPO_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = "labelling_verify.log" if args.verify_models else f"labelling_{args.stage}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / log_name)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.verify_models:
        import ollama

        client = ollama.Client(host=cfg["ollama_host"], timeout=cfg["request_timeout_s"])
        manifest, problems = verify_models(cfg, client)
        LOGGER.info("%s", json.dumps(manifest, indent=2, default=str))
        if problems:
            raise SystemExit("model verification failed:\n  " + "\n  ".join(problems))
        LOGGER.info("all %d models verified", len(manifest))
        return
    keep_system_awake()
    run_stage(cfg, args.stage, REPO_ROOT, args.config)


if __name__ == "__main__":
    main()
