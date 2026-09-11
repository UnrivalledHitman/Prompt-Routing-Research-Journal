"""Score open-ended labelling answers with the local judge. Uses the GPU.

Run it after labelling, never alongside it: the judge will not judge while a routed
model is loaded, and holds whenever another workload is using the GPU. It resumes from
its recorded judgements if interrupted.

Usage::

    python scripts/run_judge.py results/raw/labelling
    python scripts/run_judge.py results/raw/labelling --dry-run    # counts only, no GPU
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from arbiter.eval.judge import load_config, pending, run_judge  # noqa: E402

LOGGER = logging.getLogger("arbiter.run_judge")


def keep_system_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs, as ``run_labelling.py`` does."""
    if sys.platform != "win32":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> None:
    """Parse arguments, then judge or report what is left to judge."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path, help="labelling output directory")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "judge.yaml")
    parser.add_argument("--dry-run", action="store_true", help="count answers only; no GPU")
    args = parser.parse_args()

    log_dir = REPO_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"judge_{args.raw_dir.name}.log"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.dry_run:
        pending(cfg, REPO_ROOT, args.raw_dir)
        return
    keep_system_awake()
    run_judge(cfg, REPO_ROOT, args.config, args.raw_dir)


if __name__ == "__main__":
    main()
