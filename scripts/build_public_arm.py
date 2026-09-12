"""Compact the LLMRouterBench release into one row per model and query. CPU only.

The release is a tar.gz of JSON files, one per dataset and model, each holding the full
prompts and generations. This script streams it from the local Hugging Face cache, keeps
the outcome fields of the configured pool, and writes a text-free table plus a manifest
recording the archive digest and the subset chosen for each dataset.

Usage::

    python scripts/build_public_arm.py --config configs/public_arm.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tarfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

LOGGER = logging.getLogger("arbiter.build_public_arm")
PEEK_BYTES = 4096
FIELDS = ("index", "score", "prompt_tokens", "completion_tokens", "cost")


def low_priority() -> None:
    """Ask Windows to schedule this process below normal, so GPU runs keep their CPU."""
    if sys.platform != "win32":
        return
    import ctypes

    below_normal = 0x4000
    kernel32 = ctypes.windll.kernel32
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal)


def archive_path(source: dict[str, str]) -> Path:
    """Return the cached archive, without contacting the Hub."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            source["repo_id"],
            source["filename"],
            revision=source["revision"],
            repo_type=source["repo_type"],
            local_files_only=True,
        )
    )


def digest(path: Path) -> str:
    """Return the SHA-256 of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def read_release(path: Path, pool: set[str], datasets: set[str]) -> list[dict[str, Any]]:
    """Return one entry per file of the pool, with its header and its records.

    Files outside the pool are skipped after their first few kilobytes, which is enough
    to read the header fields that identify them.
    """
    files = []
    with tarfile.open(path, "r|gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            head = handle.read(PEEK_BYTES)
            text = head.decode("utf-8", "replace").split('"records"')[0]
            if not any(f'"{name}"' in text for name in pool):
                continue
            if not any(f'"{name}"' in text for name in datasets):
                continue
            data = json.loads((head + handle.read()).decode("utf-8"))
            if data["model_name"] not in pool or data["dataset_name"] not in datasets:
                continue
            files.append(
                {
                    "path": member.name,
                    "model": data["model_name"],
                    "dataset": data["dataset_name"],
                    "split": data.get("split", ""),
                    "fingerprint": data.get("data_fingerprint", ""),
                    "counts": data.get("counts", len(data["records"])),
                    "records": [
                        {field: record.get(field) for field in FIELDS} for record in data["records"]
                    ],
                }
            )
            LOGGER.debug("read %s", member.name)
    return files


def choose_subsets(
    files: list[dict[str, Any]], pool: set[str], datasets: list[str]
) -> dict[str, tuple[str, str]]:
    """Pick, for each dataset, the (split, fingerprint) every pool model was run on.

    Where several qualify, the one with the most queries is taken.

    Raises
    ------
    ValueError
        If a dataset has no subset covering the whole pool.
    """
    covered: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))
    sizes: dict[tuple[str, str, str], int] = {}
    for entry in files:
        key = (entry["split"], entry["fingerprint"])
        covered[entry["dataset"]][key].add(entry["model"])
        sizes[(entry["dataset"], *key)] = max(
            sizes.get((entry["dataset"], *key), 0), len(entry["records"])
        )
    chosen = {}
    for dataset in datasets:
        options = [key for key, models in covered[dataset].items() if models >= pool]
        if not options:
            best = max((len(models) for models in covered[dataset].values()), default=0)
            raise ValueError(
                f"{dataset}: no subset covers all {len(pool)} pool models (best {best})"
            )
        chosen[dataset] = max(options, key=lambda key: sizes[(dataset, *key)])
    return chosen


def build_frame(
    files: list[dict[str, Any]], chosen: dict[str, tuple[str, str]], pool: set[str]
) -> pd.DataFrame:
    """Flatten the chosen files into one row per model and query.

    Raises
    ------
    ValueError
        If a model is missing, duplicated, or answers a different set of queries.
    """
    rows = []
    for entry in files:
        if chosen.get(entry["dataset"]) != (entry["split"], entry["fingerprint"]):
            continue
        for record in entry["records"]:
            rows.append(
                {
                    "dataset": entry["dataset"],
                    "split": entry["split"],
                    "index": record["index"],
                    "model": entry["model"],
                    "score": record["score"],
                    "prompt_tokens": record["prompt_tokens"],
                    "completion_tokens": record["completion_tokens"],
                    "cost": record["cost"],
                }
            )
    frame = pd.DataFrame(rows)
    for dataset, group in frame.groupby("dataset"):
        counts = group.groupby("model").size()
        if set(counts.index) != pool:
            raise ValueError(f"{dataset}: pool mismatch {sorted(set(counts.index) ^ pool)}")
        if counts.nunique() != 1:
            raise ValueError(f"{dataset}: models answer different numbers of queries")
        if group.duplicated(["model", "index"]).any():
            raise ValueError(f"{dataset}: duplicate (model, query) rows")
        per_model = group.groupby("model")["index"].apply(frozenset)
        if per_model.nunique() != 1:
            raise ValueError(f"{dataset}: models answer different queries")
    return frame.sort_values(["dataset", "index", "model"]).reset_index(drop=True)


def main() -> None:
    """Read the release, compact it and write the table and manifest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "public_arm.yaml")
    parser.add_argument("--low-priority", action="store_true", help="yield CPU to a GPU run")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    if args.low_priority:
        low_priority()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pool, datasets = set(cfg["pool"]), list(cfg["datasets"])

    path = archive_path(cfg["source"])
    LOGGER.info("reading %s (%.1f GB)", path.name, path.stat().st_size / 1e9)
    files = read_release(path, pool, set(datasets))
    LOGGER.info("%d files of the pool", len(files))
    chosen = choose_subsets(files, pool, datasets)
    frame = build_frame(files, chosen, pool)

    records = REPO_ROOT / cfg["paths"]["records"]
    records.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(records, index=False, compression="gzip")
    queries = frame.groupby("dataset")["index"].nunique()
    manifest = {
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {**cfg["source"], "sha256": digest(path), "file": path.name},
        "pool": sorted(pool),
        "subsets": {
            dataset: {
                "split": split,
                "fingerprint": fingerprint,
                "queries": int(queries[dataset]),
            }
            for dataset, (split, fingerprint) in chosen.items()
        },
        "rows": int(len(frame)),
        "records": records.relative_to(REPO_ROOT).as_posix(),
    }
    out = REPO_ROOT / cfg["paths"]["manifest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "%d rows, %d queries over %d datasets -> %s\n%s",
        len(frame),
        int(queries.sum()),
        len(queries),
        records,
        queries.to_string(),
    )


if __name__ == "__main__":
    main()
