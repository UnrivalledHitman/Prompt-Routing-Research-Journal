"""Build the oracle-labelling corpus (experimental protocol, section 4). CPU only.

Reads cached Hugging Face datasets. ``HF_HUB_OFFLINE`` and ``HF_DATASETS_OFFLINE``
are set, so a source whose dataset is not already cached fails instead of being
downloaded.

Usage::

    python scripts/build_corpus.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from arbiter.data.corpus import build_corpus, load_corpus_config, write_corpus  # noqa: E402

LOGGER = logging.getLogger("arbiter.build_corpus")


def main() -> None:
    """Select, write and summarise the corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "corpus.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    from transformers import AutoTokenizer

    cfg = load_corpus_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    items, sources = build_corpus(cfg, REPO_ROOT, tokenizer)
    write_corpus(cfg, REPO_ROOT, items, sources)
    by_cell = Counter((item.family, item.subset) for item in items)
    for (family, subset), n in sorted(by_cell.items()):
        LOGGER.info("%-5s %-5s %d", family, subset, n)
    LOGGER.info("%d prompts, %d in the noise subset", len(items), sum(i.noise for i in items))


if __name__ == "__main__":
    main()
