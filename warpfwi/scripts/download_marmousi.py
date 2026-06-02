"""Stub: fetch the Marmousi II velocity model to ``data/marmousi/``.

This script is a placeholder. The Marmousi II P-wave velocity is hosted on
the AGL (University of Houston) and other mirrors whose URLs change over
time. Fill in the URL and checksum below once you have a stable mirror.

Once a .npy file exists at ``data/marmousi/vp.npy``, set
``DataConfig.marmousi_path='data/marmousi/vp.npy'`` in the notebook to use it
instead of the synthetic fallback.
"""
from __future__ import annotations

import os
import sys


MARMOUSI_URL = ""  # TODO: fill in a stable mirror URL
DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "marmousi")


def main() -> int:
    if not MARMOUSI_URL:
        print(
            "download_marmousi.py is a stub.\n"
            "Fill in MARMOUSI_URL with a stable mirror, then re-run.\n"
            f"Destination directory: {DEST_DIR}",
            file=sys.stderr,
        )
        return 2
    os.makedirs(DEST_DIR, exist_ok=True)
    # TODO: implement fetch + checksum verify.
    return 0


if __name__ == "__main__":
    sys.exit(main())
