# Inference script for multi-person UWB radar localization.
#
# Input:  .npy file of shape (T, 6, 3, 120, 2)
#         T frames × 6 radars × 3 antennas × 120 range bins × [I, Q]
# Output: .jsonl file, one record per frame: {"frame": t, "localizations": [[x, y], ...]}
#
# Usage:
#   python submission/code.py --input-path <input.npy> --output-path <output.jsonl>

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference script for multi-person UWB radar localization"
    )
    p.add_argument(
        "--input-path", required=True,
        help="Path to input .npy file of shape (T, 6, 3, 120, 2)"
    )
    p.add_argument(
        "--output-path", required=True,
        help="Path to write output .jsonl file"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    raw = np.load(input_path)
    T = raw.shape[0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_f:
        for t in range(T):
            # TODO: process frame raw[t] and produce localizations
            localizations = []
            record = {"frame": t, "localizations": localizations}
            out_f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
