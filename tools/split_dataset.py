"""Copy a class-folder dataset into train/val/test with a fixed random seed."""

import argparse
from pathlib import Path

import splitfolders

from tools.data_utils import validate_output


def split_dataset(source, output, seed=42):
    source, output = validate_output(source, output)
    splitfolders.ratio(
        str(source), output=str(output), seed=seed, ratio=(0.75, 0.15, 0.10), move=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        split_dataset(args.source, args.output, args.seed)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
