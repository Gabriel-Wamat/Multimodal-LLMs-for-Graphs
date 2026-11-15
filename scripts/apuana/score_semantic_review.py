"""Validate completed judgments and summarize each run on the same reviewed questions."""
import argparse
from pathlib import Path

from src.evaluation.semantic_review import finalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_folder", type=Path)
    parser.add_argument("reviewed_csv", type=Path)
    args = parser.parse_args()
    finalize(args.review_folder, args.reviewed_csv)


if __name__ == "__main__":
    main()
