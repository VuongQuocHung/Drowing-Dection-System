import argparse

from dataset_utils.LifeguardUtils import DataListToSplitlists


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create leakage-free, stratified folds grouped by source video."
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-fold",
        type=int,
        default=10,
        help="First output fold number; defaults to 10 to preserve old folds 1..9.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = DataListToSplitlists.generate_group_folds(
        n_splits=args.splits,
        seed=args.seed,
        start_fold=args.start_fold,
        overwrite=args.overwrite,
    )
    for row in rows:
        print(
            "Fold {fold}: train={train_snippets} snippets/{train_videos} videos, "
            "test={test_snippets} snippets/{test_videos} videos "
            "(drown={test_drown}, swim={test_swim})".format(**row)
        )


if __name__ == "__main__":
    main()
