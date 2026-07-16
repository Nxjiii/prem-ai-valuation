"""Command-line entry point for rebuilding the interim datasets."""

from prem_valuation.data import build_datasets, save_datasets


def main() -> None:
    labelled, scoring = build_datasets()
    labelled_path, scoring_path = save_datasets(labelled, scoring)

    print(f"Labelled rows: {len(labelled):,}")
    print(f"Scoring rows: {len(scoring):,}")
    print(f"Saved: {labelled_path}")
    print(f"Saved: {scoring_path}")


if __name__ == "__main__":
    main()
