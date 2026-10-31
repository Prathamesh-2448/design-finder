"""Build (or refresh) the search index from the catalog folder.

    python index.py                      # uses ./catalog and ./data
    python index.py --catalog D --data X

File name without extension = design number (catalog/D1042.jpg -> "D1042").
Unchanged images are skipped, so re-running after adding a few designs takes seconds.
The running app notices the new index on the next search; no restart needed.
"""
import argparse
import sys

import core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="catalog")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    try:
        meta = core.build_index(args.catalog, args.data)
    except (ValueError, FileNotFoundError) as err:
        sys.exit(f"Error: {err}")
    print(
        f"Done: {len(meta['designs'])} designs, {sum(meta['views_per_design'])} views "
        f"({meta['computed']} newly indexed, {meta['reused']} unchanged)."
    )


if __name__ == "__main__":
    main()
