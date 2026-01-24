import argparse
import csv
import json
import os
from typing import Dict, Any

from compress_pickle import load


def load_grids(pickle_path: str) -> Dict[str, Any]:
    if not os.path.isfile(pickle_path):
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")
    with open(pickle_path, "rb") as f:
        return load(f)


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_jsonl(grids: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for sample_id, rec in grids.items():
            row = {
                "id": sample_id,
                "path": rec.get("path"),
                "clz": rec.get("clz"),
                "actid": (rec.get("info") or {}).get("actid"),
                "info": rec.get("info"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_csv(grids: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "path", "clz", "actid"]) 
        for sample_id, rec in grids.items():
            info = rec.get("info") or {}
            writer.writerow([
                sample_id,
                rec.get("path"),
                rec.get("clz"),
                info.get("actid"),
            ])


def summarize(grids: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in grids.values():
        actid = (rec.get("info") or {}).get("actid")
        if actid is None:
            continue
        counts[str(actid)] = counts.get(str(actid), 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Export KuroSiwo V2 pickle to JSONL/CSV for inspection")
    parser.add_argument("--pickle-path", type=str, default=None, help="Path to a .gz pickle (single file mode)")
    parser.add_argument("--pickle-dir", type=str, default=None, help="Directory containing KuroV2_grid_dict*.gz")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for exported files (default: <pickle_dir>/export)")
    parser.add_argument("--format", type=str, choices=["jsonl", "csv", "both"], default="both")
    args = parser.parse_args()

    targets = []
    if args.pickle_path:
        base = os.path.splitext(os.path.basename(args.pickle_path))[0]
        targets.append((base, args.pickle_path))
    elif args.pickle_dir:
        for name in ["KuroV2_grid_dict.gz", "KuroV2_grid_dict_test_0_100.gz"]:
            p = os.path.join(args.pickle_dir, name)
            if os.path.isfile(p):
                base = os.path.splitext(os.path.basename(p))[0]
                targets.append((base, p))
    else:
        # Default: repo_root/data/KuroSiwoGRD/pickle
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        default_dir = os.path.join(repo_root, "data", "KuroSiwoGRD", "pickle")
        for name in ["KuroV2_grid_dict.gz", "KuroV2_grid_dict_test_0_100.gz"]:
            p = os.path.join(default_dir, name)
            if os.path.isfile(p):
                base = os.path.splitext(os.path.basename(p))[0]
                targets.append((base, p))
        if not targets:
            raise ValueError("Either --pickle-path or --pickle-dir must be provided, and defaults not found at data/KuroSiwoGRD/pickle")

    # Determine default output directory
    if args.out_dir is None:
        # If a dir is provided, use it; if a file is provided, use its parent dir.
        if args.pickle_dir:
            out_dir = os.path.join(args.pickle_dir, "export")
        elif args.pickle_path:
            out_dir = os.path.join(os.path.dirname(args.pickle_path), "export")
        else:
            # export/ under the default dir
            out_dir = os.path.join(os.path.dirname(targets[0][1]), "export")
    else:
        out_dir = args.out_dir

    ensure_out_dir(out_dir)

    for base, pkl in targets:
        grids = load_grids(pkl)

        if args.format in ("jsonl", "both"):
            out_jsonl = os.path.join(out_dir, f"{base}.jsonl")
            export_jsonl(grids, out_jsonl)

        if args.format in ("csv", "both"):
            out_csv = os.path.join(out_dir, f"{base}.csv")
            export_csv(grids, out_csv)

        counts = summarize(grids)
        counts_sorted = sorted(counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        print(f"[{base}] total={len(grids)} unique_actids={len(counts)}")
        print(f"[{base}] actid_counts_top10={counts_sorted[:10]}")


if __name__ == "__main__":
    main()


