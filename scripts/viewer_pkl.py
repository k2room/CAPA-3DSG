"""
    Script to view the structure of a pickled file.
    Usage:
        python viewer_pkl.py --path <path_to_pickle_file>
"""

import gzip, pickle as pkl
import argparse

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)

    return parser

def tree_summary(x, depth=0, max_list=3, max_dict=20):
    pad = "  " * depth
    if isinstance(x, dict):
        keys = list(x.keys())[:max_dict]
        print(pad + f"dict(len={len(x)}): {keys} ...")
        for k in keys:
            print(pad + f"Key: {k}")
            tree_summary(x[k], depth+1)
    elif isinstance(x, list):
        print(pad + f"list(len={len(x)}): showing first {min(len(x),max_list)}")
        for i in range(min(len(x), max_list)):
            tree_summary(x[i], depth+1)
    elif hasattr(x, "shape"):
        print(pad + f"array(shape={x.shape}, dtype={x.dtype})")
    else:
        print(pad + f"{type(x).__name__}: {str(x)[:80]}")

def main():
    args = get_parser().parse_args()
    with gzip.open(args.path, "rb") as f:
        data = pkl.load(f)

    tree_summary(data)

if __name__ == "__main__":
    main()