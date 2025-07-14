import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Load and print contents of a .npz file")
    parser.add_argument("npz_path", type=str, help="Path to the .npz file")
    args = parser.parse_args()

    data = np.load(args.npz_path)
    print(f"Contents of {args.npz_path}:")
    for key in data.files:
        print(f"- {key}:")
        print(data[key].shape)
    data.close()


if __name__ == "__main__":
    main()
