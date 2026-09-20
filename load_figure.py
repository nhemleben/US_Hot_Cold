"""Reopen a pickled matplotlib Figure (*.fig.pickle) so it can be rotated/zoomed.

Usage:
    python load_figure.py us_record_high_peaks.fig.pickle
"""
import pickle
import sys

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print("Usage: python load_figure.py <file.fig.pickle>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        fig = pickle.load(f)

    plt.show(block=True)


if __name__ == "__main__":
    main()
