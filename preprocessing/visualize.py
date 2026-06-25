# visualize.py

import matplotlib.pyplot as plt

from config import FIG_SIZE, MARKER_SIZE


def plot_lightcurve(time, flux, title="Light Curve"):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(time, flux, ".", markersize=MARKER_SIZE)
    plt.title(title)
    plt.xlabel("Time (BJD - 2457000)")
    plt.ylabel("Normalized Flux")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_detrending(time, flux, trend):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(time, flux, ".", markersize=MARKER_SIZE, label="Normalized Flux")
    plt.plot(time, trend, "r-", label="Trend")
    plt.title("Light Curve with Trend")
    plt.xlabel("Time (BJD - 2457000)")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()