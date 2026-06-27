import matplotlib.pyplot as plt


def plot_periodogram(bls_output):
    """
    Plot the Box Least Squares (BLS) periodogram.

    Parameters
    ----------
    bls_output : dict
        Output dictionary returned by run_bls().
    """

    plt.figure(figsize=(10, 5))

    # Plot the BLS power spectrum
    plt.plot(
        bls_output["periods"],
        bls_output["powers"],
        color="blue",
        linewidth=1
    )

    # Highlight the best candidate
    plt.scatter(
        bls_output["period"],
        bls_output["power"],
        color="red",
        s=80,
        label="Best Candidate",
        zorder=5
    )

    plt.title("BLS Periodogram")
    plt.xlabel("Orbital Period (days)")
    plt.ylabel("BLS Power")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()