"""Plot a 1D Gaussian (normal) distribution."""

import argparse

import numpy as np
import matplotlib.pyplot as plt


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the normal probability density function."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a Gaussian distribution.")
    parser.add_argument("--mu", type=float, default=0.0, help="Mean")
    parser.add_argument("--sigma", type=float, default=1.0, help="Standard deviation")
    parser.add_argument("--out", type=str, default=None, help="Save path (shows if omitted)")
    args = parser.parse_args()

    x = np.linspace(args.mu - 4 * args.sigma, args.mu + 4 * args.sigma, 500)
    y = gaussian(x, args.mu, args.sigma)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, color="C0", lw=2)
    ax.fill_between(x, y, alpha=0.2, color="C0")
    ax.axvline(args.mu, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(rf"Gaussian: $\mu={args.mu},\ \sigma={args.sigma}$")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
