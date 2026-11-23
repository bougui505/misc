#!/usr/bin/env python3
"""Generate a PNG image or animated GIF of the modular multiplication table mod N by M."""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a circular modular multiplication table (times table) by M modulo N."
    )
    parser.add_argument(
        "-N",
        "-n",
        "--modulus",
        dest="N",
        type=int,
        default=200,
        help="Modulus (number of points around the circle, default: 200).",
    )
    parser.add_argument(
        "-M",
        "-m",
        "--multiplier",
        dest="M",
        type=int,
        default=86,
        help="Multiplication factor (default: 86).",
    )

    fmt_group = parser.add_mutually_exclusive_group()
    fmt_group.add_argument(
        "--png",
        dest="format",
        action="store_const",
        const="png",
        help="Save a static PNG image (default).",
    )
    fmt_group.add_argument(
        "--gif",
        dest="format",
        action="store_const",
        const="gif",
        help="Save an animated GIF.",
    )
    parser.set_defaults(format=None)

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: timestable_<M>_mod_<N>.<png|gif>).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI resolution for PNG output (default: 150).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for GIF output (default: 60).",
    )

    args = parser.parse_args()
    if args.N < 2:
        parser.error("Modulus N must be >= 2.")
    if args.dpi < 1:
        parser.error("Resolution --dpi must be >= 1.")
    if args.fps < 1:
        parser.error("Frame rate --fps must be >= 1.")

    if args.format is None:
        if args.output and args.output.lower().endswith(".gif"):
            args.format = "gif"
        else:
            args.format = "png"

    return args


def _circle_points(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute coordinates of n evenly spaced points on the unit circle."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.cos(angles), np.sin(angles)


def _create_canvas() -> tuple[plt.Figure, plt.Axes]:
    """Create the base square figure with the unit circle."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="black", lw=0.8)
    return fig, ax


def save_png(n: int, m: int, output: str, dpi: int) -> None:
    """Render and save the modular multiplication circle as a PNG image."""
    x, y = _circle_points(n)
    fig, ax = _create_canvas()

    segments = []
    for k in range(1, n):
        dest = (m * k) % n
        if dest != k:
            segments.append([(x[k], y[k]), (x[dest], y[dest])])

    if segments:
        ax.add_collection(
            LineCollection(segments, colors="black", linewidths=0.5, alpha=0.7)
        )

    fig.savefig(output, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_gif(n: int, m: int, output: str, fps: int) -> None:
    """Build and save the modular multiplication circle as an animated GIF."""
    x, y = _circle_points(n)
    fig, ax = _create_canvas()

    def update(k: int) -> list:
        dest = (m * k) % n
        return ax.plot(
            [x[k], x[dest]],
            [y[k], y[dest]],
            color="black",
            lw=0.5,
            alpha=0.7,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=range(1, n),
        init_func=lambda: [],
        interval=1000 / fps,
        repeat=True,
    )
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    """Entry point."""
    args = parse_args()
    output = args.output or f"timestable_{args.M}_mod_{args.N}.{args.format}"
    if args.format == "png":
        save_png(args.N, args.M, output, args.dpi)
    else:
        save_gif(args.N, args.M, output, args.fps)
    print(output)


if __name__ == "__main__":
    main()
