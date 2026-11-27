#!/usr/bin/env python3
"""Overlay a customizable drawing grid on an image to assist with hand-drawing."""

import argparse
import math
import sys
from pathlib import Path

from PIL import (
    Image,
    ImageColor,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Overlay a customizable drawing grid on an image for hand-drawing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("input", type=Path, help="Path to the input image.")
    io_group.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <stem>_grid<ext> next to input).",
    )
    io_group.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Output quality for JPEG/WebP (1-100).",
    )
    io_group.add_argument(
        "--show",
        action="store_true",
        help="Display the resulting image after saving.",
    )

    geom_group = parser.add_argument_group("Grid Geometry")
    geom_group.add_argument(
        "-c",
        "--cols",
        type=int,
        default=None,
        help="Number of columns (default: 4 with square cells if no geometry option is given).",
    )
    geom_group.add_argument(
        "-r",
        "--rows",
        type=int,
        default=None,
        help="Number of rows (if only --cols or --rows is given, cells are square).",
    )
    geom_group.add_argument(
        "-s",
        "--cell-size",
        type=float,
        default=None,
        help="Exact square cell size in pixels (mutually exclusive with --cols/--rows).",
    )
    geom_group.add_argument(
        "--center",
        action="store_true",
        help="Center the grid when cells do not divide the image evenly.",
    )
    geom_group.add_argument(
        "--border",
        action="store_true",
        help="Draw an outer border around the image.",
    )

    style_group = parser.add_argument_group("Line Style")
    style_group.add_argument(
        "--color",
        type=str,
        default="red",
        help="Grid line color (any CSS/Pillow color name or hex, e.g. 'red', '#00ff00').",
    )
    style_group.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Main grid line thickness in pixels.",
    )
    style_group.add_argument(
        "--opacity",
        type=float,
        default=0.7,
        help="Grid line opacity from 0.0 (transparent) to 1.0 (opaque).",
    )
    style_group.add_argument(
        "--dashed",
        action="store_true",
        help="Draw dashed grid lines instead of solid lines.",
    )
    style_group.add_argument(
        "--dash-length",
        type=int,
        default=10,
        help="Dash and gap length in pixels when --dashed is enabled.",
    )
    style_group.add_argument(
        "--diagonals",
        action="store_true",
        help="Draw diagonal X-guides inside each cell (thinner, half opacity).",
    )
    style_group.add_argument(
        "--subdivisions",
        type=int,
        default=1,
        help="Subdivide each cell into NxN smaller cells (e.g. 2 for halves).",
    )

    label_group = parser.add_argument_group("Coordinate Labels")
    label_group.add_argument(
        "--labels",
        action="store_true",
        help="Add A, B, C... column labels and 1, 2, 3... row labels.",
    )
    label_group.add_argument(
        "--label-position",
        choices=["margin", "inside"],
        default="margin",
        help="Place labels in an added margin around the image or inside top/left cells.",
    )
    label_group.add_argument(
        "--label-color",
        type=str,
        default=None,
        help="Label text color (default: 'black' for margin, or --color for inside).",
    )
    label_group.add_argument(
        "--label-bg",
        type=str,
        default="white",
        help="Background color for the label margin when --label-position=margin.",
    )
    label_group.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Label font size in pixels (default: auto-scaled to cell/image size).",
    )

    adj_group = parser.add_argument_group("Image Adjustments")
    adj_group.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert the image to grayscale before drawing the grid.",
    )
    adj_group.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Contrast adjustment factor (1.0 = original).",
    )
    adj_group.add_argument(
        "--brightness",
        type=float,
        default=1.0,
        help="Brightness adjustment factor (1.0 = original).",
    )
    adj_group.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help="Gaussian blur radius in pixels (0.0 = no blur).",
    )

    args = parser.parse_args(argv)

    if args.cell_size is not None and (args.cols is not None or args.rows is not None):
        parser.error("--cell-size cannot be combined with --cols or --rows.")
    if args.cell_size is not None and args.cell_size <= 0:
        parser.error("--cell-size must be positive.")
    if args.cols is not None and args.cols <= 0:
        parser.error("--cols must be positive.")
    if args.rows is not None and args.rows <= 0:
        parser.error("--rows must be positive.")
    if args.subdivisions < 1:
        parser.error("--subdivisions must be >= 1.")
    if args.thickness < 1:
        parser.error("--thickness must be >= 1.")
    if args.dash_length < 1:
        parser.error("--dash-length must be >= 1.")

    return args


def col_label(index: int) -> str:
    """Convert a 0-based column index to Excel-style letters (0 -> A, 25 -> Z, 26 -> AA)."""
    label = ""
    n = index
    while n >= 0:
        label = chr(ord("A") + (n % 26)) + label
        n = n // 26 - 1
    return label


def parse_rgba(color_str: str, opacity: float) -> tuple[int, int, int, int]:
    """Parse a color string and opacity (0.0-1.0) into an RGBA tuple."""
    rgb = ImageColor.getrgb(color_str)
    alpha = int(round(max(0.0, min(1.0, opacity)) * 255))
    return (rgb[0], rgb[1], rgb[2], alpha)


def compute_grid_lines(
    width: int,
    height: int,
    cols: int | None,
    rows: int | None,
    cell_size: float | None,
    center: bool,
) -> tuple[list[float], list[float]]:
    """Compute x and y coordinates for the grid lines."""
    if cell_size is not None:
        step_x = step_y = float(cell_size)
    elif cols is not None and rows is not None:
        step_x = width / cols
        step_y = height / rows
    elif cols is not None:
        step_x = step_y = width / cols
    elif rows is not None:
        step_x = step_y = height / rows
    else:
        step_x = step_y = width / 4.0

    def _axis_lines(length: int, step: float, count: int | None) -> list[float]:
        if count is not None and (cols is not None and rows is not None):
            return [i * step for i in range(count + 1)]
        n_full = int(length // step)
        if center and n_full > 0:
            used = n_full * step
            offset = (length - used) / 2.0
            lines = [offset + i * step for i in range(n_full + 1)]
        else:
            lines = []
            pos = 0.0
            while pos <= length + 1e-6:
                lines.append(pos)
                pos += step
        return lines

    x_lines = _axis_lines(width, step_x, cols)
    y_lines = _axis_lines(height, step_y, rows)
    return x_lines, y_lines


def draw_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    fill: tuple[int, int, int, int],
    width: int,
    dashed: bool = False,
    dash_length: int = 10,
) -> None:
    """Draw a solid or dashed line segment."""
    if not dashed:
        draw.line([p1, p2], fill=fill, width=width)
        return

    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    total_len = math.hypot(dx, dy)
    if total_len == 0:
        return

    ux, uy = dx / total_len, dy / total_len
    pos = 0.0
    while pos < total_len:
        end = min(pos + dash_length, total_len)
        sx, sy = x1 + ux * pos, y1 + uy * pos
        ex, ey = x1 + ux * end, y1 + uy * end
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        pos += dash_length * 2


def apply_adjustments(
    img: Image.Image,
    grayscale: bool,
    contrast: float,
    brightness: float,
    blur: float,
) -> Image.Image:
    """Apply EXIF orientation, grayscale, contrast, brightness, and blur."""
    img = ImageOps.exif_transpose(img).convert("RGBA")

    if grayscale:
        gray = ImageOps.grayscale(img)
        img = Image.merge("RGBA", (gray, gray, gray, img.getchannel("A")))

    if contrast != 1.0:
        rgb = ImageEnhance.Contrast(img.convert("RGB")).enhance(contrast)
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(img.convert("RGB")).enhance(brightness)
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

    if blur > 0.0:
        rgb = img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

    return img


def draw_grid_overlay(
    img: Image.Image,
    x_lines: list[float],
    y_lines: list[float],
    color: str,
    thickness: int,
    opacity: float,
    dashed: bool,
    dash_length: int,
    diagonals: bool,
    subdivisions: int,
    border: bool,
) -> Image.Image:
    """Draw the grid, subdivisions, diagonals, and optional border onto the image."""
    width, height = img.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    main_rgba = parse_rgba(color, opacity)
    sub_rgba = parse_rgba(color, opacity * 0.5)
    sub_width = max(1, thickness // 2)

    # Subdivisions
    if subdivisions > 1:
        for i in range(len(x_lines) - 1):
            x0, x1 = x_lines[i], x_lines[i + 1]
            for s in range(1, subdivisions):
                sx = x0 + (x1 - x0) * (s / subdivisions)
                draw_line(
                    draw, (sx, 0), (sx, height), sub_rgba, sub_width, dashed, dash_length
                )
        for j in range(len(y_lines) - 1):
            y0, y1 = y_lines[j], y_lines[j + 1]
            for s in range(1, subdivisions):
                sy = y0 + (y1 - y0) * (s / subdivisions)
                draw_line(
                    draw, (0, sy), (width, sy), sub_rgba, sub_width, dashed, dash_length
                )

    # Diagonals inside each cell
    if diagonals:
        for i in range(len(x_lines) - 1):
            x0, x1 = x_lines[i], x_lines[i + 1]
            for j in range(len(y_lines) - 1):
                y0, y1 = y_lines[j], y_lines[j + 1]
                draw_line(
                    draw, (x0, y0), (x1, y1), sub_rgba, sub_width, dashed, dash_length
                )
                draw_line(
                    draw, (x1, y0), (x0, y1), sub_rgba, sub_width, dashed, dash_length
                )

    # Main grid lines
    for x in x_lines:
        draw_line(draw, (x, 0), (x, height), main_rgba, thickness, dashed, dash_length)
    for y in y_lines:
        draw_line(draw, (0, y), (width, y), main_rgba, thickness, dashed, dash_length)

    # Optional outer border
    if border:
        draw.rectangle(
            [(0, 0), (width - 1, height - 1)], outline=main_rgba, width=thickness
        )

    return Image.alpha_composite(img, overlay)


def add_labels(
    img: Image.Image,
    x_lines: list[float],
    y_lines: list[float],
    position: str,
    label_color: str | None,
    label_bg: str,
    grid_color: str,
    font_size: int | None,
) -> Image.Image:
    """Add A, B, C... column labels and 1, 2, 3... row labels."""
    width, height = img.size

    if font_size is None:
        cell_w = (x_lines[1] - x_lines[0]) if len(x_lines) > 1 else width / 4
        cell_h = (y_lines[1] - y_lines[0]) if len(y_lines) > 1 else height / 4
        font_size = max(12, min(48, int(min(cell_w, cell_h) * 0.25)))

    font = ImageFont.load_default(size=font_size)

    if position == "margin":
        text_color = ImageColor.getrgb(label_color or "black") + (255,)
        bg_color = ImageColor.getrgb(label_bg) + (255,)
        margin = max(font_size + 16, int(font_size * 1.8))

        canvas = Image.new("RGBA", (width + margin, height + margin), bg_color)
        canvas.paste(img, (margin, margin))
        draw = ImageDraw.Draw(canvas)

        for i in range(len(x_lines) - 1):
            cx = margin + (x_lines[i] + x_lines[i + 1]) / 2.0
            cy = margin / 2.0
            draw.text((cx, cy), col_label(i), fill=text_color, font=font, anchor="mm")

        for j in range(len(y_lines) - 1):
            cx = margin / 2.0
            cy = margin + (y_lines[j] + y_lines[j + 1]) / 2.0
            draw.text((cx, cy), str(j + 1), fill=text_color, font=font, anchor="mm")

        return canvas

    # position == "inside"
    text_color = ImageColor.getrgb(label_color or grid_color) + (255,)
    draw = ImageDraw.Draw(img)
    pad = max(4, font_size // 4)

    for i in range(len(x_lines) - 1):
        lx = x_lines[i] + pad
        ly = (y_lines[0] if y_lines else 0) + pad
        draw.text((lx, ly), col_label(i), fill=text_color, font=font, anchor="lt")

    for j in range(1, len(y_lines) - 1):
        lx = (x_lines[0] if x_lines else 0) + pad
        ly = y_lines[j] + pad
        draw.text((lx, ly), str(j + 1), fill=text_color, font=font, anchor="lt")

    return img


def save_image(img: Image.Image, output_path: Path, quality: int) -> None:
    """Save the image, converting RGBA to RGB for formats without alpha support."""
    ext = output_path.suffix.lower()
    save_kwargs: dict = {}
    if ext in {".jpg", ".jpeg", ".bmp"}:
        img = img.convert("RGB")
        if ext in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = quality
    elif ext == ".webp":
        save_kwargs["quality"] = quality

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, **save_kwargs)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    if not args.input.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    output_path = (
        args.output
        if args.output is not None
        else args.input.with_name(f"{args.input.stem}_grid{args.input.suffix}")
    )

    with Image.open(args.input) as src:
        img = apply_adjustments(
            src,
            grayscale=args.grayscale,
            contrast=args.contrast,
            brightness=args.brightness,
            blur=args.blur,
        )

    width, height = img.size
    x_lines, y_lines = compute_grid_lines(
        width=width,
        height=height,
        cols=args.cols,
        rows=args.rows,
        cell_size=args.cell_size,
        center=args.center,
    )

    img = draw_grid_overlay(
        img=img,
        x_lines=x_lines,
        y_lines=y_lines,
        color=args.color,
        thickness=args.thickness,
        opacity=args.opacity,
        dashed=args.dashed,
        dash_length=args.dash_length,
        diagonals=args.diagonals,
        subdivisions=args.subdivisions,
        border=args.border,
    )

    if args.labels:
        img = add_labels(
            img=img,
            x_lines=x_lines,
            y_lines=y_lines,
            position=args.label_position,
            label_color=args.label_color,
            label_bg=args.label_bg,
            grid_color=args.color,
            font_size=args.font_size,
        )

    save_image(img, output_path, args.quality)
    print(f"Saved gridded image to {output_path}")

    if args.show:
        img.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
