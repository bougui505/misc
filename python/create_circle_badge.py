#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rembg[cpu]>=2.0.80",
#     "opencv-python>=4.8.0",
#     "pillow>=10.0.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
create_circle_badge.py

Processes any portrait photograph into a high-quality circular monochrome badge
matching the official presentation aesthetic:
- Automatic neural foreground isolation (rembg) to seamless pure white (#ffffff)
- Intelligent face detection & auto-centering with balanced headroom & margins
- Fine-art B&W conversion with local contrast adaptation (CLAHE)
- Anti-aliased circular alpha mask (PNG)

Usage:
    ./scripts/create_circle_badge.py input.jpg [output.png]
    uv run ./scripts/create_circle_badge.py input.jpg -o output.png --size 512
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session


def isolate_foreground(img_bgr: np.ndarray) -> np.ndarray:
    """Isolate subject using rembg neural alpha matting."""
    print("  -> Isolating foreground with rembg...")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_in = Image.fromarray(img_rgb)
    session = new_session("bria-rmbg")
    pil_out = remove(pil_in, session=session)
    rgba = np.array(pil_out)
    return rgba  # RGBA


def detect_head_center_and_size(img_bgr: np.ndarray, alpha: np.ndarray):
    """
    Detect face center and head size using OpenCV Haar cascades,
    with robust fallback to alpha-mask center of mass.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Try Haar Cascade if available
    try:
        cascade_path = getattr(cv2, "data", None)
        if cascade_path and hasattr(cv2, "CascadeClassifier"):
            xml_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(xml_file)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(int(w * 0.1), int(h * 0.1))
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                cx = fx + fw // 2
                cy = fy + fh // 2
                crop_size = int(max(fw, fh) * 2.3)
                print(f"  -> Detected face at ({cx}, {cy}), face size {fw}x{fh}, crop window {crop_size}px")
                return cx, cy, crop_size
    except Exception as e:
        print(f"  -> Face detector skipped ({e})")

    # Fallback to alpha bounding box
    print("  -> Face detector found no matches; falling back to alpha mask silhouette analysis...")
    ys, xs = np.where(alpha > 30)
    if len(ys) == 0:
        return w // 2, h // 2, min(w, h)

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    fg_h = ymax - ymin
    fg_w = xmax - xmin

    # Head is typically in the upper 40% of the subject silhouette
    cx = int((xmin + xmax) / 2)
    cy = int(ymin + fg_h * 0.35)
    crop_size = int(max(fg_w, fg_h * 0.75) * 1.3)
    return cx, cy, crop_size


def make_badge(
    input_path: str,
    output_path: str,
    target_dim: int = 512,
    custom_cx: int = None,
    custom_cy: int = None,
    custom_size: int = None,
    skip_rembg: bool = False,
):
    inp_file = Path(input_path)
    if not inp_file.exists():
        print(f"Error: input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Loading image '{inp_file}'...")
    img = cv2.imread(str(inp_file), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: unable to decode '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    # Ensure 3-channel BGR base
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    elif img.shape[2] == 4:
        img_bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        img_bgr = img
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

    # [2/4] Background isolation
    if not skip_rembg:
        print("[2/4] Neural matting on background...")
        rgba = isolate_foreground(img_bgr)
        rgb_fg = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha_fg = rgba[:, :, 3]
    else:
        print("[2/4] Skipping neural matting; using original alpha/background...")
        rgb_fg = img_bgr
        alpha_fg = alpha

    h, w, _ = rgb_fg.shape

    # Composite cleanly on pure white
    alpha_norm = (alpha_fg.astype(float) / 255.0)[:, :, np.newaxis]
    white_bg = np.ones_like(rgb_fg, dtype=float) * 255.0
    composited_bgr = (rgb_fg.astype(float) * alpha_norm + white_bg * (1.0 - alpha_norm)).astype(np.uint8)

    # [3/4] Auto-crop & framing
    print("[3/4] Framing head and shoulders...")
    if custom_cx is not None and custom_cy is not None and custom_size is not None:
        cx, cy, crop_size = custom_cx, custom_cy, custom_size
    else:
        cx, cy, crop_size = detect_head_center_and_size(img_bgr, alpha_fg)

    pad = crop_size
    padded = cv2.copyMakeBorder(
        composited_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    y1 = cy + pad - crop_size // 2
    y2 = y1 + crop_size
    x1 = cx + pad - crop_size // 2
    x2 = x1 + crop_size

    crop = padded[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_dim, target_dim), interpolation=cv2.INTER_LANCZOS4)

    # [4/4] Fine-art B&W conversion & circle mask
    print("[4/4] Applying monochrome contrast grading and circular alpha...")
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    gray_bw = cv2.addWeighted(gray, 0.65, gray_clahe, 0.35, 0)

    # Ensure background is strictly white
    mask_near_white = (resized[:, :, 0] > 248) & (resized[:, :, 1] > 248) & (resized[:, :, 2] > 248)
    gray_bw[mask_near_white] = 255

    # Circular anti-aliased alpha mask
    circle_alpha = np.zeros((target_dim, target_dim), dtype=np.uint8)
    cv2.circle(
        circle_alpha,
        (target_dim // 2, target_dim // 2),
        (target_dim // 2) - 2,
        255,
        -1,
        cv2.LINE_AA,
    )

    rgba_final = cv2.merge([gray_bw, gray_bw, gray_bw, circle_alpha])

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), rgba_final)
    print(f"Successfully generated circular badge at '{out_file}' ({target_dim}x{target_dim} PNG)!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-quality circular monochrome collaborator badges with pure white background."
    )
    parser.add_argument("input", help="Path to input photograph (JPG, PNG, etc.)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to output PNG badge (defaults to <input_stem>_circle.png)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Target square badge dimension in pixels (default: 512)",
    )
    parser.add_argument("--cx", type=int, default=None, help="Custom crop center X coordinate")
    parser.add_argument("--cy", type=int, default=None, help="Custom crop center Y coordinate")
    parser.add_argument("--crop-size", type=int, default=None, help="Custom crop box size in pixels")
    parser.add_argument(
        "--no-rembg",
        action="store_true",
        help="Skip neural background removal (if input already has white or transparent background)",
    )

    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_bw_circle.png")

    make_badge(
        input_path=args.input,
        output_path=args.output,
        target_dim=args.size,
        custom_cx=args.cx,
        custom_cy=args.cy,
        custom_size=args.crop_size,
        skip_rembg=args.no_rembg,
    )


if __name__ == "__main__":
    main()
