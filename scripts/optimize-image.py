#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pillow>=10.0.0",
#     "numpy>=1.24",
# ]
# ///
"""Post-process a generated image IN PLACE:

  optimize-image.py <path> [max_dimension] [--cutout]

- max_dimension (optional positional): downscale the longest edge to this many px.
- --cutout: knock out a light, low-saturation background (Gemini's "white" is really a
  light grey ~238) to transparent, so the art reads as a sticker, not a pasted box. The
  saturated subject is preserved; only light + near-grey pixels go transparent (soft ramp).

Backward compatible with the older 2-positional call. uv installs deps automatically.
"""

import argparse
import sys
from PIL import Image
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("max_dimension", nargs="?", type=int, default=None)
    ap.add_argument("--cutout", action="store_true",
                    help="make a light grey/white background transparent")
    args = ap.parse_args()

    im = Image.open(args.path)

    if args.max_dimension and args.max_dimension > 0:
        w, h = im.size
        longest = max(w, h)
        if longest > args.max_dimension:
            s = args.max_dimension / longest
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)

    if args.cutout:
        arr = np.asarray(im.convert("RGBA")).astype(np.int32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mn = np.minimum(np.minimum(r, g), b)
        sat = np.maximum(np.maximum(r, g), b) - mn
        # Light pixels fade to transparent (>=218 fully, <=200 opaque, ramp between);
        # only applied where the pixel is near-grey (low saturation) so colour survives.
        ramp = np.clip((218 - mn) * 255 // 18, 0, 255)
        light_alpha = np.where(mn >= 218, 0, np.where(mn <= 200, 255, ramp))
        arr[..., 3] = np.where(sat <= 20, light_alpha, 255)
        im = Image.fromarray(arr.astype("uint8"), "RGBA")

    im.save(args.path, format="PNG", optimize=True)
    print(args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
