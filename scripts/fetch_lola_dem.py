"""
fetch_lola_dem.py — prepare the lunar height map for SSAPy-Data

Downloads the LOLA gridded topography published in NASA's CGI Moon Kit and
writes it as a compressed .npz for moon_render_lola.

    python fetch_lola_dem.py --out ../SSAPy-Data/src/ssapy_data/data/moon_dem.npz

Source: https://svs.gsfc.nasa.gov/4720 — LOLA gridded data products,
reformatted by NASA GSFC's Scientific Visualization Studio. Unsigned 16-bit
TIFF in half-metres, offset +20000 (10 km), relative to a 1737.4 km sphere,
0 deg longitude centred. NASA data, public domain, so unlike the HYG star
catalogue there is no attribution condition on redistribution — worth noting
since this ships inside SSAPy-Data.

Resolution: 16 pixels/degree is 5760x2880 and about 33 MB as TIFF. The
default here downsamples to 8 ppd (2880x1440, roughly 8 MB compressed),
which is still four times finer than a 0.5 deg render mesh. Pass
--full for the native grid.

Decoding is verified by range: the output must land near -9.1 to +10.8 km,
LOLA's true global relief.
"""

import argparse
import io
import os
import urllib.request

import numpy as np

URL = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16_uint.tif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="moon_dem.npz")
    ap.add_argument("--full", action="store_true",
                    help="keep the native 16 ppd grid instead of downsampling")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    print(f"downloading {args.url}")
    with urllib.request.urlopen(args.url, timeout=600) as resp:
        raw = resp.read()
    print(f"  {len(raw)/1e6:.1f} MB")

    img = Image.open(io.BytesIO(raw))
    arr = np.asarray(img).astype(np.float64)
    print(f"  grid {arr.shape[1]}x{arr.shape[0]}, mode {img.mode}")

    # uint16 half-metres, +20000 offset, relative to R = 1737.4 km
    elev_km = (arr - 20000.0) / 2000.0

    lo, hi = elev_km.min(), elev_km.max()
    print(f"  elevation {lo:.2f} to {hi:.2f} km  (LOLA truth: -9.1 to +10.8)")
    if not (-11.0 < lo < -7.0 and 8.0 < hi < 13.0):
        raise SystemExit("decoded elevations are outside LOLA's known range; "
                         "the source format has probably changed")

    if not args.full:
        h, w = elev_km.shape
        elev_km = elev_km.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        print(f"  downsampled to {elev_km.shape[1]}x{elev_km.shape[0]}")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, elev_km=elev_km.astype(np.float32))
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
