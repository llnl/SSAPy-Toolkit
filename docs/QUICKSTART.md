# Quickstart

## Make a figure

```python
from magfield_plot_3d import quick_figure

quick_figure("public", save_path="magnetosphere")   # -> magnetosphere.html
```

Four presets:

| preset | what it's for | time (warm cache) |
|---|---|---|
| `draft` | fast preview while iterating | ~30 s |
| `poster` | full fidelity for print | ~2 min |
| `public` | poster + Sun, reference orbits, open/closed field lines | ~2 min |
| `measure` | orthographic, undecorated, for reading geometry off | ~2 min |

Override anything:

```python
quick_figure("poster", max_r_re=25, kp=5, save_path="storm")
```

The belts-only figure works the same way:

```python
from van_allen_plot_3d import plot_van_allen_3d
plot_van_allen_3d(save_path="belts")
```

**First run is slower** — several minutes — because it downloads the Earth
texture and builds the field grid and belt samples. These are cached in
`~/.cache/ssapy_toolkit/` and reused. See `OFFLINE_SETUP.md` to pre-stage them.

---

## Reading the figure

**Field lines** are traced through the real field, coloured by strength |B|.
Solid lines close on Earth at both ends. Dashed lines are *open* — connected to
the solar wind at one end. Open lines are the polar cap; they are what the
magnetotail is made of.

**The gumdrop shape** is the magnetopause: where solar-wind pressure balances
Earth's field. Blunt on the Sun side, flaring downstream. Its size is driven by
the measured solar wind, so it moves — at the November 2025 storm peak it was
compressed from 10 R\_E to 5.3, inside geostationary orbit.

**The coloured shells** are radiation belt contours — NASA AE-8 (electrons,
blue) and AP-8 (protons, orange) — labelled in particles/cm²/s. They are
*climatological means*, not a forecast; see the accuracy note below.

**Reference orbits** (LEO, GPS, GEO) are drawn so you can see what sits where.
GEO at 6.6 R\_E is inside the outer electron belt.

**The block at lower left** lists every model used, the solar-wind drivers for
that date, and how far each has been measured from reality.

---

## Which epoch to ask for

`quick_figure` defaults to `latest_driven_epoch()` — the most recent date with
a complete OMNI solar-wind record, currently about **three weeks behind real
time**. That lag is the price of validated drivers. Asking for today's date
still works, but the solar wind falls back to nominal values and the figure
quietly stops describing any particular day.

```python
from magfield_plot_3d import latest_driven_epoch
print(latest_driven_epoch())      # newest fully-driven date
```

---

## How accurate is it

Measured against spacecraft, not asserted. All of this is printed on the figure
and lives in `magfield_plot_3d.VALIDATION`.

| region | result |
|---|---|
| 6.6 R\_E (GOES-18) | **17.2%** of \|B\| on a moderately disturbed day; 10.4% when quiet |
| 10.2 R\_E (MMS) | **23.2%** inside the magnetosphere |
| geomagnetic storm | **40%** (T96) vs 53% (T89), Kp 8.7 |
| magnetopause | agreement degrades **3.6×** on crossing it — the boundary is where the model says |
| sky | star directions within **8–24 arcsec** of astropy |
| belts | AE-8 reads **~18× above** measured flux, which itself varies **82× in 8 days** |

Two things to take from that. Accuracy **falls with distance** — good at
geostationary orbit, worse further out, and outside the magnetopause the models
are not defined at all. And the **belts are a long-term average**: no single
day looks like the drawn contours.

Regenerate any of these numbers with:

```bash
python validate_against_goes.py --date 2026-07-08          # field at GEO
python validate_tail_and_magnetopause.py --model t96       # deep field + boundary
python validate_belts_against_goes.py --days 8             # belt flux
```

---

## If something is missing

The modules degrade rather than crash, and say so:

| missing | effect |
|---|---|
| `geopack` | no external field — no magnetotail, outer field lines wrong |
| `spacepy` | belts fall back to geometric L-shells |
| `ppigrf` | van_allen falls back to dipole belts; magfield needs it |
| star catalogue | random placeholder sky |
| Earth texture | flat blue globe |

`geopack` is the one to watch: it downloads IGRF coefficients **at import**, so
behind a proxy it fails before anything runs. `OFFLINE_SETUP.md` has the fix.

---

## Tests

```bash
pytest -q -m "not slow"    # 33 checks, ~25 s
pytest -q -m slow          #  6 cold-cache / subprocess checks, ~4 min
```

The suite covers reference frames, field synthesis, integration, trapped
particles and astrometry, and fails if the two plot modules start duplicating
each other again.
