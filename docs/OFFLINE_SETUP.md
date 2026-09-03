# Offline / proxied setup

These modules reach the network in four places. Three happen at **render**
time, but one happens at **import** time, which is the failure mode that
matters: behind a restrictive proxy the module fails to import at all rather
than degrading to a simpler figure.

Everything below can be pre-staged so the toolkit runs fully offline.

---

## 1. `geopack` IGRF coefficients — *fails at import*

`import geopack` runs `init_igrf()`, which downloads coefficient files from
NOAA. If that fetch fails, the import raises and nothing else runs.

**Symptom**

```
ValueError: max() iterable argument is empty
```

**Fix** — place the coefficient files in the package's own data directory:

```bash
python -c "import geopack,os;print(os.path.join(os.path.dirname(geopack.__file__),'igrf_coeffs'))"
# copy igrf14coeffs.txt (and igrf13coeffs.txt) into that directory
```

Source: <https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt>

Without geopack the plots still work — `external_field` degrades to internal
IGRF only, with a printed warning — but the magnetotail will be absent and
field lines beyond ~5 R\_E will be wrong (the external field is 96% of the
internal field at 10 R\_E in the tail).

---

## 2. Earth texture — downloaded on first render

`texture_path="auto"` searches local paths, then downloads NASA Blue Marble
into `~/.cache/ssapy_toolkit/earth_texture.jpg`.

**Pre-stage** either by copying the bundled `earth_texture.jpg` to that cache
path, or to any of the search locations:

```
~/earth_texture.jpg
~/blue_marble.jpg
~/SSAPy-Toolkit/assets/earth_texture.jpg
./assets/earth_texture.jpg
```

Or pass `texture_path=` explicitly. `texture_path="none"` disables it.

---

## 3. AE-8 / AP-8 flux table — built on first render

Built once from `spacepy.irbempy` into
`~/.cache/ssapy_toolkit/aep8_table.npz` (~2 minutes). No network needed *if*
spacepy is installed, but spacepy itself may try to fetch its data on first
use. Copy the `.npz` between machines to skip both.

Without spacepy, `belt_style="flux"` falls back to geometric L-shells with a
printed warning. Note the cache changes what a missing dependency looks like:
once `aep8_table.npz` and the `beltflux_*.npz` samples exist, the flux belts
render **without** spacepy, because nothing needs recomputing. The fallback
only appears on a cold cache — which is exactly the state of a fresh checkout,
and is what the degradation tests in `test_magfield_physics.py` exercise.

`belt_style="flux"` also needs **ppigrf**, since the belt shells are traced
through the real field. Without it the belts fall back to the dipole L-shell
geometry.

---

## 4. Star catalogue — never downloaded

Expected on disk; the modules fall back to a random placeholder sky if absent.
Ship `bright_stars.csv` to one of:

```
~/bright_stars.csv
~/SSAPy/ssapy/data/bright_stars.csv
<module directory>/bright_stars.csv
```

---

## Cache location and hygiene

All generated assets live in `~/.cache/ssapy_toolkit/`, overridable with the
`SSAPY_TOOLKIT_CACHE` environment variable:

| file | what | rebuild cost |
|---|---|---|
| `earth_texture.jpg` | NASA Blue Marble | download |
| `aep8_table.npz` | AE-8/AP-8 over (L, B/B0) | ~2 min |
| `beltflux_*.npz` | traced belt samples | ~80 s each |

`beltflux_*.npz` is keyed on a digest of the physics source, so editing any
field or tracing routine invalidates it automatically — but old entries are
**not** pruned. Delete them periodically.

---

## Windows notes

`spacepy` ships binary wheels for Linux; on Windows a source build needs a
Fortran toolchain. If that is impractical, install everything else and accept
the documented fallbacks — the field, magnetopause, Earth, sky and geometric
L-shell belts all work without it. Only AE-8/AP-8 flux and the McIlwain L
cross-check require spacepy.

---

## Module layout

```
magnetosphere_core.py    geometry, WGS84 Earth, texture, starfield, astrometry
                         (no ppigrf / geopack / spacepy — imports anywhere)
magfield_plot_3d.py      IGRF + Tsyganenko field, tracing, AE-8/AP-8 belts
van_allen_plot_3d.py     belt-focused figure; delegates belt physics to magfield
```

The shared helpers were duplicated across the two plot modules until they
drifted three separate times. They now have one definition; a test fails if a
name is defined in both plot modules again.

## Verifying an install

```bash
pytest -q -m 'not slow'                 # 32 checks, ~60 s
pytest -q -m slow                       # 6 cold-cache / subprocess checks, ~4 min
python validate_against_goes.py --date 2025-07-02 --sat 18   # needs network
```

The physics suite is the one to run after any change: it covers reference
frames, field synthesis, integration, trapped-particle geometry and
astrometry, and it fails loudly if the two plot modules drift apart.
