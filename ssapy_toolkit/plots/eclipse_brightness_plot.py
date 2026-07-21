"""
eclipse_brightness_plot.py — core two-circle eclipse physics
==================================================================
This is the shared physics module every other eclipse file in the
toolkit imports from (`propagate_eci`, `sun_direction_eci`,
`illumination_fraction`, plus the RE_KM/R_SUN_KM/AU_KM constants) but
which wasn't part of the uploaded set — reconstructed here from its
documented call signature (consistent across eclipse_demo.py,
eclipse_space_view_plotly.py, and globe_orbit_daynight_plotly.py) so the
full eclipse pipeline can actually be run and checked end-to-end rather
than only unit-testing the renderers in isolation.

Core idea (real two-circle angular-disk-overlap illumination model):
given the illuminated body's position relative to an occluding body,
and the Sun's direction, compute what fraction of the Sun's *apparent
disk* (not just "in shadow or not") is still visible. This is what lets
partial eclipses come out as a continuous 0-1 value instead of a binary
flag, and is reused for lunar eclipses, solar eclipses, and ordinary
satellite day/night crossings by swapping which body is the occluder.
"""
from __future__ import annotations
import numpy as np

MU_EARTH_KM3S2 = 398_600.4418
RE_KM = 6_378.137
R_SUN_KM = 695_700.0
AU_KM = 149_597_870.7


def propagate_eci(a_km, e, inc_deg, raan_deg, argp_deg, nu0_deg,
                  n_orbits=1.0, n_steps=1500):
    """Two-body Keplerian propagation in ECI, vectorized over n_steps.
    Same implementation as globe_orbit_daynight_plotly.py's copy (kept
    in sync deliberately — both files need it, and this is the module
    that's supposed to own it)."""
    def _solve_kepler(M, e, tol=1e-10, max_iter=60):
        E = M.copy()
        for _ in range(max_iter):
            dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            E -= dE
            if np.max(np.abs(dE)) < tol:
                break
        return E

    inc, raan, argp = np.radians([inc_deg, raan_deg, argp_deg])
    nu0 = np.radians(nu0_deg)
    E0 = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu0/2), np.sqrt(1+e)*np.cos(nu0/2))
    M0 = E0 - e*np.sin(E0)
    T_s = 2*np.pi*np.sqrt(a_km**3/MU_EARTH_KM3S2)
    t_s = np.linspace(0, n_orbits*T_s, n_steps)
    n_rad_s = np.sqrt(MU_EARTH_KM3S2/a_km**3)
    E = _solve_kepler(M0 + n_rad_s*t_s, e)
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r_mag = a_km*(1-e*np.cos(E))
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    R11 = cO*cw - sO*sw*ci; R12 = -cO*sw - sO*cw*ci
    R21 = sO*cw + cO*sw*ci; R22 = -sO*sw + cO*cw*ci
    R31 = sw*si;             R32 = cw*si
    xp, yp = r_mag*np.cos(nu), r_mag*np.sin(nu)
    x = R11*xp + R12*yp; y = R21*xp + R22*yp; z = R31*xp + R32*yp
    return t_s, np.stack([x, y, z], axis=1), T_s


def sun_direction_eci(t_s, epoch_jd=2_460_500.0):
    """Low-precision (~0.01 deg) solar ecliptic-longitude approximation,
    same formula used throughout the toolkit — good enough for finding
    eclipse geometry, not a JPL-ephemeris replacement."""
    jd = epoch_jd + t_s/86400.0
    n_days = jd - 2_451_545.0
    L = np.radians((280.460 + 0.9856474*n_days) % 360)
    g = np.radians((357.528 + 0.9856003*n_days) % 360)
    lam = L + np.radians(1.915*np.sin(g) + 0.020*np.sin(2*g))
    return np.stack([np.cos(lam), np.sin(lam), np.zeros_like(lam)], axis=1)


def _circle_overlap_fraction(r1, r2, d):
    """
    Real circle-circle intersection area, vectorized, returning the
    fraction of the SECOND circle's area (r2 — the Sun's apparent disk)
    still visible after the first circle (r1 — the occluder's apparent
    disk) subtracts its overlapping area. All three inputs are angular
    radii/separation in the same units (radians here).
    """
    r1 = np.broadcast_to(r1, np.broadcast(r1, r2, d).shape).astype(float)
    r2 = np.broadcast_to(r2, np.broadcast(r1, r2, d).shape).astype(float)
    d = np.broadcast_to(d, np.broadcast(r1, r2, d).shape).astype(float)

    illum = np.ones_like(d)

    no_overlap = d >= (r1 + r2)
    illum = np.where(no_overlap, 1.0, illum)

    # One circle entirely inside the other
    contained = d <= np.abs(r1 - r2)
    occluder_bigger = r1 >= r2
    # occluder fully covers the Sun's disk -> total eclipse
    illum = np.where(contained & occluder_bigger, 0.0, illum)
    # occluder is smaller than the Sun's disk and sits fully inside it
    # (annular eclipse case) -> a fixed bright ring always remains
    with np.errstate(divide="ignore", invalid="ignore"):
        annular_illum = 1.0 - np.clip((r1 / np.where(r2 > 0, r2, 1)) ** 2, 0, 1)
    illum = np.where(contained & ~occluder_bigger, annular_illum, illum)

    # Genuine partial overlap — real two-circle intersection area formula
    partial = ~no_overlap & ~contained
    with np.errstate(divide="ignore", invalid="ignore"):
        d_p = np.where(partial, d, 1.0)
        r1_p = np.where(partial, r1, 1.0)
        r2_p = np.where(partial, r2, 1.0)
        arg1 = np.clip((d_p**2 + r1_p**2 - r2_p**2) / (2*d_p*r1_p), -1, 1)
        arg2 = np.clip((d_p**2 + r2_p**2 - r1_p**2) / (2*d_p*r2_p), -1, 1)
        term = (-d_p + r1_p + r2_p) * (d_p + r1_p - r2_p) * (d_p - r1_p + r2_p) * (d_p + r1_p + r2_p)
        area = (r1_p**2 * np.arccos(arg1) + r2_p**2 * np.arccos(arg2)
               - 0.5 * np.sqrt(np.clip(term, 0, None)))
        sun_area = np.pi * r2_p**2
        partial_illum = 1.0 - np.clip(area / np.where(sun_area > 0, sun_area, 1), 0, 1)
    illum = np.where(partial, partial_illum, illum)

    return np.clip(illum, 0.0, 1.0)


def illumination_fraction(r_eval_km, sun_hat, R_body_km, R_sun_km=R_SUN_KM, D_km=AU_KM):
    """
    Fraction of the Sun's apparent disk still visible from a point whose
    position relative to the occluding body is `r_eval_km`, given the
    Sun's direction `sun_hat` (unit vector, same frame as r_eval_km).

    r_eval_km : (..., 3) position of the illuminated point relative to
                the occluder (e.g. Moon relative to Earth for a lunar
                eclipse; a Moon-surface vertex relative to Earth; Earth
                relative to the Moon for a solar eclipse)
    sun_hat   : (..., 3) unit vector toward the Sun, same leading shape
    R_body_km : occluder's real radius (km)
    R_sun_km  : Sun's real radius (km)
    D_km      : distance to the Sun (km) — used only for the Sun's own
                angular radius, since D_km >> |r_eval_km| always holds
                here (a few hundred thousand km vs ~150 million km)
    """
    r_eval_km = np.asarray(r_eval_km, dtype=float)
    sun_hat = np.asarray(sun_hat, dtype=float)
    dist = np.linalg.norm(r_eval_km, axis=-1)
    dist_safe = np.where(dist > 0, dist, 1.0)
    occ_hat = r_eval_km / dist_safe[..., None]

    # Angular radius of the occluder as seen from the evaluation point,
    # and of the Sun as seen from (effectively) the same point.
    ang_occ = np.arcsin(np.clip(R_body_km / dist_safe, -1, 1))
    ang_sun = np.arcsin(np.clip(R_sun_km / D_km, -1, 1))

    # Angular separation between "direction to occluder" (-occ_hat, since
    # occ_hat points evaluation-point -> occluder... actually occ_hat as
    # defined above already points FROM the occluder TO the evaluation
    # point, so the direction from the evaluation point back to the
    # occluder is -occ_hat) and "direction to the Sun" (sun_hat).
    cos_sep = np.clip(np.sum((-occ_hat) * sun_hat, axis=-1), -1, 1)
    sep = np.arccos(cos_sep)

    return _circle_overlap_fraction(ang_occ, ang_sun, sep)