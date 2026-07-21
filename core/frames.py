"""
core/frames.py
──────────────
Reference-frame transforms for trajectories and vectors.

Supported frames
----------------
ECI   Earth-Centred Inertial  (J2000 / GCRF) — the native SSAPy frame
ECF   Earth-Centred Fixed     (rotates with Earth at OMEGA_E)
LVLH  Local Vertical / Local Horizontal  (RSW: R along r, W = h-hat, S = W×R)
RTN   Radial–Transverse–Normal  (same as LVLH / RSW — alias)
NTW   Normal–Transverse–W       (T along v, N in-plane ⊥ v, W = orbit normal)

Usage
-----
from core.frames import FrameTransform, Frame

tf = FrameTransform(Frame.LVLH)
r_lvlh = tf.transform_points(r_eci, v_eci)

# Or transform an entire Trajectory
traj_lvlh = tf.transform_trajectory(traj, ref_state)
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import numpy as np

OMEGA_E = 7.292_115_0e-5   # rad/s Earth rotation rate


# ── Frame enum ───────────────────────────────────────────────────────────────
class Frame(str, Enum):
    ECI  = "ECI"
    ECF  = "ECF"
    LVLH = "LVLH"
    RTN  = "RTN"   # alias for LVLH
    NTW  = "NTW"

    @property
    def label(self) -> str:
        return {
            "ECI":  "Earth-Centred Inertial (J2000)",
            "ECF":  "Earth-Centred Fixed (rotating)",
            "LVLH": "Local Vertical / Local Horizontal",
            "RTN":  "Radial–Transverse–Normal",
            "NTW":  "Normal–Transverse–W (velocity-aligned)",
        }[self.value]


# ── helper: unit vector ───────────────────────────────────────────────────────
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v


# ── rotation matrices ─────────────────────────────────────────────────────────
def eci_to_ecf_matrix(t_gps: float) -> np.ndarray:
    """
    3×3 rotation matrix ECI → ECF at GPS time t_gps.
    Uses a simple Greenwich sidereal angle approximation;
    if astropy is available uses a more precise GMST.
    """
    try:
        from astropy.time import Time
        t_ast = Time(t_gps, format="gps")
        theta  = float(t_ast.sidereal_time("mean", "greenwich").rad)
    except Exception:
        # fallback: approximate GST from GPS epoch
        gps_epoch_jd = 2_444_244.5
        jd = gps_epoch_jd + t_gps / 86_400.0
        T  = (jd - 2_451_545.0) / 36_525.0
        theta = (280.46061837 + 360.98564736629*(jd - 2_451_545.0)) % 360
        theta  = np.radians(theta)

    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct,  st, 0],
                     [-st, ct, 0],
                     [0,   0,  1]])


def lvlh_matrix(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    3×3 matrix whose rows are [R_hat, S_hat, W_hat] in ECI.
    R = r/|r|  (radial),  W = h/|h|  (orbit normal),  S = W × R
    Transforms ECI → LVLH/RSW.
    """
    R_hat = _unit(r)
    W_hat = _unit(np.cross(r, v))
    S_hat = np.cross(W_hat, R_hat)
    return np.array([R_hat, S_hat, W_hat])


def ntw_matrix(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    3×3 matrix whose rows are [T_hat, N_hat, W_hat] in ECI.
    T = v/|v|  (tangential / along-track),
    W = h/|h|  (orbit normal),
    N = W × T  (in-plane, cross-track)
    Transforms ECI → NTW.
    """
    T_hat = _unit(v)
    W_hat = _unit(np.cross(r, v))
    N_hat = np.cross(W_hat, T_hat)
    return np.array([T_hat, N_hat, W_hat])


def ntw_axes(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_hat, N_hat, W_hat) as separate ECI unit vectors."""
    T = _unit(v)
    W = _unit(np.cross(r, v))
    N = np.cross(W, T)
    return T, N, W


def lvlh_axes(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R_hat, S_hat, W_hat) as separate ECI unit vectors."""
    R = _unit(r)
    W = _unit(np.cross(r, v))
    S = np.cross(W, R)
    return R, S, W


# ── FrameTransform ────────────────────────────────────────────────────────────
class FrameTransform:
    """
    Transform position arrays (and optionally velocity arrays) between frames.

    All inputs/outputs in km (positions) or km/s (velocities).

    Parameters
    ----------
    frame : Frame
        Target frame.
    t_gps : float | None
        GPS epoch for ECF rotation.  Required if frame == ECF.
    """

    def __init__(self, frame: Frame | str, t_gps: float | None = None):
        self.frame = Frame(frame)
        self.t_gps = t_gps

    # ── single-point transform ────────────────────────────────────────────────
    def transform_point(
        self,
        r_eci : np.ndarray,
        v_eci : np.ndarray | None = None,
        t_gps : float | None = None,
    ) -> np.ndarray:
        """
        Transform a single ECI position into the target frame.

        r_eci : (3,) km
        v_eci : (3,) km/s  — required for LVLH / RTN / NTW
        t_gps : GPS seconds — required for ECF (overrides self.t_gps)
        """
        r = np.asarray(r_eci, dtype=float)
        v = np.asarray(v_eci, dtype=float) if v_eci is not None else None

        if self.frame == Frame.ECI:
            return r.copy()

        if self.frame == Frame.ECF:
            tg = t_gps or self.t_gps or 0.0
            return eci_to_ecf_matrix(tg) @ r

        if self.frame in (Frame.LVLH, Frame.RTN):
            if v is None:
                raise ValueError("velocity required for LVLH/RTN transform")
            return lvlh_matrix(r, v) @ r

        if self.frame == Frame.NTW:
            if v is None:
                raise ValueError("velocity required for NTW transform")
            return ntw_matrix(r, v) @ r

        raise ValueError(f"Unknown frame: {self.frame}")

    def transform_vector(
        self,
        vec   : np.ndarray,
        r_eci : np.ndarray,
        v_eci : np.ndarray | None = None,
        t_gps : float | None = None,
    ) -> np.ndarray:
        """
        Rotate an arbitrary ECI vector into the target frame.
        (Same rotation as transform_point but without the implied meaning of position.)
        """
        return self.transform_point(vec, v_eci=v_eci, t_gps=t_gps) if self.frame == Frame.ECF \
               else self._rotation_matrix(r_eci, v_eci, t_gps) @ vec

    def _rotation_matrix(self, r, v, t_gps=None) -> np.ndarray:
        if self.frame == Frame.ECI:
            return np.eye(3)
        if self.frame == Frame.ECF:
            return eci_to_ecf_matrix(t_gps or self.t_gps or 0.0)
        if self.frame in (Frame.LVLH, Frame.RTN):
            return lvlh_matrix(r, v)
        if self.frame == Frame.NTW:
            return ntw_matrix(r, v)
        return np.eye(3)

    # ── trajectory transform ──────────────────────────────────────────────────
    def transform_trajectory(
        self,
        r_eci : np.ndarray,
        v_eci : np.ndarray,
        t_gps : np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform an (N, 3) ECI position array into the target frame.

        Parameters
        ----------
        r_eci : (N, 3) km
        v_eci : (N, 3) km/s
        t_gps : (N,) GPS seconds — required for ECF

        Returns
        -------
        (N, 3) positions in target frame, km
        """
        r = np.asarray(r_eci, dtype=float)
        v = np.asarray(v_eci, dtype=float)
        N = r.shape[0]
        out = np.empty_like(r)

        if self.frame == Frame.ECI:
            return r.copy()

        if self.frame == Frame.ECF:
            if t_gps is None:
                raise ValueError("t_gps required for ECF transform")
            for i in range(N):
                out[i] = eci_to_ecf_matrix(t_gps[i]) @ r[i]
            return out

        if self.frame in (Frame.LVLH, Frame.RTN):
            for i in range(N):
                M = lvlh_matrix(r[i], v[i])
                out[i] = M @ r[i]
            return out

        if self.frame == Frame.NTW:
            for i in range(N):
                M = ntw_matrix(r[i], v[i])
                out[i] = M @ r[i]
            return out

        raise ValueError(f"Unknown frame: {self.frame}")

    # ── convenience: relative trajectory (centred on first point) ─────────────
    def relative_trajectory(
        self,
        r_eci : np.ndarray,
        v_eci : np.ndarray,
        t_gps : np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform trajectory and subtract first point — useful for
        LVLH plots centred on the reference spacecraft.
        """
        out = self.transform_trajectory(r_eci, v_eci, t_gps)
        return out - out[0]


# ── Convenience functions ─────────────────────────────────────────────────────
def eci_to_lon_lat(r_eci_km: np.ndarray, t_gps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (N,3) ECI positions → (lon_deg, lat_deg) sub-satellite point.
    Uses ECF rotation for longitude, then spherical geometry.
    """
    r = np.asarray(r_eci_km)
    t = np.asarray(t_gps)
    lons, lats = [], []
    for i in range(len(r)):
        r_ecf = eci_to_ecf_matrix(t[i]) @ r[i]
        x, y, z = r_ecf
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arcsin(z / np.linalg.norm(r_ecf)))
        lons.append(lon); lats.append(lat)
    return np.array(lons), np.array(lats)
