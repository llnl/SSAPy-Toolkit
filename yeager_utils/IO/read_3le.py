import re
import numpy as np
import pandas as pd
from ..constants import EARTH_MU, EARTH_RADIUS  # m^3/s^2, m

def read_3le(data_file, makedf=True, verbose=False):
    """
    Parse a 3-line element (0/1/2) text file into a dict of lists or a DataFrame,
    and compute classical Keplerian elements at the TLE epoch.

    All distances are meters (m), velocities m/s, angles rad/deg, times seconds.

    Added fields (SI):
      - mean_motion_rev_day, n_rad_s, period_s
      - a_m, p_m, rp_m, ra_m, hp_m, ha_m
      - inc_rad, raan_rad, argp_rad, M_rad
      - E_rad/deg, nu_rad/deg, u_rad/deg
      - r_mag_m, v_mag_m_s
      - r_eci_m_[x,y,z], v_eci_m_s_[x,y,z] (two-body, at epoch)
    """
    DAY_S = 86400.0
    TWOPI = 2.0 * np.pi
    DEG2RAD = np.pi / 180.0
    RAD2DEG = 180.0 / np.pi

    def split_line(line):
        payload = re.findall(r'"[^"]+"|\'[^\']+\'|\S+', line)
        return [s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'" else s for s in payload]

    def _days_in_year(y: int) -> float:
        return 366.0 if ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)) else 365.0

    def _century_year(two_digit: str) -> int:
        yy = int(two_digit)
        return 1900 + yy if yy >= 57 else 2000 + yy

    def vprint(message):
        if verbose:
            print(message)

    def parse_ecc(tok: str) -> float:
        # TLE eccentricity: '0006703' -> 0.0006703; allow normal floats too.
        t = str(tok).strip()
        if t == "" or t.lower() == "nan":
            return np.nan
        if "." in t:
            return float(t)
        if re.fullmatch(r"\d+", t):
            return float("0." + t)
        raise ValueError(f"Bad eccentricity token: {tok!r}")

    with open(data_file, "r", encoding="utf-8", errors="replace") as f:
        def split_mantassa(text):
            t = text.strip()
            if not t or t.strip('0+-') == '':
                return 0.0
            if len(t) < 3 or t[-2] not in '+-' or not t[-1].isdigit():
                raise ValueError(f"Bad TLE exp field: {text!r}")
            sign = '-' if t[0] == '-' else ''
            mant = t[1:-2] if t[0] in '+-' else t[:-2]
            return float(f"{sign}0.{mant}e{t[-2:]}")

        def looks_like_intl(tok: str) -> bool:
            return bool(re.fullmatch(r"\d{2}\d{3}[A-Za-z]{1,3}", tok))

        def parse_epoch_fields(epoch_token: str):
            t = epoch_token.replace(" ", "")
            yy = int(t[:2])
            epoch_year = (1900 + yy) if yy >= 57 else (2000 + yy)
            epoch_day = float(t[2:])  # DDD.dddddd
            return epoch_year, epoch_day

        cols = [
            "name",
            "satellite#",
            "classification",
            "intl_designator",
            "launch_year",
            "launch_number",
            "launch_piece",
            "elset_epoch",
            "epoch_year",
            "epoch_day",
            "decimal_year",
            "ndot",
            "nddot",
            "drag",
            "elset_type",
            "elset#",
            "catalog number",
            "inc_deg",
            "raan_deg",
            "ecc",
            "pa_deg",
            "mean_anomaly_deg",
            "mean_motion_deg",          # NOTE: this is rev/day in TLE
            "mean_motion_rev_day_deg",  # kept for compatibility
            "epoch_rev",
            "check_sum",
        ]

        data = {key: [] for key in cols}

        for raw in f:
            line = raw.rstrip("\r\n")
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")
            if line == "":
                continue

            key = line[0]
            rest = line[1:]
            splits = split_line(rest)
            vprint(splits)

            if key == "0":
                data["name"].append(splits)

            elif key == "1":
                if not splits:
                    continue
                data["satellite#"].append(splits[0][:-1])
                data["classification"].append(splits[0][-1])

                has_intl = len(splits) > 1 and looks_like_intl(splits[1])
                i = 0 if has_intl else -1

                if has_intl:
                    data["intl_designator"].append(splits[1 + i])
                    data["launch_year"].append(int(splits[1 + i][:2]))
                    data["launch_number"].append(int(splits[1 + i][2:5]))
                    data["launch_piece"].append(splits[1 + i][5:])
                else:
                    data["intl_designator"].append(None)
                    data["launch_year"].append(None)
                    data["launch_number"].append(None)
                    data["launch_piece"].append(None)

                data["elset_epoch"].append(splits[2 + i])
                ey, ed = parse_epoch_fields(splits[2 + i])
                data["epoch_year"].append(ey)
                data["epoch_day"].append(ed)

                denom = _days_in_year(int(ey))
                dec_year = float(ey) + (float(ed) - 1.0) / float(denom)
                data["decimal_year"].append(dec_year)

                data["ndot"].append(float(splits[3 + i]))
                data["nddot"].append(split_mantassa(splits[4 + i]))
                data["drag"].append(split_mantassa(splits[5 + i]))
                data["elset_type"].append(int(splits[6 + i]))
                data["elset#"].append(int(splits[7 + i]))

            elif key == "2":
                if len(splits) < 6:
                    continue
                data["inc_deg"].append(float(splits[1]))
                data["raan_deg"].append(float(splits[2]))
                data["ecc"].append(parse_ecc(splits[3]))
                data["pa_deg"].append(float(splits[4]))
                data["mean_anomaly_deg"].append(float(splits[5]))
                if len(splits) >= 7 and len(splits[6]) > 11:
                    s6 = splits[6]
                    data["mean_motion_deg"].append(float(s6[:11]))  # rev/day
                    data["epoch_rev"].append(int(s6[11:15]))
                    data["check_sum"].append(int(s6[15]))
                else:
                    data["mean_motion_deg"].append(np.nan)
                    data["epoch_rev"].append(np.nan)
                    data["check_sum"].append(np.nan)
            else:
                pass

    if not makedf:
        return data

    # Align rows by line "1"
    base_n = len(data["satellite#"])

    def _get(k, i, default=np.nan):
        lst = data.get(k, [])
        return lst[i] if i < len(lst) else default

    rows = []
    for i in range(base_n):
        row = {k: _get(k, i) for k in cols}
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)

    # -------------------- Orbital elements and derived quantities (SI) --------------------
    # Mean motion: TLE "mean_motion_deg" is rev/day
    df["mean_motion_rev_day"] = pd.to_numeric(df["mean_motion_deg"], errors="coerce")
    n_rad_s = df["mean_motion_rev_day"] * (TWOPI / DAY_S)
    df["n_rad_s"] = n_rad_s
    df["period_s"] = TWOPI / n_rad_s

    # Semi-major axis (m) via Kepler
    df["a_m"] = (EARTH_MU / (n_rad_s ** 2)) ** (1.0 / 3.0)

    # Eccentricity
    e = pd.to_numeric(df["ecc"], errors="coerce")

    # Semi-latus rectum and apsides (m)
    df["p_m"]  = df["a_m"] * (1.0 - e ** 2)
    df["rp_m"] = df["a_m"] * (1.0 - e)
    df["ra_m"] = df["a_m"] * (1.0 + e)
    df["hp_m"] = df["rp_m"] - EARTH_RADIUS
    df["ha_m"] = df["ra_m"] - EARTH_RADIUS

    # Angles in radians
    df["inc_rad"]  = pd.to_numeric(df["inc_deg"], errors="coerce") * DEG2RAD
    df["raan_rad"] = pd.to_numeric(df["raan_deg"], errors="coerce") * DEG2RAD
    df["argp_rad"] = pd.to_numeric(df["pa_deg"], errors="coerce") * DEG2RAD
    df["M_rad"]    = pd.to_numeric(df["mean_anomaly_deg"], errors="coerce") * DEG2RAD

    # Solve Kepler's equation for E and true anomaly nu
    M = df["M_rad"].to_numpy(dtype=float)
    e_arr = e.to_numpy(dtype=float)

    def solve_kepler_E(M_rad, ecc, tol=1e-12, max_iter=50):
        M0 = np.mod(M_rad, TWOPI)
        E = np.where(ecc < 0.8, M0, np.pi * np.ones_like(M0))
        mask = np.isfinite(M0) & np.isfinite(ecc)
        if not np.any(mask):
            return np.full_like(M0, np.nan)
        E_work = E.copy()
        for _ in range(max_iter):
            f = E_work - ecc * np.sin(E_work) - M0
            fp = 1.0 - ecc * np.cos(E_work)
            delta = -f / fp
            E_work = E_work + delta
            if np.all(np.abs(delta[mask]) < tol):
                break
        E[mask] = E_work[mask]
        E[~mask] = np.nan
        return E

    E = solve_kepler_E(M, e_arr)
    df["E_rad"] = E
    df["E_deg"] = E * RAD2DEG

    denom = (1.0 - e_arr * np.cos(E))
    sinv = np.sqrt(1.0 - e_arr**2) * np.sin(E) / denom
    cosv = (np.cos(E) - e_arr) / denom
    nu = np.arctan2(sinv, cosv)
    df["nu_rad"] = nu
    df["nu_deg"] = nu * RAD2DEG

    # Argument of latitude
    df["u_rad"] = df["argp_rad"] + df["nu_rad"]
    df["u_deg"] = df["u_rad"] * RAD2DEG

    # Magnitudes at epoch (two-body)
    r_mag = df["p_m"] / (1.0 + e * np.cos(nu))                       # m
    v_mag = np.sqrt(EARTH_MU * (2.0 / r_mag - 1.0 / df["a_m"]))      # m/s
    df["r_mag_m"] = r_mag
    df["v_mag_m_s"] = v_mag

    # Perifocal state (m, m/s)
    cosv_arr, sinv_arr = np.cos(nu), np.sin(nu)
    p = df["p_m"].to_numpy(dtype=float)
    r_pf_x = p * cosv_arr / (1.0 + e_arr * cosv_arr)
    r_pf_y = p * sinv_arr / (1.0 + e_arr * cosv_arr)
    r_pf_z = np.zeros_like(r_pf_x)
    sqrt_mu_p = np.sqrt(EARTH_MU / p)
    v_pf_x = -sqrt_mu_p * sinv_arr
    v_pf_y =  sqrt_mu_p * (e_arr + cosv_arr)
    v_pf_z = np.zeros_like(v_pf_x)

    # Rotation 3-1-3 to ECI
    O, i, w = df["raan_rad"].to_numpy(), df["inc_rad"].to_numpy(), df["argp_rad"].to_numpy()
    cO, sO = np.cos(O), np.sin(O)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(w), np.sin(w)

    R11 = cO*cw - sO*sw*ci
    R12 = -cO*sw - sO*cw*ci
    R13 = sO*si
    R21 = sO*cw + cO*sw*ci
    R22 = -sO*sw + cO*cw*ci
    R23 = -cO*si
    R31 = sw*si
    R32 = cw*si
    R33 = ci

    df["r_eci_m_x"]   = R11*r_pf_x + R12*r_pf_y + R13*r_pf_z
    df["r_eci_m_y"]   = R21*r_pf_x + R22*r_pf_y + R23*r_pf_z
    df["r_eci_m_z"]   = R31*r_pf_x + R32*r_pf_y + R33*r_pf_z

    df["v_eci_m_s_x"] = R11*v_pf_x + R12*v_pf_y + R13*v_pf_z
    df["v_eci_m_s_y"] = R21*v_pf_x + R22*v_pf_y + R23*v_pf_z
    df["v_eci_m_s_z"] = R31*v_pf_x + R32*v_pf_y + R33*v_pf_z

    return df
