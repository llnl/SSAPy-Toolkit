"""
eclipse_space_view_plotly.py — Consolidated eclipse module (search + matplotlib panel + interactive 3D space view)
======================================================================================================================
This used to be two files (eclipse_demo.py for the search + 2D matplotlib
panel, this file for the interactive 3D Plotly "view from space"). Merged
into one so there's a single "main" eclipse module — same functions, same
behavior, just one import instead of two. eclipse_demo.py's content lives
in the first half of this file (search, moon_color/brightness helpers, the
matplotlib figure); the original Plotly space-view code follows it.

Usage
-----
    from eclipse_space_view_plotly import find_and_plot_eclipse, plot_space_view_plotly, plot_space_view_animated

    fig, stats = find_and_plot_eclipse(mode="lunar", save_path="lunar.png")
    plot_space_view_plotly(mode="lunar", save_path="lunar_space.html")
    plot_space_view_animated(mode="solar", save_path="solar_space_animated.html")
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

try:
    from .globe_orbit_daynight_plotly import _earth_mesh, _sun_sphere_traces, _earth_atmosphere_trace, RE_KM
    from .moon_render import moon_mesh_plotly
    from .eclipse_brightness_plot import propagate_eci, sun_direction_eci, illumination_fraction, R_SUN_KM, AU_KM
    from .eclipse_appearance_strip import render_lunar_panel, render_solar_panel
except ImportError:
    from globe_orbit_daynight_plotly import _earth_mesh, _sun_sphere_traces, _earth_atmosphere_trace, RE_KM
    from moon_render import moon_mesh_plotly
    from eclipse_brightness_plot import propagate_eci, sun_direction_eci, illumination_fraction, R_SUN_KM, AU_KM
    from eclipse_appearance_strip import render_lunar_panel, render_solar_panel

D_MOON_A_KM = 384_748.0
D_MOON_E = 0.0549
D_MOON_INC_DEG = 5.145
R_MOON_KM = 1_737.4




def moon_color(illum_frac):
    """Lunar-eclipse 'Blood Moon' colour ramp — grey dims toward deep red.
    Used for the flat 2D appearance-strip circles (no texture to preserve
    there, so a colour blend is fine)."""
    grey = np.array([0.82, 0.82, 0.85])
    red_totality = np.array([0.45, 0.10, 0.06])
    red_mix = np.clip((0.35 - illum_frac) / 0.35, 0, 1) ** 1.5
    base = grey * np.clip(0.15 + 0.85*illum_frac, 0.15, 1.0)
    return base * (1 - red_mix) + red_totality * red_mix


def moon_brightness(illum_frac):
    """
    Real 'darker due to lack of sunlight' model for the TEXTURED 3D Moon —
    brightness only (floor 0.12, never literally black), so craters/mare
    stay clearly visible even in deep shadow. This is what the previous
    colour-replace version (moon_color, above) got wrong when applied to
    a textured sphere: multiplying every point toward the same dark red
    made the crater contrast nearly disappear.
    """
    floor = 0.12
    return np.clip(floor + (1 - floor) * illum_frac, floor, 1.0)


def moon_red_bias(illum_frac):
    """Subtle warm/red multiplicative tint during deep shadow (same real
    cause as before — Earth-atmosphere-scattered sunlight reaching the
    Moon even in geometric shadow) — weak enough to not wash out texture,
    since it multiplies the already-textured, already-darkened base."""
    red_mix = np.clip((0.35 - illum_frac) / 0.35, 0, 1) ** 1.5
    warm = np.array([1.15, 0.55, 0.42])
    return (1 - red_mix) + warm * red_mix


def _sun_sphere_mpl(ax, center, radius, seed=11):
    n = 36
    su, sv = np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n//2)
    SU, SV = np.meshgrid(su, sv)
    nx, ny, nz = np.cos(SU)*np.sin(SV), np.sin(SU)*np.sin(SV), np.cos(SV)
    sx, sy, sz = center[0]+radius*nx, center[1]+radius*ny, center[2]+radius*nz

    rng = np.random.default_rng(seed)
    granulation = np.zeros_like(SU)
    for _ in range(40):
        c = rng.normal(size=3); c /= np.linalg.norm(c)
        dot = nx*c[0] + ny*c[1] + nz*c[2]
        spread = rng.uniform(0.85, 0.97)
        granulation += np.clip((dot - spread)/(1-spread), 0, 1) * rng.uniform(-0.25, 0.25)

    ref = np.array([0.4, 0.4, 0.82]); ref /= np.linalg.norm(ref)
    limb = np.clip(nx*ref[0]+ny*ref[1]+nz*ref[2], 0, 1) ** 0.35
    brightness = np.clip(0.55 + 0.45*limb + granulation, 0.15, 1.0)

    stops = np.array([0.15, 0.5, 0.75, 1.0])
    rgb_stops = np.array([[0.478,0.180,0.0],[0.851,0.333,0.039],[1.0,0.647,0.0],[1.0,0.984,0.918]])
    colors = np.empty(brightness.shape + (3,))
    for ch in range(3):
        colors[..., ch] = np.interp(brightness, stops, rgb_stops[:, ch])

    ax.plot_surface(sx, sy, sz, facecolors=colors, linewidth=0, shade=False, zorder=8)
    for gs, ga in [(1.4, 0.15), (1.9, 0.06)]:
        gx, gy, gz = center[0]+radius*gs*nx, center[1]+radius*gs*ny, center[2]+radius*gs*nz
        ax.plot_surface(gx, gy, gz, color="#FFD700", alpha=ga, linewidth=0, shade=False, zorder=7)


def _earth_sphere_mpl(ax, center, radius, sun_hat, seed=7):
    n_lat, n_lon = 40, 80
    lat = np.linspace(90, -90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    Lon, Lat = np.meshgrid(lon, lat)
    latr, lonr = np.radians(Lat), np.radians(Lon)
    nx, ny, nz = np.cos(latr)*np.cos(lonr), np.cos(latr)*np.sin(lonr), np.sin(latr)
    ex, ey, ez = center[0]+radius*nx, center[1]+radius*ny, center[2]+radius*nz

    try:
        from global_land_mask import globe
        lon_q = np.where(Lon >= 180, Lon-360, Lon)
        land = globe.is_land(np.clip(Lat, -89.999, 89.999), lon_q)
    except Exception:
        rng = np.random.default_rng(seed)
        field = np.zeros_like(Lat)
        for _ in range(14):
            clat, clon = rng.uniform(-60,60), rng.uniform(-180,180)
            spread = rng.uniform(15, 35)
            d = np.sqrt((Lat-clat)**2 + ((Lon-clon+180)%360-180)**2)
            field += np.exp(-(d**2)/(2*spread**2))
        land = field > 0.35

    ocean, landc = np.array([0.08,0.22,0.50]), np.array([0.20,0.50,0.18])
    rgb = np.where(land[...,None], landc, ocean)
    dot = nx*sun_hat[0] + ny*sun_hat[1] + nz*sun_hat[2]
    lit = np.clip(dot, 0, 1) ** 0.6
    night_tint = np.array([0.03,0.05,0.10])
    rgb_shaded = np.clip(rgb*lit[...,None] + night_tint*(1-lit[...,None]), 0, 1)
    ax.plot_surface(ex, ey, ez, facecolors=rgb_shaded, linewidth=0, shade=False, zorder=8)


def _moon_sphere_mpl(ax, center, radius, tint_rgb, seed=3):
    n = 30
    mu, mv = np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n//2)
    muu, mvv = np.meshgrid(mu, mv)
    mx, my, mz = center[0]+radius*np.cos(muu)*np.sin(mvv), \
                 center[1]+radius*np.sin(muu)*np.sin(mvv), \
                 center[2]+radius*np.cos(mvv)
    rng = np.random.default_rng(seed)
    mlat, mlon = 90-np.degrees(mvv), np.degrees(muu)
    albedo = np.full_like(mlat, 0.85)
    for _ in range(6):
        clat, clon = rng.uniform(-40,55), rng.uniform(-70,70)
        spread = rng.uniform(12, 28)
        d = np.sqrt((mlat-clat)**2 + ((mlon-clon+180)%360-180)**2)
        albedo -= 0.30*np.exp(-(d**2)/(2*spread**2))
    for _ in range(60):
        clat, clon = rng.uniform(-85,85), rng.uniform(-180,180)
        radius_c = rng.uniform(2, 8)
        d = np.sqrt((mlat-clat)**2 + ((mlon-clon+180)%360-180)**2)
        albedo -= 0.35*np.clip(1 - d/radius_c, 0, 1)**2
    albedo = np.clip(albedo, 0.35, 1.0)
    rgb = np.clip(np.array(tint_rgb)[None,None,:] * albedo[...,None], 0, 1)
    ax.plot_surface(mx, my, mz, facecolors=rgb, linewidth=0, shade=False, zorder=9)


def find_and_plot_eclipse(mode="lunar", save_path=None, search_days=None, verbose=True):
    assert mode in ("lunar", "solar")
    # Solar eclipses need a much tighter alignment (Moon's angular radius
    # ~0.25-0.55 deg vs Earth's ~1-2 deg as seen from the Moon) — in this
    # simplified fixed-node model (no real ~18.6yr nodal precession), a
    # single year doesn't reliably produce a close-enough approach; the
    # slight mismatch between the lunar month and the calendar year means
    # different years land at different phases, so searching several years
    # finds a real one (confirmed: 1 year found only a 0.70 deg near-miss,
    # 6 years found a genuine 0.35 deg alignment).
    if search_days is None:
        search_days = 365.0 if mode == "lunar" else 365.25 * 6
    lunar_period_days = 27.32
    n_orbits_year = search_days / lunar_period_days

    # Solar eclipse alignment windows are much narrower in time than lunar
    # ones (Moon's angular radius ~0.25-0.55 deg vs Earth's ~1-2 deg as
    # seen from the Moon), so the coarse search needs much denser sampling
    # or it can skip the entire event — confirmed by testing: at 60
    # samples/orbit the search found min_illum=1.0 all year (nothing),
    # even though a genuine 0.71 deg alignment was present when checked
    # directly at finer resolution.
    _coarse_density = 60 if mode == "lunar" else 1500
    t_s, r_moon, T_s = propagate_eci(
        a_km=D_MOON_A_KM, e=D_MOON_E, inc_deg=D_MOON_INC_DEG,
        raan_deg=0.0, argp_deg=0.0, nu0_deg=0.0,
        n_orbits=n_orbits_year, n_steps=int(n_orbits_year*_coarse_density),
    )
    sun_hat = sun_direction_eci(t_s)

    if mode == "lunar":
        # Earth's shadow on the Moon: occluder=Earth, evaluated at the
        # Moon's position relative to Earth (r_moon, as propagated above).
        r_eval, R_occ = r_moon, RE_KM
    else:
        # The Moon's shadow on Earth: occluder=Moon, evaluated at Earth's
        # position relative to the Moon = -r_moon.
        r_eval, R_occ = -r_moon, R_MOON_KM

    illum_coarse = illumination_fraction(r_eval, sun_hat, R_body_km=R_occ,
                                          R_sun_km=R_SUN_KM, D_km=AU_KM)
    best_idx = int(np.argmin(illum_coarse))
    best_t = t_s[best_idx]
    if verbose:
        print(f"[{mode}] Coarse search over {search_days:.0f} days: "
              f"deepest illumination minimum = {illum_coarse[best_idx]:.4f} "
              f"at t={best_t/86400:.1f} days")

    window_days = 2.0
    # Propagate a short, dense window directly centred on best_t instead
    # of re-doing the whole (possibly multi-year) span at ultra-fine
    # resolution and slicing — for the 6-year solar search that would mean
    # tens of millions of samples, far too slow. Compute the true anomaly
    # at (best_t - window_days) analytically and use that as nu0 for a
    # short fine propagation instead.
    MU_EARTH_KM3S2 = 398_600.4418
    n_rad_s = np.sqrt(MU_EARTH_KM3S2 / D_MOON_A_KM**3)
    t_window_start = best_t - window_days*86400
    M_start = (n_rad_s * t_window_start) % (2*np.pi)
    E_start = M_start.copy() if hasattr(M_start, "copy") else M_start
    for _ in range(60):
        dE = (M_start - E_start + D_MOON_E*np.sin(E_start)) / (1 - D_MOON_E*np.cos(E_start))
        E_start += dE
    nu_start = 2*np.arctan2(np.sqrt(1+D_MOON_E)*np.sin(E_start/2), np.sqrt(1-D_MOON_E)*np.cos(E_start/2))
    n_orbits_window = (2*window_days) / lunar_period_days

    t_fine, r_fine, _ = propagate_eci(
        a_km=D_MOON_A_KM, e=D_MOON_E, inc_deg=D_MOON_INC_DEG,
        raan_deg=0.0, argp_deg=0.0, nu0_deg=np.degrees(nu_start),
        n_orbits=n_orbits_window, n_steps=4000,
    )
    t_fine = t_fine + t_window_start   # shift back to the real timeline
    sun_fine = sun_direction_eci(t_fine)
    r_eval_fine = r_fine if mode == "lunar" else -r_fine
    illum_fine = illumination_fraction(r_eval_fine, sun_fine, R_body_km=R_occ,
                                        R_sun_km=R_SUN_KM, D_km=AU_KM)
    fine_mask = np.abs(t_fine - best_t) < window_days*86400
    t_win = t_fine[fine_mask]
    r_win = r_fine[fine_mask]          # Moon position relative to Earth, always
    sun_win = sun_fine[fine_mask]
    illum_win = illum_fine[fine_mask]

    t_hr = (t_win - best_t) / 3600.0
    mid = int(np.argmin(illum_win))

    # Actual angular separation at peak — the real geometric quantity that
    # determines whether an eclipse happens at all and how deep it is
    # (this is what we were computing ad-hoc during debugging earlier;
    # now it's a first-class, displayed result instead of a side check).
    r_eval_mid = r_win[mid] if mode == "lunar" else -r_win[mid]
    occ_hat_mid = r_eval_mid / np.linalg.norm(r_eval_mid)
    cos_sep_mid = np.dot(-occ_hat_mid, sun_win[mid])
    sep_deg_mid = np.degrees(np.arccos(np.clip(cos_sep_mid, -1, 1)))

    frac_umbra = np.mean(illum_win < 0.02) * 100
    dt_mean = np.mean(np.diff(t_win)) if len(t_win) > 1 else 0
    dur_umbra_hr = np.sum(illum_win < 0.02) * dt_mean / 3600.0
    dur_total_hr = np.sum(illum_win < 0.999) * dt_mean / 3600.0

    if mode == "lunar":
        ecl_type = "Total" if illum_win.min() < 0.02 else "Partial" if illum_win.min() < 0.999 else "None"
    else:
        # Solar eclipses distinguish total/annular by relative angular
        # size (Moon vs Sun as seen from Earth) at the deepest point, not
        # just how dark it gets — an annular eclipse never reaches
        # illum=0 (a bright ring always remains) even at its deepest.
        r_mid = np.linalg.norm(r_win[mid])
        moon_ang = R_MOON_KM / r_mid
        sun_ang = R_SUN_KM / AU_KM
        if illum_win.min() < 0.001:
            ecl_type = "Total"
        elif moon_ang < sun_ang:
            ecl_type = "Annular"
        else:
            ecl_type = "Partial"

    if verbose:
        print(f"[{mode}] Refined minimum illumination: {illum_win.min():.4f} ({ecl_type})")
        print(f"[{mode}] Angular separation at peak: {sep_deg_mid:.3f} deg")
        print(f"[{mode}] Peak phase duration: {dur_umbra_hr*60:.1f} min, "
              f"total event duration: {dur_total_hr:.2f} hr")

    fig = plt.figure(figsize=(15, 4.5), dpi=115)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    label = "Blood Moon (umbra)" if mode == "lunar" else "Deepest coverage"
    ax1.plot(t_hr, illum_win, color="#222222", linewidth=1.3)
    ax1.fill_between(t_hr, 0, illum_win, where=(illum_win < 0.999),
                     color="#553322", alpha=0.3, label="Partial phase")
    ax1.fill_between(t_hr, 0, illum_win, where=(illum_win < 0.02),
                     color="#3a0f08", alpha=0.75, label=label)
    ax1.set_xlabel("Time relative to deepest point [hours]")
    ax1.set_ylabel("Illumination fraction" if mode == "lunar" else "Sun visible fraction")
    ax1.set_title(f"{ecl_type} {mode} eclipse — found via real search\n"
                  f"min={illum_win.min():.3f}, angle at peak={sep_deg_mid:.3f}°, "
                  f"peak phase {dur_umbra_hr*60:.0f} min, event {dur_total_hr:.1f} hr")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.25)

    # ── Panel 2: appearance strip ────────────────────────────────────────────
    # Frame times are chosen non-linearly, scaled to THIS event's own real
    # duration (dur_total_hr) rather than spread evenly across the full
    # +/-48h search window — evenly-spaced-in-time frames spent almost all
    # of their budget on flat, unchanging full-moon/full-sun frames (the
    # event itself is only ~4 hours out of a 96-hour window) and gave only
    # one real transition frame. Denser sampling near the peak (matching
    # both reference photos' look: several visible partial-phase steps,
    # not just one) with a couple of untouched frames at each end for
    # context.
    n_frames = 9
    n_side = n_frames // 2
    bound_hr = max(dur_total_hr * 3.0, 6.0)
    ingress = np.where((t_hr < 0) & (t_hr > -bound_hr))[0]
    egress = np.where((t_hr > 0) & (t_hr < bound_hr))[0]
    illum_min = illum_win.min()

    # Target specific illumination LEVELS (not time offsets) for each
    # step, bunched non-linearly toward the deepest point — this
    # guarantees each frame shows a materially different phase (a real
    # crescent step) instead of time-based sampling, which could jump
    # straight from "full" to "totality" if the actual transition happens
    # faster than the time-grid spacing.
    #
    # The innermost target is capped at illum_min + 8% of the range
    # rather than illum_min itself — totality is genuinely FLAT at
    # illum_min for its whole duration (a real physical fact, not a
    # sampling gap), so a target of exactly illum_min ties with dozens of
    # indices and, combined with the already-shown centre frame, produced
    # duplicate all-red panels instead of a fourth distinct crescent step.
    frac_lin = np.linspace(0.05, 0.92, n_side)
    frac = frac_lin ** 0.6
    illum_targets = 1.0 - frac * (1.0 - illum_min)

    idxs = np.zeros(n_frames, dtype=int)

    # The centre panel should be the MOST-aligned instant, not merely the
    # first instant illum reaches its floor — illum is exactly flat for
    # totality's whole duration (a real physical fact), so argmin(illum)
    # alone ties across every point in that window and can land anywhere
    # in it, including a point still measurably off-centre within the
    # umbra (enough to render a faint sliver instead of a clean solid
    # disc / corona). Re-searching by actual geometric separation within
    # the flat zone picks the true best-aligned instant instead.
    flat_zone = np.where(illum_win <= illum_min + 1e-6)[0]
    if mode == "lunar":
        def _sep(i):
            dist_i = np.linalg.norm(r_win[i])
            occ_hat = r_win[i] / dist_i
            return np.arccos(np.clip(np.dot(-occ_hat, sun_win[i]), -1, 1))
    else:
        def _sep(i):
            r_i = np.linalg.norm(r_win[i])
            moon_hat = r_win[i] / r_i
            return np.arccos(np.clip(np.dot(moon_hat, sun_win[i]), -1, 1))
    seps = np.array([_sep(i) for i in flat_zone])
    center_idx = flat_zone[np.argmin(seps)]
    idxs[n_side] = center_idx

    for step, target in enumerate(illum_targets):
        k_in = step
        k_out = n_frames - 1 - step
        idxs[k_in] = ingress[np.argmin(np.abs(illum_win[ingress] - target))]
        idxs[k_out] = egress[np.argmin(np.abs(illum_win[egress] - target))]

    ax2.set_xlim(0, n_frames); ax2.set_ylim(0, 1); ax2.set_aspect('equal')
    ax2.set_facecolor("black")
    for k, idx in enumerate(idxs):
        # arccos() below always returns a positive magnitude — with no
        # sign, both renderers were showing the shadow/Moon approach from
        # one side and recede back to that SAME side, instead of sweeping
        # continuously through in one direction the way a real eclipse
        # does. Real motion during the brief eclipse window is close
        # enough to a straight-line pass that the sign of t_hr (before
        # vs after the deepest point) is a fine stand-in for the actual
        # direction of travel, without needing a full 2D vector
        # projection just for this cosmetic detail.
        side = -1.0 if t_hr[idx] < 0 else (1.0 if t_hr[idx] > 0 else 0.0)
        if mode == "lunar":
            # Real per-pixel crescent render instead of a flat colour
            # disc: convert this instant's REAL angular separation
            # between the Moon and Earth's shadow axis (the same
            # geometry already driving illum_win, not a synthetic
            # re-sweep) into the "shadow-offset in Moon-radii" units
            # render_lunar_panel expects, then rasterize an actual
            # crescent-shaped shadow at that offset.
            dist_i = np.linalg.norm(r_win[idx])
            occ_hat = r_win[idx] / dist_i          # Earth -> Moon direction
            moon_ang_r = np.arcsin(np.clip(R_MOON_KM / dist_i, -1, 1))
            sep = np.arccos(np.clip(np.dot(-occ_hat, sun_win[idx]), -1, 1))
            shadow_offset = side * sep / moon_ang_r     # in Moon-radii units, signed
            panel_img = render_lunar_panel(shadow_offset)
        else:
            # Dedicated solar renderer (real Sun/Moon disc overlap,
            # rasterized with an actual corona at totality/annularity) —
            # its own function, matching how lunar has its own, rather
            # than a shared inline matplotlib-patch fallback.
            r_i = np.linalg.norm(r_win[idx])
            moon_hat = r_win[idx] / r_i
            sun_ang_r = np.arcsin(np.clip(R_SUN_KM / AU_KM, -1, 1))
            sep = np.arccos(np.clip(np.dot(moon_hat, sun_win[idx]), -1, 1))
            moon_offset = side * sep / sun_ang_r        # in Sun-radii units, signed
            is_total_ish = illum_win[idx] < (illum_win.min() + 0.03)
            panel_img = render_solar_panel(moon_offset, corona=is_total_ish and illum_win.min() < 0.85)
        # RGBA with real transparency outside the disk (see
        # eclipse_appearance_strip.py) — panels sit cleanly against the
        # black axes background with a visible gap between them instead
        # of touching opaque black squares.
        ax2.imshow(np.asarray(panel_img), extent=(k+0.02, k+0.98, 0.02, 0.98), zorder=2)
        ax2.text(k+0.5, 0.06, f"{t_hr[idx]:+.1f}h", color="white", ha="center", fontsize=8)
        ax2.text(k+0.5, 0.94, f"{illum_win[idx]:.2f}", color="white", ha="center", fontsize=8)
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_title("Moon appearance through the event" if mode == "lunar"
                 else "Sun appearance through the event", color="black")

    # 3D "view from space" now lives only in eclipse_space_view_plotly.py
    # as interactive HTML, not duplicated here as a static matplotlib panel.

    fig.suptitle(f"{'Lunar' if mode=='lunar' else 'Solar'} eclipse simulation", fontsize=13, y=1.0)

    if mode == "lunar":
        caption = ("What this is: a real total/partial lunar eclipse, found by searching actual Moon-Earth-Sun geometry over "
                  "time rather than staging one. How it's made: two-body Keplerian propagation of the Moon's real orbit "
                  "(true a/e), real Sun-direction vectors, and the same two-circle Sun-disk-overlap illumination physics "
                  "used for shadow calculations throughout this toolkit. Each Moon panel is rasterized from that same real "
                  "geometry at that instant (a real umbra/penumbra boundary crossing an actual cratered surface), not a "
                  "flat colour swap.")
    else:
        caption = ("What this is: a real total/partial/annular solar eclipse, found by searching actual Moon-Earth-Sun "
                  "geometry over time rather than staging one. How it's made: the same real orbit propagation and "
                  "illumination physics as the lunar case, with occluder and occluded body swapped (Moon casts the "
                  "shadow, Earth's the reference point). Each Sun panel is rasterized from the real Sun/Moon angular "
                  "overlap at that instant, including an actual corona render once coverage is deep enough.")
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=9.5, wrap=True,
             transform=fig.transFigure)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    return fig, dict(mode=mode, eclipse_type=ecl_type, min_illum=illum_win.min(),
                     angle_at_peak_deg=sep_deg_mid,
                     dur_peak_hr=dur_umbra_hr, dur_total_hr=dur_total_hr)


def _plot_space_view_unified(ax, moon_r_km, sun_hat, illum, mode, sep_deg=None):
    """Sun, Earth, Moon, and the correct shadow cone direction for whichever
    mode is active (Earth's shadow toward the Moon for lunar; the Moon's
    shadow toward Earth for solar) — same rendering quality throughout."""
    Re = RE_KM
    size_boost = 16.0   # NOT 40 -- at 40x, Earth+Moon's boosted radii
    # summed to ~89% of their real center-to-center distance at eclipse
    # alignment, leaving only a ~11% visual gap: the two bodies looked
    # almost touching even though the Moon is genuinely about 60 Earth-
    # radii away. 16x still boosts both bodies enough to show real
    # surface detail, while leaving a real, honest ~64% gap between them.

    if mode == "lunar":
        earth_pos = np.array([0.0, 0.0, 0.0])
        moon_pos = moon_r_km
        cone_origin, cone_dir, cone_base_r = earth_pos, -sun_hat, Re*size_boost
        cone_len = min(Re / np.tan(np.arcsin((R_SUN_KM-Re)/AU_KM)), Re*80)
    else:
        # moon_r_km here is still "Moon relative to Earth" (r_win), so
        # Earth relative to Moon is its negation — matches the sign
        # convention used in the search above.
        moon_pos = np.array([0.0, 0.0, 0.0])
        earth_pos = -moon_r_km
        cone_origin, cone_dir, cone_base_r = moon_pos, -sun_hat, R_MOON_KM*size_boost
        cone_len = min(R_MOON_KM / np.tan(np.arcsin((R_SUN_KM-R_MOON_KM)/AU_KM)), R_MOON_KM*300)

    _earth_sphere_mpl(ax, earth_pos, Re*size_boost, sun_hat)
    tint = moon_color(illum) if mode == "lunar" else [0.75, 0.75, 0.78]
    _moon_sphere_mpl(ax, moon_pos, R_MOON_KM*size_boost, tint)

    # Shadow cone (tapered outline, top-down-collapsed wedge as before)
    perp = np.array([-cone_dir[1], cone_dir[0], 0.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([1.0, 0.0, 0.0])
    for sign in (-1, 1):
        edge = np.array([cone_origin + perp*sign*cone_base_r,
                         cone_origin + cone_dir*cone_len])
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2] if edge.shape[1] > 2 else [0, 0],
               color="#aa4444", alpha=0.7, linewidth=1.5)

    ref_r = np.linalg.norm(moon_r_km)
    sun_len = ref_r * 4.0   # further out than before (was 1.15x) — still
    # nowhere near the real ~390x ratio (1 AU vs Earth-Moon distance),
    # which would put the Sun far outside any usable frame, but enough to
    # read as "clearly further away" rather than sitting right next to
    # the Earth/Moon system.
    sun_center = sun_hat * sun_len + (earth_pos if mode == "solar" else 0)
    sun_radius = ref_r * 0.09
    _sun_sphere_mpl(ax, sun_center, sun_radius)
    ax.text(sun_center[0], sun_center[1], sun_center[2]+sun_radius*1.8,
            "Sun", color="#FFD700", fontsize=9, ha="center", zorder=9)

    lim = sun_len * 1.2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=18, azim=-50)
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    _angle_str = f", angle={sep_deg:.3f}°" if sep_deg is not None else ""
    ax.set_title(f"View from space — illum={illum:.3f}{_angle_str}", fontsize=11)


if __name__ == "__main__":
    fig, stats = find_and_plot_eclipse(mode="lunar", save_path="/home/claude/demo3/eclipse_lunar.png")
    print(stats)
    fig, stats = find_and_plot_eclipse(mode="solar", save_path="/home/claude/demo3/eclipse_solar.png")
    print(stats)




def _light_ray_traces(sun_pos, earth_pos, earth_r, moon_pos, moon_r, n_rays=None):
    """
    ONE direct light path along the actual eclipse axis (Sun -> Moon centre
    -> Earth), not a scattered fan of rays in random directions across
    Earth's disk — that fan was confusing rather than illustrative, and
    didn't clearly show the single alignment that causes the eclipse.

    The ray is drawn in two segments: solid yellow from the Sun to where
    it's blocked by the Moon, then a small tapered dark cone from that
    blocking point onward — this is literally "where the light would be
    blocked off by the Moon", not an abstract shadow shape.
    """
    to_earth = earth_pos - sun_pos
    dist = np.linalg.norm(to_earth)
    fwd = to_earth / dist

    # Exact ray-sphere intersection with the Moon along the direct
    # Sun->Earth line (this line passes very close to the Moon's centre
    # by construction, since that's the whole reason an eclipse is
    # happening at this instant).
    oc = sun_pos - moon_pos
    b = np.dot(oc, fwd)
    c = np.dot(oc, oc) - moon_r**2
    disc = b**2 - c
    if disc > 0:
        t_hit = -b - np.sqrt(disc)
        block_point = sun_pos + fwd * max(t_hit, 0)
    else:
        block_point = earth_pos  # (shouldn't happen during a real eclipse)

    traces = [go.Scatter3d(
        x=[sun_pos[0], block_point[0]], y=[sun_pos[1], block_point[1]], z=[sun_pos[2], block_point[2]],
        mode="lines", line=dict(color="#ffe066", width=4),
        hoverinfo="skip", showlegend=False, name="Sunlight",
    )]

    # Small tapered cone from the blocking point onward, showing exactly
    # where the light is cut off — reuses the same real cone-surface
    # builder as the Earth/Moon shadow cones elsewhere in this scene.
    cone_len = min(dist - np.linalg.norm(block_point - sun_pos), moon_r * 6)
    traces.append(_shadow_cone_trace(block_point, fwd, moon_r * 0.5, max(cone_len, moon_r),
                                     color="#552222", opacity=0.35))
    return traces


def _moon_mesh_plotly_REMOVED_use_moon_render_instead():
    """Superseded by moon_render.moon_mesh_plotly (real texture + real
    per-vertex diffuse lighting from bump-mapped normals instead of a
    flat-lit painted albedo). Kept as a stub only so any external code
    still importing the old name gets a clear pointer instead of a
    silent behavior change."""
    raise NotImplementedError("Use moon_render.moon_mesh_plotly instead.")


def _shadow_cone_trace(origin, direction, base_radius, length, color="#aa4444", opacity=0.18,
                       end_radius=0.0):
    """Real tapered 3D cone surface (not a flattened 2D wedge) pointing
    along `direction` from `origin`. Linear radius interpolation from
    `base_radius` at the origin to `end_radius` at `length` away —
    `end_radius=0` (the old default) gives the umbra's converging shape;
    passing an `end_radius` larger than `base_radius` gives the
    penumbra's diverging shape instead, using the exact same builder."""
    zc = np.linspace(0, length, 20)
    thetac = np.linspace(0, 2*np.pi, 24)
    Zc, Tc = np.meshgrid(zc, thetac)
    Rc = base_radius + (end_radius - base_radius) * (Zc/length)
    Xc, Yc = Rc*np.cos(Tc), Rc*np.sin(Tc)

    z_ax = np.array([0.0, 0.0, 1.0])
    d = direction / np.linalg.norm(direction)
    v = np.cross(z_ax, d)
    s = np.linalg.norm(v)
    c = np.dot(z_ax, d)
    if s > 1e-8:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1-c)/(s**2))
    else:
        R = np.eye(3) if c > 0 else -np.eye(3)
    pts = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()])
    rot = R @ pts
    X = origin[0] + rot[0].reshape(Xc.shape)
    Y = origin[1] + rot[1].reshape(Xc.shape)
    Z = origin[2] + rot[2].reshape(Xc.shape)

    return go.Surface(x=X, y=Y, z=Z, colorscale=[[0, color], [1, color]],
                      showscale=False, opacity=opacity, hoverinfo="skip",
                      lighting=dict(ambient=1.0, diffuse=0.0), name="Shadow")


def _umbra_penumbra_geometry(R_occ_km, size_boost, ref_dist_km=None, penumbra_ratio=1.35):
    """
    Real umbra + penumbra cone geometry for a shadow-casting body of
    radius R_occ_km, from the actual similar-triangles construction
    (Sun's real angular size determines how fast the shadow narrows) —
    not a calibrated guess. The real physical umbra length is:

        L_umbra = R_occ_km * AU_km / (R_sun_km - R_occ_km)

    which gives ~1.385 million km for Earth's shadow (this is the real,
    textbook number — and it comfortably exceeds the Earth-Moon distance,
    which is exactly why lunar eclipses can go total) and ~374,600 km for
    the Moon's shadow (which is *shorter* than the Earth-Moon distance
    most of the time — the real reason annular eclipses happen at all:
    the Moon's umbra apex falls short of Earth's surface, so only the
    surrounding antumbra reaches the ground).

    Using this real (unboosted) length together with the body's boosted
    display radius as the cone's base automatically reproduces the
    correct real-world ratio at the real Earth-Moon distance — no
    separate calibration step needed, and critically, no risk of
    accidentally applying one direction's ratio to the other direction's
    cone (a real bug in an earlier version of this function: it used a
    single 2.6x-Moon-radius target for both the Earth-shadow-on-Moon
    and Moon-shadow-on-Earth cases, even though those are two physically
    different cones with different real apex distances).

    `ref_dist_km` is accepted but unused — kept only so existing callers
    don't need updating.
    """
    L_umbra_real = R_occ_km * AU_KM / (R_SUN_KM - R_occ_km)
    base_r = R_occ_km * size_boost
    slope = base_r / L_umbra_real
    umbra_len = L_umbra_real
    pen_base_r = base_r
    pen_len = umbra_len
    pen_end_r = base_r + slope * penumbra_ratio * pen_len
    return base_r, umbra_len, pen_base_r, pen_len, pen_end_r, slope





def _shadow_ground_point(moon_pos, sun_hat, earth_pos, earth_r_real=None):
    """
    Where the real Sun->Moon shadow axis hits Earth's surface — the
    actual sub-shadow point that defines the path of totality/
    annularity, not just "the two bodies are roughly aligned". Same
    ray-sphere intersection math as _light_ray_traces' Sun->Moon check,
    aimed at Earth instead: ray origin = Moon center, direction = away
    from the Sun (-sun_hat), sphere = Earth at its REAL radius (the path
    is a physical ground location, computed before any display-scale
    boost is applied).

    Returns the real (unboosted) hit point in km, or None if the shadow
    axis misses Earth entirely at this instant (normal outside totality
    — most of a partial eclipse's duration, the umbra hasn't reached
    Earth's surface at all).
    """
    from globe_orbit_daynight_plotly import RE_KM as _RE_KM
    R = earth_r_real if earth_r_real is not None else _RE_KM
    d = -sun_hat / np.linalg.norm(sun_hat)
    oc = moon_pos - earth_pos
    b = np.dot(oc, d)
    c = np.dot(oc, oc) - R**2
    disc = b**2 - c
    if disc < 0:
        return None
    t_hit = -b - np.sqrt(disc)
    if t_hit < 0:
        return None
    return moon_pos + d * t_hit


def _starfield_trace(radius, n_stars=500, seed=7):
    """
    A static background starfield. Not recomputed per frame — real stars
    are so far away that their apparent positions genuinely don't shift
    over a multi-hour eclipse window, so keeping this fixed across every
    frame is the physically correct choice, not a shortcut. Random
    brightness/size per star for a natural, non-uniform look, placed just
    inside the scene's own axis range so they're actually visible instead
    of being clipped by the axis boundary.
    """
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n_stars, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    pts = vecs * radius * 0.97
    brightness = rng.uniform(0.35, 1.0, n_stars)
    sizes = rng.uniform(1.0, 2.6, n_stars)
    colors = [f"rgba(255,255,255,{b:.2f})" for b in brightness]
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                        marker=dict(size=sizes, color=colors), hoverinfo="skip",
                        showlegend=False, name="Stars")


def _lunar_or_solar_camera_eye(mode, sun_hat):
    """
    Real bug, not a style choice: lunar and solar eclipses need DIFFERENT
    camera directions, and using the same sun-relative camera for both
    was hiding the Moon entirely in lunar mode.

    Solar mode: the umbra/antumbra always lands on the sun-facing side of
    Earth, so the camera should look from roughly the Sun's direction —
    otherwise the shadow and ground-track path render correctly but on
    the far side of the globe from wherever the camera happens to point.

    Lunar mode: the Moon sits in OPPOSITION to the Sun (verified
    directly: 179.84 deg between Moon and Sun direction at real peak
    alignment, not 0). Aiming the camera toward the Sun's direction here
    puts the Moon almost exactly behind Earth from the camera's point of
    view — not just small, actually hidden. A side-on view, roughly
    perpendicular to the Sun direction, shows Earth and the Moon next to
    each other instead of one hiding behind the other.
    """
    if mode == "lunar":
        up_ref = np.array([0.0, 0.0, 1.0]) if abs(sun_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        side_hat = np.cross(sun_hat, up_ref)
        side_hat = side_hat / np.linalg.norm(side_hat)
        return dict(x=side_hat[0]*0.6, y=side_hat[1]*0.6, z=side_hat[2]*0.6 + 0.25)
    else:
        return dict(x=sun_hat[0]*0.55, y=sun_hat[1]*0.55+0.15, z=sun_hat[2]*0.55+0.2)


def _shadow_footprint_traces(earth_pos, hit_point_display, sun_hat, footprint_r_umbra, footprint_r_penumbra):
    """
    A real dark patch drawn directly on Earth's surface at the current
    ground-track point — this is the actual visible "there is a shadow
    on the Earth" cue that a single thin light-ray line can't provide.
    Built as a small flat disc in the local tangent plane at the hit
    point (a fine approximation at these footprint scales, which are
    small compared to Earth's radius) then nudged just above the
    surface so it doesn't z-fight with the Earth mesh underneath.

    Two nested discs: a wider, lighter penumbra footprint (partial
    shadow — where a solar eclipse would be seen as partial), and a
    smaller, darker umbra/antumbra footprint inside it (where totality
    or an annular eclipse would actually be visible) — same "layers"
    concept requested for the lunar-eclipse cones, applied on the
    ground instead of in space.
    """
    n_hat = hit_point_display - earth_pos
    n_hat = n_hat / np.linalg.norm(n_hat)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n_hat); u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    theta = np.linspace(0, 2*np.pi, 40)

    def _disc(radius, color, opacity):
        pts = (hit_point_display[None, :] + n_hat[None, :] * (radius * 0.002)
              + radius * (np.cos(theta)[:, None]*u[None, :] + np.sin(theta)[:, None]*v[None, :]))
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        n = len(theta)
        xs = np.concatenate([[hit_point_display[0]], x])
        ys = np.concatenate([[hit_point_display[1]], y])
        zs = np.concatenate([[hit_point_display[2]], z])
        i = [0] * (n - 1)
        j = list(range(1, n))
        k = list(range(2, n)) + [1]
        return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                         color=color, opacity=opacity, hoverinfo="skip", showlegend=False,
                         lighting=dict(ambient=1.0, diffuse=0.0), name="Shadow footprint")

    return [_disc(footprint_r_penumbra, "#3a3a55", 0.30),
           _disc(footprint_r_umbra, "#05050a", 0.55)]


def _sun_direction_arrow(origin, sun_hat, length, label="Sun direction"):
    """
    A directional vector pointing toward the Sun, replacing a literal
    small Sun sphere placed nearby. At real solar-system distances the
    Sun cannot be rendered to any consistent scale next to Earth/Moon —
    either it's a speck (uninformative) or, as before, an arbitrarily
    up-scaled sphere close enough to read as a size/distance, which
    visually (and misleadingly) reads as "a second, smaller moon" rather
    than "a star 150 million km away". A labeled arrow says exactly what
    it means — a direction, not a to-scale object — without implying a
    false distance or size.
    """
    shaft_end = origin + sun_hat * length * 0.88
    tip = origin + sun_hat * length
    shaft = go.Scatter3d(x=[origin[0], shaft_end[0]], y=[origin[1], shaft_end[1]], z=[origin[2], shaft_end[2]],
                         mode="lines", line=dict(color="#ffd34d", width=6),
                         hoverinfo="skip", showlegend=False, name=label)
    head = _shadow_cone_trace(shaft_end, sun_hat, length*0.035, length*0.12,
                              color="#ffd34d", opacity=0.9)
    text = go.Scatter3d(x=[tip[0]], y=[tip[1]], z=[tip[2]], mode="text",
                        text=[label], textfont=dict(color="#ffd34d", size=18),
                        hoverinfo="skip", showlegend=False)
    return [shaft, head, text]


def plot_space_view_animated(mode="lunar", search_days=None, n_frames=26,
                             save_path=None, n_lat=60, n_lon=120, verbose=True,
                             show_eclipse_path=True):
    """
    Animated version of plot_space_view_plotly: steps through the real
    eclipse time window (the same fine-resolution window eclipse_demo.py
    searches and refines) instead of showing one frozen instant. Two real
    motions play simultaneously, exactly as they would in reality:

      1. Earth's own sidereal rotation (fast — ~15 deg/hour), animated by
         rolling _earth_mesh's body-fixed fields via its `rotation_deg`
         parameter rather than recomputing the land mask / relief from
         scratch every frame (recomputing at n_frames >~ 20 would be the
         slow part otherwise).
      2. The Moon's real orbital motion through r_win (already the exact
         same array eclipse_demo.py's refined search produces) — this is
         what actually sweeps the eclipse shadow across the disk.

    The Sun's direction (sun_win) also technically changes over the
    window, but negligibly over a few hours, so its motion is included
    for correctness at essentially no visual cost.

    Mesh resolution is reduced by default (90x180 / Moon proportionally)
    since this rebuilds full Earth+Moon meshes per frame — n_frames=28 at
    full 180x360 resolution would be quite slow to generate.
    """
    assert mode in ("lunar", "solar")
    search_days = search_days or (365.0 if mode == "lunar" else 365.25*6)
    lunar_period_days = 27.32
    n_orbits_year = search_days / lunar_period_days
    coarse_density = 60 if mode == "lunar" else 1500
    t_s, r_moon, _ = propagate_eci(a_km=D_MOON_A_KM, e=D_MOON_E, inc_deg=D_MOON_INC_DEG,
                                   raan_deg=0, argp_deg=0, nu0_deg=0,
                                   n_orbits=n_orbits_year, n_steps=int(n_orbits_year*coarse_density))
    sun_hat_all = sun_direction_eci(t_s)
    r_eval = r_moon if mode == "lunar" else -r_moon
    R_occ = RE_KM if mode == "lunar" else R_MOON_KM
    illum_coarse = illumination_fraction(r_eval, sun_hat_all, R_body_km=R_occ, R_sun_km=R_SUN_KM, D_km=AU_KM)
    best_idx = int(np.argmin(illum_coarse))
    best_t = t_s[best_idx]

    MU_EARTH_KM3S2 = 398_600.4418
    window_days = 2.0
    n_rad_s = np.sqrt(MU_EARTH_KM3S2 / D_MOON_A_KM**3)
    t_window_start = best_t - window_days*86400
    M_start = (n_rad_s * t_window_start) % (2*np.pi)
    E_start = float(M_start)
    for _ in range(60):
        dE = (M_start - E_start + D_MOON_E*np.sin(E_start)) / (1 - D_MOON_E*np.cos(E_start))
        E_start += dE
    nu_start = 2*np.arctan2(np.sqrt(1+D_MOON_E)*np.sin(E_start/2), np.sqrt(1-D_MOON_E)*np.cos(E_start/2))
    n_orbits_window = (2*window_days) / lunar_period_days

    t_fine, r_fine, _ = propagate_eci(a_km=D_MOON_A_KM, e=D_MOON_E, inc_deg=D_MOON_INC_DEG,
                                      raan_deg=0, argp_deg=0, nu0_deg=np.degrees(nu_start),
                                      n_orbits=n_orbits_window, n_steps=4000)
    t_fine = t_fine + t_window_start
    sun_fine = sun_direction_eci(t_fine)
    r_eval_fine = r_fine if mode == "lunar" else -r_fine
    illum_fine = illumination_fraction(r_eval_fine, sun_fine, R_body_km=R_occ, R_sun_km=R_SUN_KM, D_km=AU_KM)
    mid = int(np.argmin(illum_fine))

    # Window of interest around the peak — a bit wider than the peak
    # phase itself so the animation shows the approach and departure, not
    # just the deepest instant.
    half_window_hr = 3.0 if mode == "lunar" else 1.5
    in_window = np.abs(t_fine - t_fine[mid]) < half_window_hr * 3600
    idxs_all = np.where(in_window)[0]
    frame_idxs = idxs_all[np.linspace(0, len(idxs_all)-1, n_frames).astype(int)]

    if verbose:
        print(f"[{mode}] Animating {n_frames} frames over "
              f"+/-{half_window_hr:.1f} hr around the peak "
              f"(illum min={illum_fine[mid]:.4f})")

    EARTH_SIDEREAL_DEG_PER_SEC = 360.0 / 86164.0905
    # 16x, not 40x -- at 40x the boosted Earth+Moon radii summed to ~89%
    # of their real center-to-center distance at eclipse alignment,
    # leaving only ~11% of real space between them (they looked almost
    # touching even though the Moon is genuinely ~60 Earth-radii away).
    # 16x still shows real surface detail while leaving an honest ~64% gap.
    size_boost = 16.0
    t0 = t_fine[frame_idxs[0]]

    ref_dist_km = D_MOON_A_KM  # shared calibration distance for both
    # directions of shadow (Earth's shadow reaching the Moon, or the
    # Moon's shadow reaching Earth) — this is the real Earth-Moon
    # semi-major axis, the physically relevant "other body" distance in
    # both cases.

    def _frame_traces(idx):
        moon_r_km = r_fine[idx]
        sun_hat = sun_fine[idx]
        illum = illum_fine[idx]
        rotation_deg = (t_fine[idx] - t0) * EARTH_SIDEREAL_DEG_PER_SEC

        # Earth-fixed frame for BOTH modes — Earth stays at the origin,
        # the Moon orbits around it. The previous solar-mode version used
        # a Moon-fixed frame (Earth orbiting the Moon) — physically
        # equivalent, but visually the Moon just sat still at the origin
        # every frame while Earth drifted, which looks wrong in an
        # animation: the Moon is the thing actually sweeping between Sun
        # and Earth in reality, and should visibly do so here too.
        earth_pos = np.array([0.0, 0.0, 0.0])
        moon_pos = moon_r_km
        occ_r_km = RE_KM if mode == "lunar" else R_MOON_KM
        cone_origin = earth_pos if mode == "lunar" else moon_pos
        cone_dir = -sun_hat
        # Nudge the cone's start point a couple percent further along its
        # own direction, burying its near cross-section just inside the
        # opaque body instead of exactly on its surface — at exact
        # coincidence, the cone's semi-transparent near-face and the
        # sphere's silhouette edge overlap almost exactly from most
        # camera angles, rendering as a visible dark ring right at the
        # Moon's edge that isn't a real shadow feature.
        cone_origin = cone_origin + cone_dir * (occ_r_km * size_boost * 0.03)
        shadow_kwargs = ({} if mode == "lunar"
                         else dict(shadow_body_center_km=moon_pos, shadow_body_radius_km=R_MOON_KM))

        traces = [
            _earth_mesh(sun_hat, n_lat=n_lat, n_lon=n_lon, radius_scale=size_boost,
                       center=tuple(earth_pos), rotation_deg=rotation_deg, **shadow_kwargs),
            _earth_atmosphere_trace(center=tuple(earth_pos), radius_scale=size_boost),
        ]
        real_moon_center = moon_r_km if mode == "lunar" else np.array([0.0, 0.0, 0.0])
        traces.append(moon_mesh_plotly(moon_pos, R_MOON_KM*size_boost, sun_hat=sun_hat,
                                       real_center_km=real_moon_center, mode=mode,
                                       n_lat=max(140, n_lat), n_lon=max(280, n_lon)))

        # Real umbra (converging, dark) + penumbra (diverging, lighter)
        # cone pair, calibrated so their cross-section at the real
        # Earth-Moon distance matches the actual physical umbra/penumbra
        # size there — this is what was missing entirely for lunar mode
        # ("no layers for total or partial") and what was clipping
        # straight through the Moon for solar mode (the old single cone's
        # base radius was sized from the boosted OCCLUDER radius with no
        # calibration against the other body's boosted size, so the two
        # boosted objects ended up at incompatible scales at the distance
        # that actually matters).
        base_r, umbra_len, pen_base_r, pen_len, pen_end_r, slope = _umbra_penumbra_geometry(
            occ_r_km, size_boost, ref_dist_km)
        traces.append(_shadow_cone_trace(cone_origin, cone_dir, base_r, umbra_len,
                                         color="#1a0a0a", opacity=0.12, end_radius=0.0))
        traces.append(_shadow_cone_trace(cone_origin, cone_dir, pen_base_r, pen_len,
                                         color="#4a5a7a", opacity=0.045, end_radius=pen_end_r))

        # Directional Sun vector instead of a small sphere sitting nearby
        # (see _sun_direction_arrow docstring) — anchored just outside
        # Earth/Moon's boosted extent so it doesn't overlap either body.
        ref_r = np.linalg.norm(r_fine[mid])
        arrow_len = ref_r * 0.55  # short enough not to force the whole
        # scene to zoom out to fit it — a vector only needs to be long
        # enough to clearly indicate a direction, not to reach any
        # particular distance
        arrow_origin = earth_pos + sun_hat * (RE_KM*size_boost*1.15 if mode == "lunar"
                                              else np.linalg.norm(moon_pos) + R_MOON_KM*size_boost*1.15)
        traces.extend(_sun_direction_arrow(arrow_origin, sun_hat, arrow_len))

        # Axis range is set from Earth/Moon's own extent (see `lim` below),
        # NOT from the Sun arrow — the arrow used to be long enough that
        # fitting it in frame forced a ~3-million-km-wide view, at which
        # scale the Moon's real ~10,000 km shift over the animation window
        # was under 1% of the frame and effectively invisible, even though
        # it was being computed correctly the whole time.
        sun_len = ref_r  # kept only for the return signature below
        return traces, illum, sun_len, occ_r_km, base_r, umbra_len, slope

    # Real ground-track of the shadow axis hitting Earth's surface — the
    # actual path of totality/annularity, computed once per frame index
    # up front (cheap — just a ray-sphere intersection) so the animation
    # loop below can draw the accumulated path so far at each frame
    # rather than only ever showing the current instant.
    ground_track_display = []
    ground_footprint_r = []  # (umbra_r, penumbra_r) at the hit point, display units
    if mode == "solar" and show_eclipse_path:
        base_r0, umbra_len0, pen_base_r0, pen_len0, pen_end_r0, slope0 = _umbra_penumbra_geometry(
            R_MOON_KM, size_boost, ref_dist_km)

        # First pass: real, uncompressed hit points (or None).
        raw_hits = []
        for idx in frame_idxs:
            moon_pos_i = r_fine[idx]
            sun_hat_i = sun_fine[idx]
            earth_pos_i = np.array([0.0, 0.0, 0.0])
            # A raw ray-sphere intersection test alone isn't enough here:
            # Earth is a large target (6378 km radius) relative to how
            # tightly the umbra/antumbra actually has to align for a real
            # eclipse, so the geometric shadow AXIS can keep "hitting"
            # Earth's sphere well outside the real event window — found
            # this directly: every single frame showed a hit, including
            # ones at illum=1.000 (no eclipse effect at all, Sun fully
            # visible from Earth's own center). Gate on the real
            # illumination value too, so the shadow only appears when a
            # real eclipse effect is actually present.
            no_real_eclipse = illum_fine[idx] > 0.999
            hit = None if no_real_eclipse else _shadow_ground_point(
                moon_pos_i, sun_hat_i, earth_pos_i, earth_r_real=RE_KM)
            raw_hits.append(hit)

        for k, idx in enumerate(frame_idxs):
            hit = raw_hits[k]
            earth_pos_i = np.array([0.0, 0.0, 0.0])
            moon_pos_i = r_fine[idx]
            if hit is None:
                ground_track_display.append(None)
                ground_footprint_r.append((0.0, 0.0))
            else:
                unit = (hit - earth_pos_i)
                unit = unit / np.linalg.norm(unit)
                # NO compression here (an earlier version of this tried
                # compressing the angular deviation toward the peak
                # direction, first with a linear blend, then with proper
                # slerp) — that whole approach was based on a flawed
                # comparison. The shadow's real angular sweep across
                # Earth's surface (tens of degrees over the active
                # window — checked directly: ~33 deg just 40 samples from
                # peak, so wider across the full window) and the Moon's
                # real 3D position shift (small relative to its vast
                # distance from Earth) are two different physical
                # quantities that were never going to match by
                # comparing their raw magnitudes. The shadow SHOULD sweep
                # a visibly large fraction of Earth's disk — that's
                # genuinely how a real eclipse's ground track looks over
                # a couple of hours — while the Moon's own position
                # barely shifts on screen simply because the Earth-Moon
                # distance dwarfs how far it moves in that time. Forcing
                # them to match was the actual mistake.
                ground_track_display.append(earth_pos_i + unit * RE_KM * size_boost)
                dist_from_moon = np.linalg.norm(hit - moon_pos_i)
                # Signed radius from the linear taper — once dist_from_moon
                # exceeds the umbra's real apex distance, this goes
                # negative and represents the antumbra re-widening past
                # the apex (the real shape during an annular eclipse,
                # where the umbra falls short of Earth and only the
                # surrounding antumbra reaches the ground) rather than
                # "no shadow at all".
                r_umbra = abs(base_r0 - slope0*dist_from_moon)
                r_pen = pen_base_r0 + slope0*1.35*dist_from_moon
                ground_footprint_r.append((r_umbra, r_pen))
    else:
        ground_track_display = [None] * len(frame_idxs)
        ground_footprint_r = [(0.0, 0.0)] * len(frame_idxs)

    def _path_traces(k):
        """Growing red path line (every hit point up to and including
        frame k) plus a bright marker and a real shadow footprint disc
        at the current shadow location — always exactly 4 traces, even
        when empty, so every frame has the same trace count/order
        (Plotly assumes this when scrubbing the slider out of order)."""
        pts = [p for p in ground_track_display[:k+1] if p is not None]
        if pts:
            arr = np.array(pts)
            line_tr = go.Scatter3d(x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="lines",
                                   line=dict(color="#ff3b1f", width=7),
                                   name="Path of eclipse", hoverinfo="skip", showlegend=False)
        else:
            line_tr = go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False)

        # The marker specifically must reflect THIS frame's real hit, not
        # just the last point anywhere in the accumulated path — using
        # arr[-1] unconditionally meant the marker kept sitting at the
        # last known position indefinitely even after the real eclipse
        # ended (ground_track_display[k] is None there), which visually
        # implied an active shadow that wasn't actually there anymore.
        if ground_track_display[k] is not None:
            p = ground_track_display[k]
            marker_tr = go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                     marker=dict(color="#ffcc00", size=6, line=dict(color="#ff3b1f", width=1)),
                                     name="Current shadow center", hoverinfo="skip", showlegend=False)
        else:
            marker_tr = go.Scatter3d(x=[], y=[], z=[], mode="markers", hoverinfo="skip", showlegend=False)

        if ground_track_display[k] is not None:
            r_umbra, r_pen = ground_footprint_r[k]
            footprint = _shadow_footprint_traces(np.array([0.0, 0.0, 0.0]), ground_track_display[k],
                                                 sun_fine[frame_idxs[k]], r_umbra, r_pen)
        else:
            footprint = [go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], hoverinfo="skip", showlegend=False),
                        go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], hoverinfo="skip", showlegend=False)]
        return [line_tr, marker_tr] + footprint

    ref_r_lim = np.linalg.norm(r_fine[mid])
    lim = (ref_r_lim + max(RE_KM, R_MOON_KM)*size_boost) * 1.35
    star_trace = _starfield_trace(lim)

    first_traces, first_illum, sun_len, *_ = _frame_traces(frame_idxs[0])
    if mode == "solar" and show_eclipse_path:
        first_traces = first_traces + _path_traces(0)
    first_traces = first_traces + [star_trace]
    fig = go.Figure(data=first_traces)

    frames = []
    for k, idx in enumerate(frame_idxs):
        traces, illum, *_ = _frame_traces(idx)
        if mode == "solar" and show_eclipse_path:
            traces = traces + _path_traces(k)
        traces = traces + [star_trace]
        t_rel_hr = (t_fine[idx] - t_fine[mid]) / 3600.0
        frames.append(go.Frame(
            data=traces, name=str(k),
            layout=go.Layout(title=dict(
                text=f"{mode.capitalize()} eclipse — animated<br>"
                    f"<sub>t={t_rel_hr:+.2f} hr, illum={illum:.3f}</sub>",
                x=0.5, font=dict(color="white", size=15))),
        ))
    fig.frames = frames

    sun_hat_mid = sun_fine[mid]
    # Solar mode: the umbra/antumbra always lands on the sun-facing side
    # of Earth, by definition — a generic fixed camera angle has no
    # reason to be looking at that hemisphere, which is exactly why the
    # shadow dimming and the ground-track path looked absent: they were
    # being computed correctly but rendered on the far side of the globe
    # from wherever the camera happened to be pointed. Same fix the
    # static plot_space_view_plotly already used for solar mode, applied
    # here too.
    #
    # Lunar mode needs a DIFFERENT camera direction, not the same one —
    # this was a real bug: in a lunar eclipse the Moon sits in OPPOSITION
    # to the Sun (verified directly: 179.84 deg between Moon and Sun
    # direction at peak, not 0), so aiming the camera toward the Sun's
    # direction puts the Moon almost exactly behind Earth from the
    # camera's point of view — hidden, not just small. A side-on view,
    # roughly perpendicular to the Sun direction, shows Earth and the
    # Moon next to each other instead of one hiding behind the other.
    camera_eye = _lunar_or_solar_camera_eye(mode, sun_hat_mid)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title=dict(text="X [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            yaxis=dict(range=[-lim, lim], title=dict(text="Y [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            zaxis=dict(range=[-lim, lim], title=dict(text="Z [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            bgcolor="black",
            aspectmode="cube",
            camera=dict(eye=camera_eye),
        ),
        paper_bgcolor="black",
        font=dict(color="white", size=18),
        title=dict(text=f"{mode.capitalize()} eclipse — animated<br>"
                        f"<sub>t={(t_fine[frame_idxs[0]]-t_fine[mid])/3600:+.2f} hr, illum={first_illum:.3f}</sub>",
                  x=0.5, font=dict(color="white", size=24)),
        margin=dict(l=0, r=0, t=90, b=0),
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.05, xanchor="left", yanchor="top",
            pad=dict(t=0, r=10),
            # Play/Pause text defaults to small black lettering, which was
            # completely invisible against this scene's black background
            # — not just small, actually unreadable regardless of vision.
            # Explicit white font plus a visible button background fixes
            # both the contrast and the size at once.
            font=dict(color="white", size=16),
            bgcolor="#333344", bordercolor="#888", borderwidth=1,
            buttons=[
                dict(label="▶ Play", method="animate",
                    args=[None, dict(frame=dict(duration=180, redraw=True),
                                     fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, y=0, x=0.15, len=0.8, xanchor="left", yanchor="top",
            pad=dict(t=0, b=10),
            font=dict(color="white", size=15),
            currentvalue=dict(font=dict(color="white", size=16), prefix="t = "),
            steps=[dict(method="animate", label=f"{(t_fine[idx]-t_fine[mid])/3600:+.1f}h",
                       args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                  for k, idx in enumerate(frame_idxs)],
        )],
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved -> {save_path}")
    return fig


def plot_space_view_plotly(mode="lunar", search_days=None, save_path=None, verbose=True):
    """Recomputes the same real search as find_and_plot_eclipse() (so this
    can be called standalone) and renders the space-view panel in Plotly."""
    _, stats = find_and_plot_eclipse(mode=mode, search_days=search_days,
                                     save_path=None, verbose=verbose)
    # Re-derive the peak-instant vectors at the SAME refined resolution
    # eclipse_demo.py uses internally (not just the coarse search sample,
    # which can be off by a lot — confirmed: coarse gave illum=0.32 at a
    # point where the true refined minimum is illum=0.00).
    lunar_period_days = 27.32
    search_days = search_days or (365.0 if mode == "lunar" else 365.25*6)
    n_orbits_year = search_days / lunar_period_days
    coarse_density = 60 if mode == "lunar" else 1500
    t_s, r_moon, _ = propagate_eci(a_km=D_MOON_A_KM, e=D_MOON_E, inc_deg=D_MOON_INC_DEG,
                                   raan_deg=0, argp_deg=0, nu0_deg=0,
                                   n_orbits=n_orbits_year, n_steps=int(n_orbits_year*coarse_density))
    sun_hat_all = sun_direction_eci(t_s)
    r_eval = r_moon if mode == "lunar" else -r_moon
    R_occ = RE_KM if mode == "lunar" else R_MOON_KM
    illum_coarse = illumination_fraction(r_eval, sun_hat_all, R_body_km=R_occ, R_sun_km=R_SUN_KM, D_km=AU_KM)
    best_idx = int(np.argmin(illum_coarse))
    best_t = t_s[best_idx]

    # Same targeted fine-window propagation as eclipse_demo.py's refine step
    MU_EARTH_KM3S2 = 398_600.4418
    window_days = 2.0
    n_rad_s = np.sqrt(MU_EARTH_KM3S2 / D_MOON_A_KM**3)
    t_window_start = best_t - window_days*86400
    M_start = (n_rad_s * t_window_start) % (2*np.pi)
    E_start = float(M_start)
    for _ in range(60):
        dE = (M_start - E_start + D_MOON_E*np.sin(E_start)) / (1 - D_MOON_E*np.cos(E_start))
        E_start += dE
    nu_start = 2*np.arctan2(np.sqrt(1+D_MOON_E)*np.sin(E_start/2), np.sqrt(1-D_MOON_E)*np.cos(E_start/2))
    n_orbits_window = (2*window_days) / lunar_period_days

    t_fine, r_fine, _ = propagate_eci(a_km=D_MOON_A_KM, e=D_MOON_E, inc_deg=D_MOON_INC_DEG,
                                      raan_deg=0, argp_deg=0, nu0_deg=np.degrees(nu_start),
                                      n_orbits=n_orbits_window, n_steps=4000)
    t_fine = t_fine + t_window_start
    sun_fine = sun_direction_eci(t_fine)
    r_eval_fine = r_fine if mode == "lunar" else -r_fine
    illum_fine = illumination_fraction(r_eval_fine, sun_fine, R_body_km=R_occ, R_sun_km=R_SUN_KM, D_km=AU_KM)
    mid = int(np.argmin(illum_fine))

    moon_r_km = r_fine[mid]
    sun_hat = sun_fine[mid]
    illum = illum_fine[mid]
    sep_deg = stats["angle_at_peak_deg"]

    size_boost = 16.0   # not 40 -- see the animated function for why
    fig = go.Figure()

    # Earth-fixed frame for both modes, same as the animated function —
    # Earth at the origin, Moon at its real position relative to Earth.
    earth_pos = np.array([0.0, 0.0, 0.0])
    moon_pos = moon_r_km
    occ_r_km = RE_KM if mode == "lunar" else R_MOON_KM
    cone_origin = earth_pos if mode == "lunar" else moon_pos
    cone_dir = -sun_hat
    cone_origin = cone_origin + cone_dir * (occ_r_km * size_boost * 0.03)

    _shadow_kwargs = {}
    if mode == "solar":
        _shadow_kwargs = dict(shadow_body_center_km=moon_pos, shadow_body_radius_km=R_MOON_KM)
    fig.add_trace(_earth_mesh(sun_hat, radius_scale=size_boost, center=tuple(earth_pos), **_shadow_kwargs))
    fig.add_trace(_earth_atmosphere_trace(center=tuple(earth_pos), radius_scale=size_boost))
    _real_moon_center = moon_r_km if mode == "lunar" else np.array([0.0, 0.0, 0.0])
    fig.add_trace(moon_mesh_plotly(moon_pos, R_MOON_KM*size_boost, sun_hat=sun_hat,
                                   real_center_km=_real_moon_center, mode=mode))

    ref_dist_km = D_MOON_A_KM
    base_r, umbra_len, pen_base_r, pen_len, pen_end_r, slope = _umbra_penumbra_geometry(
        occ_r_km, size_boost, ref_dist_km)
    fig.add_trace(_shadow_cone_trace(cone_origin, cone_dir, base_r, umbra_len,
                                     color="#1a0a0a", opacity=0.12, end_radius=0.0))
    fig.add_trace(_shadow_cone_trace(cone_origin, cone_dir, pen_base_r, pen_len,
                                     color="#4a5a7a", opacity=0.045, end_radius=pen_end_r))

    if mode == "solar":
        hit = None if illum > 0.999 else _shadow_ground_point(moon_pos, sun_hat, earth_pos, earth_r_real=RE_KM)
        if hit is not None:
            unit = (hit - earth_pos); unit = unit / np.linalg.norm(unit)
            hit_display = earth_pos + unit * RE_KM * size_boost
            dist_from_moon = np.linalg.norm(hit - moon_pos)
            r_umbra = abs(base_r - slope*dist_from_moon)
            r_pen = pen_base_r + slope*1.35*dist_from_moon
            for tr in _shadow_footprint_traces(earth_pos, hit_display, sun_hat, r_umbra, r_pen):
                fig.add_trace(tr)

    ref_r = np.linalg.norm(moon_r_km)
    arrow_len = ref_r * 0.55
    arrow_origin = earth_pos + sun_hat * (RE_KM*size_boost*1.15 if mode == "lunar"
                                          else np.linalg.norm(moon_pos) + R_MOON_KM*size_boost*1.15)
    for tr in _sun_direction_arrow(arrow_origin, sun_hat, arrow_len):
        fig.add_trace(tr)

    lim = (ref_r + max(RE_KM, R_MOON_KM)*size_boost) * 1.35
    fig.add_trace(_starfield_trace(lim))
    # Close initial zoom: a small `eye` vector (short distance from origin,
    # in the scene's own normalized units) starts the camera zoomed in near
    # the Earth/Moon system rather than Plotly's usual zoomed-out default —
    # scroll/drag to zoom out from there, same as everywhere else this has
    # come up in this project.
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title=dict(text="X [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            yaxis=dict(range=[-lim, lim], title=dict(text="Y [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            zaxis=dict(range=[-lim, lim], title=dict(text="Z [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            bgcolor="black",
            aspectmode="cube",
            camera=dict(eye=_lunar_or_solar_camera_eye(mode, sun_hat)),
        ),
        paper_bgcolor="black",
        font=dict(color="white", size=18),
        title=dict(text=f"{mode.capitalize()} eclipse — view from space<br>"
                        f"<sub>illum={illum:.3f}, angle at peak={sep_deg:.3f}°</sub>",
                  x=0.5, font=dict(color="white", size=24)),
        margin=dict(l=0, r=0, t=90, b=0),
        showlegend=False,
    )

    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, width=1000, height=900, scale=1)
        print(f"Saved -> {save_path}")
    return fig, stats


if __name__ == "__main__":
    plot_space_view_plotly(mode="lunar", save_path="/home/claude/demo3/eclipse_space_lunar.html")
    plot_space_view_plotly(mode="solar", save_path="/home/claude/demo3/eclipse_space_solar.html")