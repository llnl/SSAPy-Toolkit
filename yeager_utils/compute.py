# flake8: noqa: E501

import numpy as np
from .constants import EARTH_RADIUS, MOON_RADIUS, RGEO
from .utils import divby0
from ssapy.body import get_body


def caclulate_errors(data, CI=.05):
    data_median = []
    data = np.sort(data)
    median_ = np.nanmedian(data)
    data_median.append(median_)
    err = [median_ - data[int(CI * len(data))], data[int((1 - CI) * len(data))] - median_]
    return err, data_median


def FFT(data, time_between_samples=1):
    N = len(data)
    k = int(N / 2)
    f = np.linspace(0.0, 1 / (2 * time_between_samples), N // 2)
    Y = np.abs(np.fft.fft(data))[:k]
    return (f, Y)


def FFTP(data, time_between_samples=1):
    N = len(data)
    k = int(N / 2)
    f = np.linspace(0.0, 1 / (2 * time_between_samples), N // 2)
    Tp = [divby0(1, float(item), len(data) * time_between_samples) for item in f]
    Y = np.abs(np.fft.fft(data))[:k]
    return (Tp, Y)


def proper_motion(x, y, z, vx, vy, vz, xe=0, ye=0, ze=0, vxe=0, vye=0, vze=0, input_unit='si'):
    # Units: distance (au), time (2*pi years), angles arcseconds
    # Find Position and Velocity Relative to Earth
    x_rot = x - xe
    y_rot = y - ye
    z_rot = z - ze
    vx_rot = vx - vxe
    vy_rot = vy - vye
    vz_rot = vz - vze

    d_earth_mag = np.linalg.norm([x_rot, y_rot, z_rot])
    if d_earth_mag == 0:
        pm = np.nan
        return pm
    v_ast_earth = np.array([vx_rot, vy_rot, vz_rot])
    los_vector = np.array([x_rot, y_rot, z_rot])

    v_los = np.linalg.norm((np.dot(v_ast_earth, los_vector) / np.linalg.norm(los_vector)))
    v_transverse = np.sqrt(np.linalg.norm(v_ast_earth)**2 - v_los**2)
    if input_unit == 'si':
        pm = v_transverse / d_earth_mag * 206265
        return pm
    elif input_unit == 'rebound':
        pm = v_transverse / d_earth_mag * 206265 / (31557600 * 2 * np.pi)  # v_transverse/d_earth_mag is in (au/sim_time)/au (2*pi*year), convert to arcseconds / second
        return pm
    else:
        print('Error - units provided not available, provide either SI or rebound units.')
        return


def getAngle(a, b, c):  # a,b,c where b is the vertex
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    ba = np.subtract(a, b)
    bc = np.subtract(c, b)
    cosine_angle = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.arccos(cosine_angle)


def moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels):  # In SI units, takes single values or arrays returns a fractional flux
    # https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    moon_phase_angle = getAngle(r_sun, r_moon, r_sat)  # Phase of the moon as viewed from the sat.
    sun_angle = getAngle(r_sun, r_sat, r_moon)  # angle from Sun to object to Earth
    moon_to_earth_angle = getAngle(r_moon, r_sat, r_earth)
    r_moon_sat = np.linalg.norm(r_sat - r_moon, axis=-1)
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)  # Earth is the observer.
    flux_moon_to_sat = 2 / 3 * albedo_moon * MOON_RADIUS**2 / (np.pi * (r_moon_sat)**2) * (np.sin(moon_phase_angle) + (np.pi - moon_phase_angle) * np.cos(moon_phase_angle))  # Fraction of sunlight reflected from the Moon to satellite
    # Fraction of light from back of solar panel
    flux_back = np.zeros_like(sun_angle)
    flux_back[sun_angle > np.pi / 2] = np.abs(albedo_back * area_panels / (np.pi * r_earth_sat[sun_angle > np.pi / 2]**2) * np.cos(np.pi - moon_to_earth_angle[sun_angle > np.pi / 2]) * flux_moon_to_sat[sun_angle > np.pi / 2])  # Fraction of Moon light reflected off back of solar panels - which are assumed to be always facing the Sun. Angle: Sun - Observer - Sat
    flux_front = np.zeros_like(sun_angle)
    flux_front[sun_angle < np.pi / 2] = np.abs(albedo_front * area_panels / (np.pi * r_earth_sat[sun_angle < np.pi / 2]**2) * np.cos(moon_to_earth_angle[sun_angle < np.pi / 2]) * flux_moon_to_sat[sun_angle < np.pi / 2])  # Fraction of Sun light scattered off front of the solar panels - which are assumed to be always facing the Sun. Angle: Sun - Sat - Observer
    flux_panels = flux_back + flux_front
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_moon_to_sat
    return {'moon_bus': flux_bus, 'moon_panels': flux_panels}


def earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels):  # In SI units, takes single values or arrays returns a flux
    # https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    phase_angle = getAngle(r_sun, r_sat, r_earth)  # angle from Sun to object to Earth
    earth_angle = np.pi - phase_angle  # Sun to Earth to oject.
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)  # Earth is the observer.
    flux_earth_to_sat = 2 / 3 * albedo_earth * EARTH_RADIUS**2 / (np.pi * (r_earth_sat)**2) * (np.sin(earth_angle) + (np.pi - earth_angle) * np.cos(earth_angle))  # Fraction of sunlight reflected from the Earth to satellite
    # Fraction of light from back of solar panel
    flux_back = np.zeros_like(phase_angle)
    flux_back[phase_angle > np.pi / 2] = albedo_back * area_panels / (np.pi * r_earth_sat[phase_angle > np.pi / 2]**2) * np.cos(np.pi - phase_angle[phase_angle > np.pi / 2]) * flux_earth_to_sat[phase_angle > np.pi / 2]  # Fraction of Earth light reflected off back of solar panels - which are assumed to be always facing the Sun. Angle: Sun - Observer - Sat
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_earth_to_sat
    return {'earth_bus': flux_bus, 'earth_panels': flux_back}


def sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels):  # In SI units, takes single values or arrays returns a fractional flux
    # https://amostech.com/TechnicalPapers/2013/POSTER/COGNION.pdf
    phase_angle = getAngle(r_sun, r_sat, r_earth)  # angle from Sun to object to Earth
    r_earth_sat = np.linalg.norm(r_sat - r_earth, axis=-1)  # Earth is the observer.
    flux_front = np.zeros_like(phase_angle)
    flux_front[phase_angle < np.pi / 2] = albedo_front * area_panels / (np.pi * r_earth_sat[phase_angle < np.pi / 2]**2) * np.cos(phase_angle[phase_angle < np.pi / 2])  # Fraction of Sun light scattered off front of the solar panels - which are assumed to be always facing the Sun. Angle: Sun - Sat - Observer
    flux_bus = 2 / 3 * albedo * radius**2 / (np.pi * (r_earth_sat)**2) * (np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle))  # Fraction of light reflected off satellite from Sun
    return {'sun_bus': flux_bus, 'sun_panels': flux_front}


def calc_M_v(r_sat, r_earth, r_sun, r_moon=False, radius=0.4, albedo=0.20, sun_Mag=4.80, albedo_earth=0.30, albedo_moon=0.12, albedo_back=0.50, albedo_front=0.05, area_panels=100, return_components=False):
    r_sun_sat = np.linalg.norm(r_sat - r_sun, axis=-1)
    frac_flux_sun = {'sun_bus': 0, 'sun_panels': 0}
    frac_flux_earth = {'earth_bus': 0, 'earth_panels': 0}
    frac_flux_moon = {'moon_bus': 0, 'moon_panels': 0}
    frac_flux_sun = sun_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_front, area_panels)
    frac_flux_earth = earth_shine(r_sat, r_earth, r_sun, radius, albedo, albedo_earth, albedo_back, area_panels)
    if r_moon is not False:
        frac_flux_moon = moon_shine(r_moon, r_sat, r_earth, r_sun, radius, albedo, albedo_moon, albedo_back, albedo_front, area_panels)
    merged_dict = {**frac_flux_sun, **frac_flux_earth, **frac_flux_moon}
    total_frac_flux = sum(merged_dict.values())
    Mag_v = (2.5 * np.log10((r_sun_sat / (10 * u.Unit('parsec').to(u.Unit('m'))))**2) + sun_Mag) - 2.5 * np.log10(total_frac_flux)
    if return_components:
        return Mag_v, merged_dict
    else:
        return Mag_v


def M_v_lambertian(r_sat, times, radius=1.0, albedo=0.20, sun_Mag=4.80, albedo_earth=0.30, albedo_moon=0.12, plot=False):
    pc_to_m = 3.085677581491367e+16
    r_sun = get_body('Sun').position(times).T
    r_moon = get_body('Moon').position(times).T
    r_earth = np.zeros_like(r_sun)

    r_sun_sat = np.linalg.norm(r_sat - r_sun, axis=-1)
    r_earth_sat = np.linalg.norm(r_sat, axis=-1)
    r_moon_sat = np.linalg.norm(r_sat - r_moon, axis=-1)

    sun_angle = getAngle(r_sun, r_sat, r_earth)
    earth_angle = np.pi - sun_angle
    moon_phase_angle = getAngle(r_sun, r_moon, r_sat)  # Phase of the moon as viewed from the sat.
    moon_to_earth_angle = getAngle(r_moon, r_sat, r_earth)

    flux_moon_to_sat = 2 / 3 * albedo_moon * MOON_RADIUS**2 / (np.pi * (r_moon_sat)**2) * (np.sin(moon_phase_angle) + (np.pi - moon_phase_angle) * np.cos(moon_phase_angle))  # Fraction of sunlight reflected from the Moon to satellite
    flux_earth_to_sat = 2 / 3 * albedo_earth * EARTH_RADIUS**2 / (np.pi * (r_earth_sat)**2) * (np.sin(earth_angle) + (np.pi - earth_angle) * np.cos(earth_angle))  # Fraction of sunlight reflected from the Earth to satellite

    frac_flux_sun = 2 / 3 * albedo * radius**2 / (np.pi * (r_earth_sat)**2) * (np.sin(sun_angle) + (np.pi - sun_angle) * np.cos(sun_angle))  # Fraction of light reflected off satellite from Sun
    frac_flux_earth = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_earth_to_sat
    frac_flux_moon = 2 / 3 * albedo * radius**2 / (np.pi * r_earth_sat**2) * flux_moon_to_sat
    Mag_v = (2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(frac_flux_sun + frac_flux_earth + frac_flux_moon)
    if plot:
        import matplotlib.pyplot as plt
        sun_scale = 149597870700.0 * (RGEO / np.max(r_earth_sat) ) * 0.75
        color_map ='inferno_r'
        fig = plt.figure(figsize=(18, 4))
        ax = fig.add_subplot(1, 4, 1)
        ax.scatter(r_earth[:, 0], r_earth[:, 1], c='Blue', s=10)
        scatter = ax.scatter(r_sat[:, 0] / RGEO, r_sat[:, 1] / RGEO, c=sun_angle, cmap=color_map)
        colorbar = plt.colorbar(scatter)
        ax.scatter(r_sun[:, 0] / sun_scale, r_sun[:, 1] / sun_scale, c=plt.cm.Oranges(np.linspace(0.25, 0.75, len(r_sat[:, 0]))), s=10)
        ax.set_title('Solar Phase')
        ax.set_xlabel('X [GEO]')
        ax.set_ylabel('Y [GEO]')
        ax.axis('equal')

        ax = fig.add_subplot(1, 4, 2)
        ax.scatter(r_earth[0], r_earth[1], c='Blue', s=10)
        scatter = ax.scatter(r_sat[:, 0] / RGEO, r_sat[:, 1] / RGEO, c=(2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(frac_flux_sun), cmap=color_map)
        colorbar = plt.colorbar(scatter)
        ax.scatter(r_sun[:, 0] / sun_scale, r_sun[:, 1] / sun_scale, c=plt.cm.Oranges(np.linspace(0.25, 0.75, len(r_sat[:, 0]))), s=10)
        ax.set_title('Solar M_v')
        ax.axis('equal')

        ax = fig.add_subplot(1, 4, 3)
        ax.scatter(r_earth[:, 0], r_earth[:, 1], c='Blue', s=10)
        scatter = ax.scatter(r_sat[:, 0] / RGEO, r_sat[:, 1] / RGEO, c=(2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(frac_flux_earth), cmap=color_map)
        colorbar = plt.colorbar(scatter)
        ax.scatter(r_sun[:, 0] / sun_scale, r_sun[:, 1] / sun_scale, c=plt.cm.Oranges(np.linspace(0.25, 0.75, len(r_sat[:, 0]))), s=10)

        ax.set_title('Earth M_v')
        ax.axis('equal')

        ax = fig.add_subplot(1, 4, 4)
        ax.scatter(r_earth[:, 0], r_earth[:, 1], c='Blue', s=10)
        scatter = ax.scatter(r_sat[:, 0] / RGEO, r_sat[:, 1] / RGEO, c=(2.5 * np.log10((r_sun_sat / (10 * pc_to_m))**2) + sun_Mag) - 2.5 * np.log10(frac_flux_moon), cmap=color_map)
        ax.scatter(r_moon[:, 0] / RGEO, r_moon[:, 1] / RGEO, c=plt.cm.Greys(np.linspace(0.5, 1, len(r_sat[:, 0]))), s=5)

        colorbar = plt.colorbar(scatter)
        ax.set_title('Lunar M_v')
        ax.axis('equal')
        plt.show()

    return Mag_v


def megno(r):
    n_states = len(r)
    perturbed_states = r + 1e-8 * np.random.randn(n_states, 3)
    delta_states = perturbed_states - r
    delta_states_norm = np.linalg.norm(delta_states, axis=1)
    ln_delta_states_norm = np.log(delta_states_norm)

    megno_values = np.zeros(n_states)

    for i in range(1, n_states):
        m = np.mean(ln_delta_states_norm[:i])
        megno = (ln_delta_states_norm[i] + 2 * m) / (i)
        megno_values[i] = megno

    return np.mean(megno_values)


# orbital period from keplerian orbital elements (koe)
def orbital_period(a, mu_barycenter=3.986004418e14):
    return np.sqrt(4 * np.pi**2 / mu_barycenter * a**3) / 86400
