import numpy as np
from random import shuffle


def radius_from_H_albedo(H, albedo=.1):
    ######################################################################
    # ast radius from albedo and H mag
    ######################################################################
    radius = 1329e3 / (2 * np.sqrt(albedo)) * 10 ** (-0.2 * H)  # http://www.physics.sfasu.edu/astro/asteroids/sizemagnitude.html
    return radius


def H_mag(radius, albedo):
    return 5 * np.log(664500 / (radius * np.sqrt(albedo))) / np.log(10)


# Color correction, Johnson V to LSST Filters
def johnsonV_to_lsst_array(M_app, filters, ast_types):
    corrections = np.zeros(np.shape(M_app))
    corrections[np.where((np.array(filters) == 'u') & (np.array(ast_types) == 0))] = -1.614
    corrections[np.where((np.array(filters) == 'u') & (np.array(ast_types) == 1))] = -1.927
    corrections[np.where((np.array(filters) == 'g') & (np.array(ast_types) == 0))] = -0.302
    corrections[np.where((np.array(filters) == 'g') & (np.array(ast_types) == 1))] = -0.395
    corrections[np.where((np.array(filters) == 'r') & (np.array(ast_types) == 0))] = 0.172
    corrections[np.where((np.array(filters) == 'r') & (np.array(ast_types) == 1))] = 0.255
    corrections[np.where((np.array(filters) == 'i') & (np.array(ast_types) == 0))] = 0.291
    corrections[np.where((np.array(filters) == 'i') & (np.array(ast_types) == 1))] = 0.455
    corrections[np.where((np.array(filters) == 'z') & (np.array(ast_types) == 0))] = 0.298
    corrections[np.where((np.array(filters) == 'z') & (np.array(ast_types) == 1))] = 0.401
    corrections[np.where((np.array(filters) == 'y') & (np.array(ast_types) == 0))] = 0.303
    corrections[np.where((np.array(filters) == 'y') & (np.array(ast_types) == 1))] = 0.406
    return M_app - corrections


def johnsonV_to_ztf_array(M_app, filters, ast_types):
    corrections = np.zeros(np.shape(M_app))
    corrections[np.where((np.array(filters) == 1) & (np.array(ast_types) == 0))] = -0.302
    corrections[np.where((np.array(filters) == 1) & (np.array(ast_types) == 1))] = -0.395
    corrections[np.where((np.array(filters) == 2) & (np.array(ast_types) == 0))] = 0.172
    corrections[np.where((np.array(filters) == 2) & (np.array(ast_types) == 1))] = 0.255
    corrections[np.where((np.array(filters) == 3) & (np.array(ast_types) == 0))] = 0.291
    corrections[np.where((np.array(filters) == 3) & (np.array(ast_types) == 1))] = 0.455
    return M_app - corrections


######################################################################
# get ETA albedo -- > P2R = fd * (pv*np.exp(-pv**2/(2*d**2))/d**2) + (1-fd)*(pv*np.exp(-pv**2/(2*b**2))/b**2)
######################################################################
def get_albedo_array(num=1):
    num = int(num)
    albedo_out = []
    ast_type_out = []
    while np.size(albedo_out) != num:
        fd = 0.253
        d = 0.030
        b = 0.168
        albedo = np.random.uniform(0, 1, size=num)
        # Albedos from NEO population - https://iopscience.iop.org/article/10.3847/0004-6256/152/4/79
        sample_ys = np.random.uniform(0, 6, size=num)
        c_type = fd * (albedo * np.exp(-albedo**2 / (2 * d**2)) / d**2)
        s_type = (1 - fd) * (albedo * np.exp(-albedo**2 / (2 * b**2)) / b**2)
        c_albedo = albedo[np.where(sample_ys < c_type)]
        s_albedo = albedo[np.where(sample_ys < s_type)]
        c_type = np.zeros(np.size(c_albedo))
        s_type = np.ones(np.size(s_albedo))
        albedo = np.concatenate((c_albedo, s_albedo))
        ast_type = np.concatenate((c_type, s_type))
        if np.size(albedo_out) == 0:
            albedo_out = albedo
            ast_type_out = ast_type
            continue
        albedo_out = np.hstack((albedo_out, albedo))
        ast_type_out = np.hstack((ast_type_out, ast_type))
        if np.size(albedo_out) > num:
            # Shuffle the last arrays so there's no biasing towards c_type
            temp = list(zip(albedo, ast_type))
            np.random.shuffle(temp)
            albedo, ast_type = zip(*temp)
            albedo_out = albedo_out[:num]
            ast_type_out = ast_type_out[:num]
    return (albedo_out, ast_type_out)


def granvik_low_slope(x):
    return 0.3034 * x - 3.491


def granvik_high_slope(x):
    return 0.7235 * x - 13.12


def get_neo_H_mag_array(num=1, upper_mag=28, min_mag=10):
    num = int(num)
    H_mag_out = []
    # Extending granvik H mags
    # break_point = 23
    while np.size(H_mag_out) != num:
        xs = np.random.uniform(min_mag, upper_mag, size=num)
        ys = np.random.uniform(1, 10**granvik_high_slope(upper_mag), size=num)
        xs_low = xs[np.where(xs < 23)]
        ys_low = ys[np.where(xs < 23)]
        xs_high = xs[np.where(xs >= 23)]
        ys_high = ys[np.where(xs >= 23)]
        index_low = np.where(ys_low < 10**granvik_low_slope(xs_low))
        index_high = np.where(ys_high < 10**granvik_high_slope(xs_high))
        H_mag = np.hstack((xs_low[index_low], xs_high[index_high]))
        if np.size(H_mag_out) == 0:
            H_mag_out = H_mag
            continue
        H_mag_out = np.hstack((H_mag_out, H_mag))
        if np.size(H_mag_out) > num:
            H_mag_out = H_mag_out[:num]
    shuffle(H_mag_out)
    return H_mag_out


######################################################################
# get ETA diameter
######################################################################
def get_eta_radius_albedo_H_array(num=1, upper_mag=28, min_mag=10):
    albedo, ast_type = get_albedo_array(num=num)
    H = get_neo_H_mag_array(num=num, upper_mag=upper_mag, min_mag=min_mag)
    radius = 1329e3 / (2 * np.sqrt(albedo)) * 10**(-0.2 * H)
    return {'radius': radius, 'albedo': albedo, 'type': ast_type, 'H': H}
