import numpy as np
from astropy.time import Time
from astropy import units as u


def _gpsToTT(t):
    # Assume t is GPS seconds.  Convert to TT seconds by adding 51.184.
    # Divide by 86400 to get TT days.
    # Then add the TT time of the GPS epoch, expressed as an MJD, which
    # is 44244.0
    return 44244.0 + (t + 51.184) / 86400


def dms_to_dd(dms):  # Degree minute second to Degree decimal
    dms, out = [[dms] if type(dms) is str else dms][0], []
    for i in dms:
        deg, minute, sec = [float(j) for j in i.split(':')]
        if deg < 0:
            minute, sec = float(f'-{minute}'), float(f'-{sec}')
        out.append(deg + minute / 60 + sec / 3600)
    return [out[0] if type(dms) is str or len(dms) == 1 else out][0]


def dd_to_dms(degree_decimal):
    _d, __d = np.trunc(degree_decimal), degree_decimal - np.trunc(degree_decimal)
    __d = [-__d if degree_decimal < 0 else __d][0]
    _m, __m = np.trunc(__d * 60), __d * 60 - np.trunc(__d * 60)
    _s = round(__m * 60, 4)
    _s = [int(_s) if int(_s) == _s else _s][0]
    if _s == 60:
        _m, _s = _m + 1, '00'
    elif _s > 60:
        _m, _s = _m + 1, _s - 60

    return f'{int(_d)}:{int(_m)}:{_s}'


def hms_to_dd(hms):
    _type = type(hms)
    hms, out = [[hms] if _type == str else hms][0], []
    for i in hms:
        if i[0] != '-':
            hour, minute, sec = i.split(':')
            hour, minute, sec = float(hour), float(minute), float(sec)
            out.append(hour * 15 + (minute / 4) + (sec / 240))
        else:
            print('hms cannot be negative.')

    return [out[0] if _type == str or len(hms) == 1 else out][0]


def dd_to_hms(degree_decimal):
    if type(degree_decimal) is str:
        degree_decimal = dms_to_dd(degree_decimal)
    if degree_decimal < 0:
        print('dd for HMS conversion cannot be negative, assuming positive.')
        _dd = -degree_decimal / 15
    else:
        _dd = degree_decimal / 15
    _h, __h = np.trunc(_dd), _dd - np.trunc(_dd)
    _m, __m = np.trunc(__h * 60), __h * 60 - np.trunc(__h * 60)
    _s = round(__m * 60, 4)
    _s = [int(_s) if int(_s) == _s else _s][0]
    if _s == 60:
        _m, _s = _m + 1, '00'
    elif _s > 60:
        _m, _s = _m + 1, _s - 60

    return f'{int(_h)}:{int(_m)}:{_s}'


def get_times(duration, freq, t):
    """
    Calculate a list of times spaced equally apart over a specified duration.

    Parameters
    ----------
    duration: int
        The length of time to calculate times for.
    freq: int, unit: str
        frequency of time outputs in units provided
    t: ssapy.utils.Time, optional
        The starting time. Default is "2025-01-01".
    example input:
    duration=(30, 'day'), freq=(1, 'hr'), t=Time("2025-01-01", scale='utc')
    Returns
    -------
    times: array-like
        A list of times spaced equally apart over the specified duration.
    """
    if isinstance(t, str):
        t = Time(t, scale='utc')
    unit_dict = {'second': 1, 'sec': 1, 's': 1, 'minute': 60, 'min': 60, 'hour': 3600, 'hr': 3600, 'h': 3600, 'day': 86400, 'd': 86400, 'week': 604800, 'month': 2630016, 'mo': 2630016, 'year': 31557600, 'yr': 31557600}
    dur_val, dur_unit = duration
    freq_val, freq_unit = freq
    if dur_unit[-1] == 's' and len(dur_unit) > 1:
        dur_unit = dur_unit[:-1]
    if freq_unit[-1] == 's' and len(freq_unit) > 1:
        freq_unit = freq_unit[:-1]
    if dur_unit.lower() not in unit_dict:
        raise ValueError(f'Error, {dur_unit} is not a valid time unit. Valid options are: {", ".join(unit_dict.keys())}.')
    if freq_unit.lower() not in unit_dict:
        raise ValueError(f'Error, {freq_unit} is not a valid time unit. Valid options are: {", ".join(unit_dict.keys())}.')
    dur_seconds = dur_val * unit_dict[dur_unit.lower()]
    freq_seconds = freq_val * unit_dict[freq_unit.lower()]
    timesteps = int(dur_seconds / freq_seconds) + 1

    times = t + np.linspace(0, dur_seconds, timesteps) / unit_dict['day'] * u.day
    return times
