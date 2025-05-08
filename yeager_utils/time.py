import numpy as np
from astropy.time import Time, TimeDelta
from astropy import units as u
from datetime import datetime
from typing import Union, List, Tuple


def now() -> str:
    """
    Returns the current time in the format 'YYYY-MM-DD HH:MM'.

    Returns
    -------
    str
        The current date and time as a formatted string.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def to_gps(t):
    if np.size(t) > 1:
        if isinstance(t[0], Time):
            try:
                t = t.gps
            except AttributeError:
                t = [time.iso for time in t]
                t = Time(t, format='iso').gps
    else:
        if isinstance(t, Time):
            t = t.gps
    return t


def _gpsToTT(t: float) -> float:
    """
    Convert GPS time in seconds to Terrestrial Time (TT) in days.

    Parameters
    ----------
    t : float
        GPS time in seconds.

    Returns
    -------
    float
        The corresponding TT time in days.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return 44244.0 + (t + 51.184) / 86400


def dms_to_dd(dms: Union[str, List[str]]) -> Union[float, List[float]]:
    """
    Convert Degree-Minute-Second (DMS) to Decimal Degrees (DD).

    Parameters
    ----------
    dms : str or list of str
        A string or a list of strings representing degrees, minutes, and seconds.

    Returns
    -------
    float or list of float
        The decimal degree equivalent(s) of the input DMS.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    dms, out = [[dms] if isinstance(dms, str) else dms][0], []
    for i in dms:
        deg, minute, sec = [float(j) for j in i.split(':')]
        if deg < 0:
            minute, sec = -minute, -sec
        out.append(deg + minute / 60 + sec / 3600)
    return out[0] if isinstance(dms, str) or len(dms) == 1 else out


def dd_to_dms(degree_decimal: float) -> str:
    """
    Convert Decimal Degrees (DD) to Degree-Minute-Second (DMS).

    Parameters
    ----------
    degree_decimal : float
        The decimal degree value.

    Returns
    -------
    str
        The corresponding DMS string in the format 'deg:min:sec'.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    _d, __d = np.trunc(degree_decimal), degree_decimal - np.trunc(degree_decimal)
    __d = -__d if degree_decimal < 0 else __d
    _m, __m = np.trunc(__d * 60), __d * 60 - np.trunc(__d * 60)
    _s = round(__m * 60, 4)
    _s = int(_s) if int(_s) == _s else _s
    if _s == 60:
        _m, _s = _m + 1, '00'
    elif _s > 60:
        _m, _s = _m + 1, _s - 60

    return f'{int(_d)}:{int(_m)}:{_s}'


def hms_to_dd(hms: Union[str, List[str]]) -> Union[float, List[float]]:
    """
    Convert Hour-Minute-Second (HMS) to Decimal Degrees (DD).

    Parameters
    ----------
    hms : str or list of str
        A string or a list of strings representing hours, minutes, and seconds.

    Returns
    -------
    float or list of float
        The decimal degree equivalent(s) of the input HMS.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    _type = type(hms)
    hms, out = [[hms] if _type == str else hms][0], []
    for i in hms:
        if i[0] != '-':
            hour, minute, sec = i.split(':')
            hour, minute, sec = float(hour), float(minute), float(sec)
            out.append(hour * 15 + (minute / 4) + (sec / 240))
        else:
            print('hms cannot be negative.')

    return out[0] if _type == str or len(hms) == 1 else out


def dd_to_hms(degree_decimal: float) -> str:
    """
    Convert Decimal Degrees (DD) to Hour-Minute-Second (HMS).

    Parameters
    ----------
    degree_decimal : float
        The decimal degree value.

    Returns
    -------
    str
        The corresponding HMS string in the format 'hour:minute:second'.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(degree_decimal, str):
        degree_decimal = dms_to_dd(degree_decimal)
    if degree_decimal < 0:
        print('dd for HMS conversion cannot be negative, assuming positive.')
        _dd = -degree_decimal / 15
    else:
        _dd = degree_decimal / 15
    _h, __h = np.trunc(_dd), _dd - np.trunc(_dd)
    _m, __m = np.trunc(__h * 60), __h * 60 - np.trunc(__h * 60)
    _s = round(__m * 60, 4)
    _s = int(_s) if int(_s) == _s else _s
    if _s == 60:
        _m, _s = _m + 1, '00'
    elif _s > 60:
        _m, _s = _m + 1, _s - 60

    return f'{int(_h)}:{int(_m)}:{_s}'


from typing import Union, Tuple
import numpy as np
from astropy.time import Time
import astropy.units as u

def get_times(duration: Union[int, Tuple[int, str]],
              freq: Tuple[int, str] = (1, 's'),
              t0: Union[str, Time] = "2025-01-01") -> np.ndarray:
    """
    Calculate a list of times spaced equally apart over a specified duration.

    Parameters
    ----------
    duration : int or tuple
        A duration in seconds (int), or a tuple like (30, 'day').
    freq : tuple
        A tuple containing the frequency value and its unit, default is (1, 's').
    t0 : str or Time, optional
        The starting time. Default is "2025-01-01".

    Returns
    -------
    np.ndarray
        A list of times spaced equally apart over the specified duration.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(t0, str):
        t0 = Time(t0, scale='utc')

    unit_dict = {
        'second': 1, 'sec': 1, 's': 1,
        'minute': 60, 'min': 60,
        'hour': 3600, 'hr': 3600, 'h': 3600,
        'day': 86400, 'd': 86400,
        'week': 604800,
        'month': 2630016, 'mo': 2630016,
        'year': 31557600, 'yr': 31557600
    }

    if isinstance(duration, int):
        dur_seconds = duration
    else:
        dur_val, dur_unit = duration
        dur_unit = dur_unit.lower().rstrip('s')
        if dur_unit not in unit_dict:
            raise ValueError(f'Error, {dur_unit} is not a valid time unit. Valid options are: {", ".join(unit_dict.keys())}.')
        dur_seconds = dur_val * unit_dict[dur_unit]

    freq_val, freq_unit = freq
    freq_unit = freq_unit.lower().rstrip('s')
    if freq_unit not in unit_dict:
        raise ValueError(f'Error, {freq_unit} is not a valid time unit. Valid options are: {", ".join(unit_dict.keys())}.')
    freq_seconds = freq_val * unit_dict[freq_unit]

    timesteps = int(dur_seconds / freq_seconds) + 1
    times = t0 + np.linspace(0, dur_seconds, timesteps) / unit_dict['day'] * u.day
    return times
