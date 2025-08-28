import numpy as np


def dms_to_dd(dms):
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


def dd_to_dms(degree_decimal):
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
