import numpy as np


def hms_to_dd(hms):
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


def dd_to_hms(degree_decimal):
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
