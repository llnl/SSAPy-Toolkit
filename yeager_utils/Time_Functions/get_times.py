import numpy as np
from astropy.time import Time
import astropy.units as u


def get_times(duration,
              freq=(1, 's'),
              t0=Time(0, format='gps')):
    """
    Calculate a list of times spaced equally apart over a specified duration.

    Parameters
    ----------
    duration : int, float, or tuple
        A duration in seconds (float/int), or a tuple like (30, 'day').
    freq : int, float, or tuple
        A frequency in seconds (float/int), or a tuple like (10, 'min').
    t0 : str, float, int, or Time, optional
        The starting time. If str, interpreted as UTC.
        If numeric, interpreted as GPS seconds. Default is Time(0, format='gps').

    Returns
    -------
    np.ndarray
        A list of times spaced equally apart over the specified duration.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(t0, str):
        t0 = Time(t0, scale='utc')
    if isinstance(t0, (float, int)):
        t0 = Time(t0, scale='utc', format='gps')

    unit_dict = {
        'second': 1, 'sec': 1, 's': 1,
        'minute': 60, 'min': 60,
        'hour': 3600, 'hr': 3600, 'h': 3600,
        'day': 86400, 'd': 86400,
        'week': 604800,
        'month': 2630016, 'mo': 2630016,
        'year': 31557600, 'yr': 31557600
    }

    def to_seconds(value):
        if isinstance(value, (int, float)):
            return float(value)
        else:
            val, unit = value
            unit = unit.lower()
            if len(unit) > 1:
                unit = unit.rstrip('s')

            if unit not in unit_dict:
                raise ValueError(
                    f'Error, {unit} is not a valid time unit. '
                    f'Valid options are: {", ".join(unit_dict.keys())}.'
                )
            return float(val) * unit_dict[unit]

    dur_seconds = to_seconds(duration)
    freq_seconds = to_seconds(freq)

    timesteps = int(dur_seconds / freq_seconds) + 1
    times = t0 + np.linspace(0, dur_seconds, timesteps) / unit_dict['day'] * u.day
    return times
