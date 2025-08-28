from typing import Tuple
import numpy as np
from ..utils import divby0


def FFT(data: np.ndarray, time_between_samples: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a Fast Fourier Transform on the input data.

    Parameters:
    data (np.ndarray): The input time-series data.
    time_between_samples (float): Time interval between data samples.

    Returns:
    tuple: The frequency array and the FFT result.

    Author: Travis Yeager (yaeger7@llnl.gov)
    """
    N = len(data)
    k = int(N / 2)
    f = np.linspace(0.0, 1 / (2 * time_between_samples), N // 2)
    Y = np.abs(np.fft.fft(data))[:k]
    return f, Y


def FFTP(data: np.ndarray, time_between_samples: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a Fast Fourier Transform and calculate the period.

    Parameters:
    data (np.ndarray): The input time-series data.
    time_between_samples (float): Time interval between data samples.

    Returns:
    tuple: The period array and the FFT result.

    Author: Travis Yeager (yaeger7@llnl.gov)
    """
    N = len(data)
    k = int(N / 2)
    f = np.linspace(0.0, 1 / (2 * time_between_samples), N // 2)
    Tp = [divby0(1, float(item), len(data) * time_between_samples) for item in f]
    Y = np.abs(np.fft.fft(data))[:k]
    return Tp, Y
