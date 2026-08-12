from psutil import Process
import os


def get_memory_usage() -> str:
    """
    Get the current memory usage of the process in GB.

    Returns
    -------
    str
        Formatted memory used by the process in gigabytes.
    """
    memory_used = Process(os.getpid()).memory_info().rss / 1024 ** 3
    return f"Memory used: {memory_used:.2f} GB"
