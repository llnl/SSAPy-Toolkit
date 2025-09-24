def accel_add(*accel_funcs):
    """
    Returns a function that sums multiple acceleration functions.
    
    Parameters
    ----------
    accel_funcs : list of functions
        Each must take (r) or (r, t) depending on your usage.
    
    Returns
    -------
    combined : function
        A function that evaluates and sums all input accelerations.
    """
    def combined(r, t=None):
        total = np.zeros(3)
        for f in accel_funcs:
            try:
                total += f(r, t)
            except TypeError:
                total += f(r)
        return total
    return combined
