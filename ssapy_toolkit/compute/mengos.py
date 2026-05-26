import numpy as np


def megno(r: np.ndarray) -> float:
    """
    Calculate the MEGNO (Mean Exponential Growth of Nearby Orbits) value for a set of orbital states.

    The MEGNO is a measure of the chaos in the orbital evolution. It quantifies the exponential 
    divergence of nearby trajectories over time, used to detect chaotic regions in orbital dynamics.

    Parameters:
    - r: A 2D numpy array of shape (n_states, 3) representing the initial positions of the orbital states 
         in 3D space (x, y, z).

    Returns:
    - A float representing the mean MEGNO value for the given orbital states.
    """
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
