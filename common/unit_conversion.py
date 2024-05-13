import numpy as np


def trj2tcmb(freq):
    x = 6.62607015e-34 * freq * 1e9 / 2.725 / 1.380649e-23
    return (np.exp(x) - 1)**2 / x**2 / np.exp(x)


def tcmb2trj(freq):
    x = 6.62607015e-34 * freq * 1e9 / 2.725 / 1.380649e-23
    return x**2 * np.exp(x) / (np.exp(x) - 1)**2