import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from perlin_numpy import generate_perlin_noise_2d as pn2d


def generate_land(
    x: jnp.ndarray, y: jnp.ndarray, water_level: float = 0.4, resolution: jnp.ndarray = (1, 1), 
) -> jnp.ndarray:
    """
    Generate a 2D array representing land using Perlin noise.

    Parameters
    ----------
    x : jnp.ndarray
        array of x axis values
    y : jnp.ndarray
        array of y axis values
    water_level : float, optional
        water height that floods the noise, by default 0.4
    resolution : jnp.ndarray, optional
        resolution of the noise, or density of the land. Each entry must be divisors of the length of x and y respectively. Higher resolution generates more detailed land, by default (1, 1)

    Returns
    -------
    jnp.ndarray
        a 2D array of shape (len(x) by len(y)) representing land, where 0 is water and 1 is land
    """
    np.random.seed(0)
    land = jnp.array((pn2d((len(x), len(y)), res = resolution) > water_level).astype(int))
    return land

