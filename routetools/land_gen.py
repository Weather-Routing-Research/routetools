import jax.numpy as jnp
import numpy as np
from perlin_numpy import generate_perlin_noise_2d as pn2d


def generate_land(
    x: jnp.ndarray,
    y: jnp.ndarray,
    water_level: float | None = None,
    resolution: jnp.ndarray | None = None,
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
        resolution of the noise, or density of the land. Each entry must be divisors of
        the length of x and y respectively. Higher resolution generates more detailed
        land, by default (1, 1)

    Returns
    -------
    jnp.ndarray
        a 2D array of shape (len(x) by len(y)) representing land, where 0 is water and
        1 is land
    """
    # Ensure jnp array
    resolution = jnp.array([1, 1]) if resolution is None else resolution
    resolution = jnp.asarray(resolution)

    # np.random.seed(0)
    assert (
        len(x) % resolution[0] == 0
    ), "Resolution must be a divisor of the length of x"
    assert (
        len(y) % resolution[1] == 0
    ), "Resolution must be a divisor of the length of y"

    land = pn2d((len(x), len(y)), res=resolution)
    if water_level is None:
        water_level = 0.75 * np.max(land.flatten(), axis=0).astype(float)
    print(water_level)
    return jnp.array((land > water_level).astype(int))
