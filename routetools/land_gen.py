import jax.numpy as jnp
import numpy as np
from perlin_numpy import generate_perlin_noise_2d as pn2d


def generate_land(
    x: jnp.ndarray,
    y: jnp.ndarray,
    water_level: float | None = None,
    resolution: tuple[int, int] | None = None,
    random_seed: int | None = None,
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
    resolution : tuple, optional
        resolution of the noise, or density of the land. Each entry must be divisors of
        the length of x and y respectively. Higher resolution generates more detailed
        land, by default (1, 1)

    Returns
    -------
    jnp.ndarray
        a 2D array of shape (len(x) by len(y)) representing land, where 0 is water and
        1 is land
    """
    # Ensure resolution is 2D
    if resolution is None:
        resolution = (1, 1)
    elif len(resolution) != 2:
        raise ValueError(
            f"""
            Resolution must be a tuple of length 2, not {len(resolution)}
            """
        )

    # Ensure resolution is a divisor of x and y
    if len(x) % resolution[0] != 0:
        raise ValueError(
            f"""
            Resolution ({resolution[0]}) must be a divisor of the length of x ({len(x)})
            """
        )
    if len(y) % resolution[1] != 0:
        raise ValueError(
            f"""
            Resolution ({resolution[1]}) must be a divisor of the length of y ({len(y)})
            """
        )

    # Random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate land
    land = pn2d((len(x), len(y)), res=resolution)

    # Set water level
    if water_level is None:
        water_level = 0.75 * np.max(land.flatten(), axis=0).astype(float)

    # Return land mask
    return jnp.array((land > water_level).astype(int))
