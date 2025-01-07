import jax.numpy as jnp
import numpy as np
from perlin_numpy import generate_perlin_noise_2d as pn2d
from scipy.ndimage import map_coordinates


def generate_land_array(
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


def check_land_array(
    curve: jnp.ndarray, land_matrix: jnp.ndarray, limits: tuple[float] | None = None
) -> jnp.ndarray:
    """
    Check if points on a curve are on land using bilinear interpolation.

    :param curve: a batch of curves (an array of shape W x L x 2)
    :param land_matrix: a 2D boolean array indicating land (1) or water (0)
    :return: a boolean array of shape (W, L) indicating if each point is on land
    """
    # Extract x and y coordinates from the curve
    x_coords = curve[..., 0]
    y_coords = curve[..., 1]

    # Default interpolation assumes start at 0
    # When limits are provided, shift the coordinates to start at the limits
    if limits is not None:
        xmin, xmax, ymin, ymax = limits
        x_coords = (x_coords - xmin) / (xmax - xmin) * (land_matrix.shape[0] - 1)
        y_coords = (y_coords - ymin) / (ymax - ymin) * (land_matrix.shape[1] - 1)

    # Use bilinear interpolation to check if the points are on land
    land_values: jnp.ndarray
    land_values = map_coordinates(
        land_matrix, [x_coords, y_coords], order=1, mode="nearest"
    )

    # Return a boolean array where land_values > 0.3 indicates land
    return land_values > 0.1


def generate_land_function(
    x: jnp.ndarray,
    y: jnp.ndarray,
    water_level: float | None = None,
    resolution: tuple[int, int] | None = None,
    random_seed: int | None = None,
) -> callable[[jnp.ndarray], jnp.ndarray]:
    """
    Generate a function that checks if points on a curve are on land.

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
    callable
        a function that checks if points on a curve are on land
    """
    land_array = generate_land_array(
        x, y, water_level=water_level, resolution=resolution, random_seed=random_seed
    )
    limits = (jnp.min(x), jnp.max(x), jnp.min(y), jnp.max(y))

    def check_land(curve: jnp.ndarray) -> jnp.ndarray:
        return check_land_array(curve, land_array, limits=limits)

    return check_land
