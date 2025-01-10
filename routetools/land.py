import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from perlin_numpy import generate_perlin_noise_2d as pn2d


class Land:
    """Class to check if points on a curve are on land."""

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        water_level: float = 0.7,
        resolution: int | tuple[int, int] | None = None,
        random_seed: int | None = None,
    ):
        """Class to check if points on a curve are on land.

        Parameters
        ----------
        x : jnp.ndarray
            array of x axis values
        y : jnp.ndarray
            array of y axis values
        water_level : float, optional
            the threshold value to determine land from water, by default 0.7
        resolution : int | tuple, optional
            resolution of the noise, or density of the land. Each entry must be divisors
            of the length of x and y respectively. Higher resolution generates more
            detailed land, by default (1, 1)
        random_seed : int, optional
            random seed for reproducibility, by default None
        """
        # Ensure resolution is 2D
        if resolution is None:
            resolution = (1, 1)
        elif isinstance(resolution, int):
            resolution = (resolution, resolution)
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
                Resolution ({resolution[0]}) must be a divisor of
                the length of x ({len(x)})
                """
            )
        if len(y) % resolution[1] != 0:
            raise ValueError(
                f"""
                Resolution ({resolution[1]}) must be a divisor of
                the length of y ({len(y)})
                """
            )

        # Random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate land
        land = pn2d((len(x), len(y)), res=resolution)
        # Normalize land between 0 and 1
        land = (land - jnp.min(land)) / (jnp.max(land) - jnp.min(land))

        # Store the class properties
        self._array = jnp.array(land)
        self.x = x
        self.y = y
        self.xmin = x.min()
        self.xnorm = (self._array.shape[0] - 1) / (x.max() - x.min())
        self.ymin = y.min()
        self.ynorm = (self._array.shape[1] - 1) / (y.max() - y.min())
        self.resolution = resolution
        self.random_seed = random_seed
        self.water_level = water_level
        self.shape = self._array.shape

    @property
    def array(self) -> jnp.ndarray:
        """Return a boolean array indicating land presence."""
        return jnp.asarray((self._array >= self.water_level).astype(int))

    def __call__(
        self,
        curve: jnp.ndarray,
        interpolate: int = 0,
    ) -> jnp.ndarray:
        """
        Check if points on a curve are on land using bilinear interpolation.

        :param curve: a batch of curves (an array of shape W x L x 2)
        :return: a boolean array of shape (W, L) indicating if each point is on land
        """
        if interpolate > 0:
            # Interpolate x times to check if the curve passes through land
            curve_new = jnp.repeat(curve, interpolate + 1, axis=0)
            left = jnp.concatenate(
                [jnp.arange(interpolate + 2, 1, -1)] * (curve.shape[0] - 1)
            )
            right = jnp.concatenate(
                [jnp.arange(0, interpolate + 1, 1)] * (curve.shape[0] - 1)
            )
            left = curve_new[: -(interpolate + 1), :] * left[:, None]
            right = curve_new[(interpolate + 1) :, :] * right[:, None]
            interp = (left + right) / (interpolate + 2)
            curve_new = curve_new.at[: -(interpolate + 1)].set(interp)[:-interpolate, :]
        else:
            curve_new = curve

        # Extract x and y coordinates from the curve
        x_coords = curve_new[..., 0]
        y_coords = curve_new[..., 1]

        # Shift the coordinates to start at the limits
        x_coords = (x_coords - self.xmin) * self.xnorm
        y_coords = (y_coords - self.ymin) * self.ynorm

        # Use bilinear interpolation to check if the points are on land
        land_values = map_coordinates(
            self._array, [x_coords, y_coords], order=1, mode="nearest"
        )

        # Return a boolean array where land_values > 0 indicates land
        is_land = jnp.asarray(land_values >= self.water_level)

        if interpolate > 0:
            is_land = jnp.convolve(is_land, jnp.ones(interpolate + 1), mode="full")[
                :: interpolate + 1
            ]
        return is_land

    def penalization(
        self,
        curve: jnp.ndarray,
        penalty: float,
        interpolate: int = 100,
    ) -> jnp.ndarray:
        """
        Return an array indicating land presence, in one of two versions.

        (A) (no penalty) A boolean array indicating if the curve passes through land.
        (B) (penalty) the sum of the number of points on land times the penalty.

        Parameters
        ----------
        land_function : Callable[[jnp.ndarray], jnp.ndarray] | None, optional
            A function that checks if points on a curve are on land, by default None
        curve : jnp.ndarray
            A batch of curves (an array of shape W x L x 2)
        penalty : float
            The penalty for passing through land.
        interpolate : int, optional
            The number of times to interpolate the curve, by default 100
        """

        # Check if the curve passes through land
        def func(curve):
            return self.__call__(curve, interpolate=interpolate)

        is_land = jax.vmap(func)(curve)

        return jnp.sum(is_land, axis=1) * penalty
