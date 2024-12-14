import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from perlin_numpy import generate_perlin_noise_2d as pn2d


def generate_land(
    x: jnp.ndarray, y: jnp.ndarray, threshold: float = 0, resolution: int = 1
) -> jnp.ndarray:
    """
    Generate land.
    """
    land = jnp.array((-pn2d((len(x), len(y)), (resolution, resolution)) > threshold).astype(int))
    return land

