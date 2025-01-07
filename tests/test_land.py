import jax.numpy as jnp

from routetools.land import generate_land_array


def test_generate_land_array():
    land_array = generate_land_array(jnp.array([0, 1, 2, 3]), jnp.array([0, 1, 2, 3]))
    assert land_array.shape == (4, 4)
