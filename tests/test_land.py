import jax.numpy as jnp

from routetools.land import generate_land_array


def test_generate_land_array():
    x = jnp.linspace(0, 10, 100)
    land_array = generate_land_array(x, x, random_seed=1)
    assert land_array.shape == (100, 100)
    assert land_array.max() == 1
    assert land_array.min() == 0


def test_water_level():
    x = jnp.linspace(0, 10, 100)
    # When the water level is at 0, the land array should be all land
    land_array = generate_land_array(x, x, water_level=0, random_seed=1)
    assert land_array.min() == 1
    # When the water level is over 1, the land array should be all water
    land_array = generate_land_array(x, x, water_level=1.1, random_seed=1)
    assert land_array.max() == 0
