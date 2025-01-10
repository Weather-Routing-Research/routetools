import jax.numpy as jnp

from routetools.land import Land


def test_generate_land_array():
    x = jnp.linspace(0, 10, 100)
    land = Land(x, x, random_seed=1)
    assert land.shape == (100, 100)
    assert land.array.max() == 1
    assert land.array.min() == 0


def test_water_level():
    x = jnp.linspace(0, 10, 100)
    # When the water level is at 0, the land array should be all land
    land = Land(x, x, water_level=0, random_seed=1)
    assert land.array.min() == 1
    # When the water level is over 1, the land array should be all water
    land = Land(x, x, water_level=1.1, random_seed=1)
    assert land.array.max() == 0


def test_land_function():
    x = jnp.linspace(-5, 5, 100)
    # First generate the array
    land = Land(x, x, water_level=0.5, random_seed=1)
    # Prepare a curve of (X, X) coordinates
    curve = jnp.stack([x, x], axis=-1)
    # This curve should return the diagonal of the land array
    assert jnp.all(land(curve) == jnp.diag(land.array))

    # A point outside the limits should return the closest
    out = land(jnp.array([[-6], [-5]]))
    assert jnp.all(out == land.array[0, 0])
    # Same in both bounds
    out = land(jnp.array([[6], [5]]))
    assert jnp.all(out == land.array[-1, -1])
