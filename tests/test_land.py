import jax.numpy as jnp

from routetools.land import generate_land_array, generate_land_function


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


def test_land_function():
    x = jnp.linspace(-5, 5, 100)
    # First generate the array
    land_array = generate_land_array(x, x, water_level=0.5, random_seed=1)
    # As we use the same random seed, the function should return the same array
    land_function = generate_land_function(x, x, water_level=0.5, random_seed=1)
    # Prepare a curve of (X, X) coordinates
    curve = jnp.stack([x, x], axis=-1)
    # This curve should return the diagonal of the land array
    assert jnp.all(land_function(curve) == jnp.diag(land_array))

    # A point outside the limits should return the closest
    out = land_function(jnp.array([[-6], [-5]]))
    assert jnp.all(out == land_array[0, 0])
    # Same in both bounds
    out = land_function(jnp.array([[6], [5]]))
    assert jnp.all(out == land_array[-1, -1])
