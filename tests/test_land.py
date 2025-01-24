import jax.numpy as jnp
import pytest

from routetools.land import Land


def test_generate_land_array():
    xlim = [-5, 5]
    land = Land(xlim, xlim, random_seed=1, resolution=10)
    assert land.shape == (100, 100)
    assert land.array.max() == 1
    assert land.array.min() == 0


@pytest.mark.parametrize("water_level", [0, 1])
def test_water_level(water_level: float):
    xlim = [-5, 5]
    land = Land(xlim, xlim, water_level=water_level, random_seed=1)
    assert land.array.mean() == (1 - water_level)


def test_land_inbounds():
    xlim = [-5, 5]
    x = jnp.linspace(-5, 5, 100)
    # First generate the array
    land = Land(
        xlim, xlim, water_level=0.5, random_seed=1, resolution=10, interpolate=0
    )
    # Prepare a curve of (X, X) coordinates
    curve = jnp.stack([x, x], axis=-1)
    out = land(curve)
    expected = jnp.diag(land.array)
    # This curve should return the diagonal of the land array
    assert jnp.allclose(out, expected)


def test_land_outbounds():
    xlim = [-5, 5]
    # First generate the array
    land = Land(
        xlim, xlim, water_level=0.5, random_seed=1, resolution=10, interpolate=0
    )
    # A point outside the limits should return the closest
    out = land(jnp.array([[-6], [-5]]))
    expected = land.array[0, 0]
    assert jnp.allclose(out, expected)
    # Same in both bounds
    out = land(jnp.array([[6], [5]]))
    expected = land.array[-1, -1]
    assert jnp.allclose(out, expected)
