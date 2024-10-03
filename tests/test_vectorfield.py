import jax.numpy as jnp
import pytest

from routetools.vectorfield import vectorfield_fourvortices


@pytest.mark.parametrize(
    "x, y, u_expected,v_expected",
    [
        (0, 0, -0.2812, 0.1371),
        (1, 1, -0.4663, 0.2295),
    ],
)
def test_vectorfield_fourvortices(
    x: float, y: float, u_expected: float, v_expected: float
):
    """
    Test the vectorfield_fourvortices function.
    """
    x, y = jnp.array(x), jnp.array(y)
    u_expected, v_expected = jnp.array(u_expected), jnp.array(v_expected)
    u, v = vectorfield_fourvortices(x, y)
    assert jnp.allclose(u, u_expected, rtol=1e-3)
    assert jnp.allclose(v, v_expected, rtol=1e-3)
