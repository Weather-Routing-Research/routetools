import numpy as np
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
    x, y = np.array(x), np.array(y)
    u_expected, v_expected = np.array(u_expected), np.array(v_expected)
    u, v = vectorfield_fourvortices(x, y, None)
    assert np.allclose(u, u_expected, rtol=1e-3)
    assert np.allclose(v, v_expected, rtol=1e-3)
