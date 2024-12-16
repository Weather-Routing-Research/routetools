import jax.numpy as jnp
import pytest

from routetools.cmaes import optimize
from routetools.vectorfield import vectorfield_fourvortices, vectorfield_techy


@pytest.mark.parametrize(
    "vectorfield, optimize_time",
    [
        (vectorfield_fourvortices, True),
        (vectorfield_fourvortices, False),
        (vectorfield_techy, True),
    ],
)
def test_cmaes(vectorfield: callable, optimize_time: bool):
    src = jnp.array([0, 0])
    dst = jnp.array([6, 2])

    curve = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        popsize=1000,
        sigma0=5,
        tolfun=1e-6,
    )
