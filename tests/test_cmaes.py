import jax.numpy as jnp
import pytest

from routetools.cmaes import optimize
from routetools.vectorfield import vectorfield_fourvortices, vectorfield_techy


@pytest.mark.parametrize(
    "vectorfield, src, dst, optimize_time",
    [
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
            True,
        ),
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
            False,
        ),
        (
            vectorfield_techy,
            jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)]),
            jnp.array([0, 1]),
            True,
        ),
    ],
)
def test_cmaes(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    optimize_time: bool,
):
    optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        popsize=10,
        sigma0=5,
        tolfun=1e-6,
    )
