import jax.numpy as jnp
import pytest

from routetools.cmaes import optimize
from routetools.land import generate_land_function
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
def test_cmaes_with_land(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    optimize_time: bool,
):
    xlim = (src[0], dst[0])
    ylim = (src[1], dst[1])
    x = jnp.linspace(*xlim, 100)
    y = jnp.linspace(*ylim, 100)
    land_function = generate_land_function(x, y, random_seed=1)

    optimize(
        vectorfield,
        src=src,
        dst=dst,
        land_function=land_function,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        popsize=10,
        sigma0=5,
        tolfun=1e-6,
    )
