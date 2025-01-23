import jax.numpy as jnp
import pytest

from routetools.cmaes import optimize
from routetools.land import Land
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
    L: int = 64,
):
    curve, cost = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        L=L,
        popsize=10,
        sigma0=5,
        tolfun=0.1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[0] == L
    assert curve.shape[1] == 2
    assert isinstance(cost, float)


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
    xlim = sorted((src[0], dst[0]))
    ylim = sorted((src[1], dst[1]))
    land = Land(xlim, ylim, random_seed=1, resolution=10)

    curve, cost = optimize(
        vectorfield,
        src=src,
        dst=dst,
        land=land,
        penalty=0.1,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        popsize=10,
        sigma0=5,
        tolfun=0.1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[1] == 2
    assert isinstance(cost, float)


@pytest.mark.parametrize(
    "vectorfield, src, dst, optimize_time, K, L, num_pieces",
    [
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
            True,
            7,
            61,
            2,
        ),
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
            False,
            13,
            64,
            3,
        ),
        (
            vectorfield_techy,
            jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)]),
            jnp.array([0, 1]),
            True,
            13,
            61,
            4,
        ),
    ],
)
def test_cmaes_piecewise(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    optimize_time: bool,
    K: int,
    L: int,
    num_pieces: int,
):
    curve, cost = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        K=K,
        L=L,
        num_pieces=num_pieces,
        popsize=10,
        sigma0=5,
        tolfun=0.1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[0] == L
    assert curve.shape[1] == 2
    assert isinstance(cost, float)
