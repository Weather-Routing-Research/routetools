import jax.numpy as jnp
import pytest

from routetools.cmaes import optimize
from routetools.land import Land
from routetools.vectorfield import (
    vectorfield_fourvortices,
    vectorfield_swirlys,
    vectorfield_techy,
)


@pytest.mark.parametrize(
    "vectorfield, src, dst, expected",
    [
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
            10.0,
        ),
        (
            vectorfield_techy,
            jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)]),
            jnp.array([0, 1]),
            1.04,
        ),
    ],
)
def test_cmaes_constant_speed(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    expected: float,
    L: int = 64,
):
    curve, dict_cmaes = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_stw=1,
        L=L,
        popsize=10,
        sigma0=1,
        seed=1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[0] == L
    assert curve.shape[1] == 2
    cost = dict_cmaes["cost"]
    assert isinstance(cost, float)
    assert cost <= expected, f"cost: {cost} > expected: {expected}"


@pytest.mark.parametrize(
    "vectorfield, src, dst, expected",
    [
        (
            vectorfield_swirlys,
            jnp.array([0, 0]),
            jnp.array([6, 5]),
            6.0,
        ),
    ],
)
def test_cmaes_constant_time(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    expected: float,
    L: int = 64,
):
    curve, dict_cmaes = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_time=30,
        L=L,
        popsize=1000,
        sigma0=2,
        seed=1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[0] == L
    assert curve.shape[1] == 2
    cost = dict_cmaes["cost"]
    assert isinstance(cost, float)
    assert cost <= expected, f"cost: {cost} > expected: {expected}"


@pytest.mark.parametrize(
    "vectorfield, src, dst",
    [
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
        ),
        (
            vectorfield_techy,
            jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)]),
            jnp.array([0, 1]),
        ),
    ],
)
def test_cmaes_constant_speed_with_land(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
):
    xlim = sorted((src[0], dst[0]))
    ylim = sorted((src[1], dst[1]))
    land = Land(xlim, ylim, random_seed=1, resolution=10)

    curve, dict_cmaes = optimize(
        vectorfield,
        src=src,
        dst=dst,
        land=land,
        penalty=0.1,
        travel_stw=1,
        popsize=10,
        sigma0=1,
        seed=1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[1] == 2
    assert isinstance(dict_cmaes["cost"], float)


@pytest.mark.parametrize(
    "vectorfield, src, dst, expected, K, L, num_pieces",
    [
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
            10.0,
            13,
            61,
            4,
        ),
        (
            vectorfield_techy,
            jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)]),
            jnp.array([0, 1]),
            1.04,
            7,
            61,
            2,
        ),
    ],
)
def test_cmaes_constant_speed_piecewise(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    expected: float,
    K: int,
    L: int,
    num_pieces: int,
):
    curve, dict_cmaes = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_stw=1,
        K=K,
        L=L,
        num_pieces=num_pieces,
        popsize=100,
        sigma0=5,
        tolfun=0.1,
        seed=1,
    )
    cost = dict_cmaes["cost"]
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[0] == L
    assert curve.shape[1] == 2
    assert isinstance(cost, float)
    assert cost <= expected, f"cost: {cost} > expected: {expected}"
