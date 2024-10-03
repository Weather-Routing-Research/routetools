from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer
from jax import grad, jacfwd, jacrev, jit, vmap

from routetools.cmaes import cost_function
from routetools.vectorfield import vectorfield_fourvortices


def hessian(
    f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], argnums: int = 0
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Compute the Hessian of a function.

    Parameters
    ----------
    f : Callable
        Function to differentiate.
    argnums : int, optional
        Argument number to differentiate, by default 0

    Returns
    -------
    Callable
        Hessian of the function.
    """
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)


def optimize_fms(
    vectorfield: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    src: jnp.ndarray | None = None,
    dst: jnp.ndarray | None = None,
    curve: jnp.ndarray | None = None,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    tolfun: float = 1e-4,
    damping: float = 0.9,
    verbose: bool = True,
    **kwargs: dict[str, Any],
) -> jnp.ndarray:
    """
    Optimize a curve using the FMS algorithm.

    Source:
    https://doi.org/10.1016/j.ifacol.2021.11.097

    Parameters
    ----------
    vectorfield : Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
        Vector field function.
    src : jnp.ndarray | None, optional
        Origin point, by default None
    dst : jnp.ndarray | None, optional
        Destination point, by default None
    curve : jnp.ndarray | None, optional
        Curve to optimize, shape L x 2, by default None
    travel_stw : float | None, optional
        Fixed speed through water, by default None
    travel_time : float | None, optional
        Fixed travel time, by default None
    tolfun : float, optional
        Tolerance for the cost reduction between epochs, by default 1e-4
    damping : float, optional
        Damping factor, by default 0.9
    verbose : bool, optional
        Print optimization progress, by default True

    Returns
    -------
    jnp.ndarray
        Optimized curve with shape L x 2
    """
    # Initialize solution
    if (src is not None) and (dst is not None):
        # Make a straight line between src and dst
        curve = jnp.linspace(src, dst, 200)
    elif curve is None:
        raise ValueError("Either src and dst or curve must be provided")
    assert curve.ndim == 2, "Input curve must have 2 dimensions (L x 2)"
    assert curve.shape[1] == 2, "Last dimension must be 2 (X, Y)"

    # Initialize lagrangians
    if travel_stw is not None:
        # Average distance between points
        d = jnp.mean(jnp.linalg.norm(curve[:, 1:] - curve[:, :-1], axis=-1))
        h = float(d / travel_stw)

        def lagrangian(q0: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:
            q0 = q0[None, None, :]
            q1 = q1[None, None, :]
            sog = (q1 - q0) / h
            l1 = cost_function(
                vectorfield, q0, sog=sog, travel_stw=travel_stw, travel_time=h
            )
            l2 = cost_function(
                vectorfield, q1, sog=sog, travel_stw=travel_stw, travel_time=h
            )
            ld = jnp.sum(h / 2 * (l1**2 + l2**2))
            return ld

    elif travel_time is not None:
        h = float(travel_time / curve.shape[0])

        def lagrangian(q0: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:
            q0 = q0[None, None, :]
            q1 = q1[None, None, :]
            sog = (q1 - q0) / h
            l1 = cost_function(vectorfield, q0, sog=sog, travel_time=h)
            l2 = cost_function(vectorfield, q1, sog=sog, travel_time=h)
            ld = jnp.sum(h / 2 * (l1 + l2))
            return ld

    else:
        raise ValueError("Either travel_stw or travel_time must be provided")

    d1ld = grad(lagrangian, argnums=0)
    d2ld = grad(lagrangian, argnums=1)
    d11ld = hessian(lagrangian, argnums=0)
    d22ld = hessian(lagrangian, argnums=1)

    @jit
    def jacobian(qkm1: jnp.ndarray, qk: jnp.ndarray, qkp1: jnp.ndarray) -> jnp.ndarray:
        b = -d2ld(qkm1, qk) - d1ld(qk, qkp1)
        a = d22ld(qkm1, qk) + d11ld(qk, qkp1)
        q: jnp.ndarray = jnp.linalg.solve(a, b)
        return q

    jac_vectorized = vmap(jacobian, in_axes=(0, 0, 0), out_axes=(0))

    @jit
    def solve_equation(curve: jnp.ndarray) -> jnp.ndarray:
        curve_new = jnp.copy(curve)
        q = jac_vectorized(curve[:-2], curve[1:-1], curve[2:])
        return curve_new.at[1:-1].set(damping * q + curve[1:-1])

    cost_now = cost_function(
        vectorfield,
        curve[None, ...],
        travel_stw=travel_stw,
        travel_time=travel_time,
    )[0]
    delta = jnp.inf

    # Loop iterations
    idx = 0
    while delta >= tolfun:
        cost_old = cost_now
        curve = solve_equation(curve)
        cost_now = cost_function(
            vectorfield,
            curve[None, ...],  # type: ignore[index]
            travel_stw=travel_stw,
            travel_time=travel_time,
        )[0]
        delta = 1 - cost_now / cost_old
        idx += 1
        if verbose and idx % 50 == 0:
            print(f"Iteration {idx}: cost = {cost_now:.3f} | delta = {delta:.3f}")

    return curve  # type: ignore[return-value]


def main(gpu: bool = True, optimize_time: bool = False) -> None:
    """
    Demonstrate usage of the optimization algorithm.

    The vector field is a superposition of four vortices.
    """
    if not gpu:
        jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]

    # Check if JAX is using the GPU
    print("JAX devices:", jax.devices())

    src = jnp.array([0, 0])
    dst = jnp.array([6, 2])

    curve = optimize_fms(
        vectorfield_fourvortices,
        src=src,
        dst=dst,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        tolfun=1e-6,
    )

    xmin, xmax = curve[:, 0].min(), curve[:, 0].max()
    ymin, ymax = curve[:, 1].min(), curve[:, 1].max()

    x: jnp.ndarray = jnp.arange(xmin, xmax, 0.5)
    y: jnp.ndarray = jnp.arange(ymin, ymax, 0.5)
    X, Y = jnp.meshgrid(x, y)
    U, V = vectorfield_fourvortices(X, Y)

    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.plot(curve[:, 0], curve[:, 1], color="red")
    plt.plot(src[0], src[1], "o", color="blue")
    plt.plot(dst[0], dst[1], "o", color="green")
    plt.savefig("output/demo.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
