import time

import cma
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit

from demo.vectorfield import vectorfield_swirlys


@jit
def batch_bezier(t: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate a batch of Bézier curves (using de Casteljau's algorithm).

    :param t: evaluation points (vector of shape K), all between 0 and 1
    :param control: batched matrix of control points, with shape B x P x N
    :return: batch of curves (matrix of shape B x K x N)
    """
    control = jnp.tile(control[:, :, None, :], [1, 1, len(t), 1])
    while control.shape[1] > 1:
        control = (1 - t[None, None, :, None]) * control[:, :-1, :, :] + t[
            None, None, :, None
        ] * control[:, 1:, :, :]
    return control[:, 0, ...]


def control_to_curve(
    control: jnp.ndarray, src: jnp.ndarray, dst: jnp.ndarray, L: int = 64
) -> jnp.ndarray:
    """
    Convert a batch of free parameters into a batch of Bézier curves.

    :param x: a B x 2K matrix. The first K columns are the x positions of the Bézier
    control points, and the last K are the y positions
    :return: curves, control points (both batched)..
    """
    control = control.reshape(control.shape[0], -1, 2)

    # Add the fixed endpoints
    first_point = jnp.broadcast_to(src, (control.shape[0], 1, 2))
    last_point = jnp.broadcast_to(dst, (control.shape[0], 1, 2))
    control = jnp.hstack([first_point, control, last_point])
    return batch_bezier(t=jnp.linspace(0, 1, L), control=control)


def cost_function(
    vectorfield: callable,
    curve: jnp.ndarray,
    travel_speed: float | None = None,
    travel_time: float | None = None,
) -> jnp.ndarray:
    """
    Compute the fuel consumption of a batch of paths navigating over a vector field.

    Two modes are supported:
    - Fixed travel speed (w.r.t. water)
    - Fixed travel time.

    :param vectorfield: a function that returns the horizontal and vertical components
    of the vector field.
    :param curve: batch of trajectories (an array of shape B x L x 2)
    :param travel_speed: the boat will have this fixed speed w.r.t. the water.
    If set, then `travel_time` must be None
    :param travel_time: the boat can regulate its speed but must complete
    each path in exactly this time. If set, then `travel_speed` must be None
    :param L: number of points evaluated in each Bézier curve
    :return: a batch of scalars (vector of shape B)
    """
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
    uinterp, vinterp = vectorfield(curvex, curvey)

    # Curve segments
    delta_x = jnp.diff(curve[:, :, 0], axis=1)
    delta_y = jnp.diff(curve[:, :, 1], axis=1)

    if travel_time is None:  # We navigate the path at fixed speed
        # Source: Zermelo's problem for constant wind
        # https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem#Constant-wind_case
        v2 = travel_speed**2
        w2 = uinterp**2 + vinterp**2
        dw = delta_x * uinterp + delta_y * vinterp
        d2 = delta_x**2 + delta_y**2  # Segment lengths
        cost = jnp.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
        cost = jnp.where(
            v2 <= w2, float("inf"), cost
        )  # Current > speed -> infeasible path
        total_cost = jnp.sum(cost, axis=1)
    else:  # We must navigate the path in a fixed time
        L = curve.shape[1]
        T = travel_time / (L - 1)
        cost = ((delta_x / T - uinterp) ** 2 + (delta_y / T - vinterp) ** 2) / 2
        total_cost = jnp.sum(cost, axis=1) * T
    return total_cost


def optimize(
    vectorfield: callable,
    src: jnp.ndarray,
    dst: jnp.ndarray,
    travel_speed: float | None = None,
    travel_time: float | None = None,
    K: int = 6,
    L: int = 64,
    popsize: int = 200,
    tolfun: float = 1e-4,
    verbose: bool = True,
    **kwargs,
) -> jnp.ndarray:
    """
    Solve the vessel routing problem for a given vector field.

    Two modes are supported:
        - Fixed speed-through-water. Optimize the vessel heading
        - Fixed total travel time. Optimize heading and speed-through-water

    Algorithm: parameterize the space of solutions with a Bézier curve,
    and optimize the control points using the CMA-ES optimization method.

    Parameters
    ----------
    vectorfield : callable
        A function that returns the horizontal and vertical components of the vector
    src : jnp.ndarray
        Source point (2D)
    dst : jnp.ndarray
        Destination point (2D)
    travel_speed : float, optional
        The boat will have this fixed speed w.r.t. the water.
        If set, then `travel_time` must be None. By default None
    travel_time : float, optional
        The boat can regulate its speed but must complete the path in exactly this time.
        If set, then `travel_speed` must be None
    K : int, optional
        Number of free Bézier control points. By default 6
    L : int, optional
        Number of points evaluated in each Bézier curve. By default 64
    popsize : int, optional
        Population size for the CMA-ES optimizer. By default 200
    tolfun : float, optional
        Tolerance for the optimizer. By default 1e-4
    verbose : bool, optional
        By default True
    **kwargs: optional additional parameters for CMA-ES

    Returns
    -------
    jnp.ndarray
        The optimized curve (shape L x 2)
    """
    ### Minimize
    start = time.time()

    x0 = np.linspace(src, dst, K).flatten()  # initial solution
    # initial standard deviation to sample new solutions
    sigma0 = np.linalg.norm(dst - src)
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, inopts={"popsize": popsize, "tolfun": tolfun} | kwargs
    )
    while not es.stop():
        X = es.ask()  # sample len(X) candidate solutions
        curve = control_to_curve(jnp.array(X), src, dst, L=L)
        cost = cost_function(vectorfield, curve, travel_speed, travel_time)
        es.tell(X, cost.tolist())  # update the optimizer
        if verbose:
            es.disp()
    if verbose:
        print("Optimization time:", time.time() - start)
        print("Fuel cost:", es.best.f)

    Xbest = es.best.x[None, :]
    curve_best = control_to_curve(Xbest, src, dst, L=L)[0, ...]
    return curve_best


def main():
    """
    Demonstrate usage of the optimization algorithm.

    The vector field is a superposition of four vortices.
    """
    src = np.array([0, 0])
    dst = np.array([6, 5])

    curve = optimize(
        vectorfield_swirlys,
        src=src,
        dst=dst,
        travel_speed=None,
        travel_time=30,
        popsize=1000,
        tolfun=1e-4,
    )

    xmin, xmax = curve[:, 0].min(), curve[:, 0].max()
    ymin, ymax = curve[:, 1].min(), curve[:, 1].max()

    x = np.arange(xmin, xmax, 0.5)
    y = np.arange(ymin, ymax, 0.5)
    X, Y = np.meshgrid(x, y)
    U, V = vectorfield_swirlys(X, Y)

    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.plot(curve[:, 0], curve[:, 1], color="red")
    plt.plot(src[0], src[1], "o", color="blue")
    plt.plot(dst[0], dst[1], "o", color="green")
    plt.savefig("output/demo.png")
    plt.close()


if __name__ == "__main__":
    main()
