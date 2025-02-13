import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import cma
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer
from jax import jit

from routetools.land import Land
from routetools.vectorfield import vectorfield_fourvortices


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


@partial(jit, static_argnums=(3, 4))
def control_to_curve(
    control: jnp.ndarray,
    src: jnp.ndarray,
    dst: jnp.ndarray,
    L: int = 64,
    num_pieces: int = 1,
) -> jnp.ndarray:
    """
    Convert a batch of free parameters into a batch of Bézier curves.

    Parameters
    ----------
    control : jnp.ndarray
        A B x 2K matrix. The first K columns are the x positions of the Bézier
        control points, and the last K are the y positions
    src : jnp.ndarray
        Source point (2D)
    dst : jnp.ndarray
        Destination point (2D)
    L : int, optional
        Number of points evaluated in each Bézier curve, by default 64
    num_pieces : int, optional
        Number of Bézier curves, by default 1
    """
    control = control.reshape(control.shape[0], -1, 2)

    # Get the bounding box of the control points
    bbox = jnp.vstack([src, dst])
    bbmin, bbmax = jnp.sort(bbox, axis=0)

    # Control points are in the range [-1, 1]. Scale them to the source and destination
    control = (control + 1) / 2 * (bbmax - bbmin) + bbmin

    # Add the fixed endpoints
    first_point = jnp.broadcast_to(src, (control.shape[0], 1, 2))
    last_point = jnp.broadcast_to(dst, (control.shape[0], 1, 2))
    control = jnp.hstack([first_point, control, last_point])

    result: jnp.ndarray
    if num_pieces > 1:
        # Ensure that the number of control points is divisible by the number of pieces
        control_per_piece = (control.shape[1] - 1) / num_pieces
        if control_per_piece < 3:
            raise ValueError(
                "The number of control points - 1 must be at least 3 per piece. "
                f"Got {control.shape[1]} control points and {num_pieces} pieces."
            )
        elif int(control_per_piece) != control_per_piece:
            control_rec = int(control_per_piece) * num_pieces + 1
            raise ValueError(
                "The number of control points must be divisible by num_pieces. "
                f"Got {control.shape[1]} control points and {num_pieces} pieces."
                f"Consider using {control_rec} control points."
            )
        else:
            control_per_piece = int(control_per_piece)
        # Ensure the number of waypoints is divisible by the number of pieces
        waypoints_per_piece = (L - 1) / num_pieces
        if int(waypoints_per_piece) != waypoints_per_piece:
            L_rec = int(waypoints_per_piece) * num_pieces + 1
            raise ValueError(
                "The number of waypoints - 1 must be divisible by num_pieces. "
                f"Got {L} waypoints and {num_pieces} pieces. "
                f"Consider using {L_rec} waypoints."
            )
        else:
            waypoints_per_piece = int(waypoints_per_piece) + 1

        # Split the control points into pieces
        ls_pieces: list[jnp.ndarray] = []
        for i in range(num_pieces):
            start = i * control_per_piece
            end = (i + 1) * control_per_piece + 1
            piece: jnp.ndarray = batch_bezier(
                t=jnp.linspace(0, 1, waypoints_per_piece),
                control=control[:, start:end, :],
            )[:, :-1]
            # The last point of each piece is omitted to avoid duplicates
            ls_pieces.append(piece)
        # Concatenate the pieces into a single curve
        result = jnp.hstack(ls_pieces)
        # Add the destination (last) point (was omitted in the loop)
        result = jnp.hstack([result, last_point])
    else:
        result = batch_bezier(t=jnp.linspace(0, 1, L), control=control)
    return result


@partial(jit, static_argnums=(0, 3, 4))
def cost_function(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    sog: jnp.ndarray | None = None,
    travel_stw: float | None = None,
    travel_time: float | None = None,
) -> jnp.ndarray:
    """
    Compute the fuel consumption of a batch of paths navigating over a vector field.

    Two modes are supported:
    - Fixed travel speed through water (STW)
    - Fixed travel time.

    :param vectorfield: a function that returns the horizontal and vertical components
    of the vector field.
    :param curve: batch of trajectories (an array of shape B x L x 2).
    :param sog: batch of speeds over ground, SOG (an array of shape B x L-1 x 2)
    :param travel_stw: the boat will have this fixed speed through water, STW.
    :param travel_time: When the curve is a single point, this is the time delta. Else,
    the boat can regulate its STW but must complete each path in exactly this time.
    :param L: number of points evaluated in each Bézier curve
    :return: a batch of scalars (vector of shape B)
    """
    if curve.shape[1] > 1:
        # When the curve is defined by more than one point,
        # we will interpolate the vector field at the midpoints
        curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
        curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
        # We can also compute the distances between points
        dx = jnp.diff(curve[:, :, 0], axis=1)
        dy = jnp.diff(curve[:, :, 1], axis=1)
        if travel_time is not None:
            # The times between points are only relevant when travel_time is set
            dt = travel_time / (curve.shape[1] - 1)
            # If the SOG is not provided, but we need to compute the cost at constant
            # time, we compute the speed over ground from the displacement
            dxdt = dx / dt
            dydt = dy / dt
        else:
            dt = 0  # Time between points is irrelevant
    elif (sog is not None) and (travel_time is not None):
        # If the curve is defined by a single point,
        # we will take the vector field at that point
        curvex = curve[:, :, 0]
        curvey = curve[:, :, 1]
        # The time between points is the total travel time
        dt = travel_time
        # We can compute the displacements over ground
        dxdt = sog[:, :, 0]
        dydt = sog[:, :, 1]
        dx = dxdt * dt
        dy = dydt * dt
    else:
        raise ValueError(
            "When curve has only one point, SOG and travel_time must be provided"
        )

    # Compute the time at each point
    curvet = jnp.arange(curve.shape[1] - 1, dtype=float) * dt
    # Interpolate the vector field at the midpoints
    uinterp, vinterp = vectorfield(curvex, curvey, curvet[jnp.newaxis, :])

    if travel_stw is not None:
        # We navigate the path at fixed speed through water (STW)
        # Source: Zermelo's problem for constant wind
        # https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem#Constant-wind_case
        v2 = travel_stw**2
        w2 = uinterp**2 + vinterp**2
        dw = dx * uinterp + dy * vinterp
        d2 = dx**2 + dy**2  # Segment lengths
        cost = jnp.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
        cost = jnp.where(
            v2 <= w2, float("inf"), cost
        )  # Current > speed -> infeasible path
        total_cost = jnp.sum(cost, axis=1)
    elif travel_time is not None:
        # We must navigate the path in a fixed time
        cost = ((dxdt - uinterp) ** 2 + (dydt - vinterp) ** 2) / 2
        total_cost = jnp.sum(cost, axis=1) * dt
        # TODO: JIT works for static currents. When the vectorfield changes in
        # time, you need to approximate the cost using Newton-Raphson.
    else:
        raise ValueError("travel_stw must be provided when travel_time is None")

    # Turn any possible infinite costs into 10x the highest value
    total_cost = jnp.where(jnp.isinf(total_cost), jnp.nan, total_cost)
    total_cost = jnp.nan_to_num(
        total_cost, nan=jnp.nanmax(total_cost, initial=1e10) * 10
    )
    return total_cost


def _cma_evolution_strategy(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    src: jnp.ndarray,
    dst: jnp.ndarray,
    x0: jnp.ndarray,
    land: Land | None = None,
    penalty: float = 10,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    L: int = 64,
    num_pieces: int = 1,
    popsize: int = 200,
    sigma0: float | None = None,
    tolfun: float = 1e-4,
    seed: float = jnp.nan,
    verbose: bool = True,
    **kwargs: dict[str, Any],
) -> cma.CMAEvolutionStrategy:
    curve: jnp.ndarray
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, inopts={"popsize": popsize, "tolfun": tolfun, "seed": seed} | kwargs
    )
    if land is not None:
        assert penalty is not None, "penalty must be a number"

    while not es.stop():
        X = es.ask()  # sample len(X) candidate solutions
        curve = control_to_curve(jnp.array(X), src, dst, L=L, num_pieces=num_pieces)

        cost: jnp.ndarray = cost_function(
            vectorfield,
            curve,
            travel_stw=travel_stw,
            travel_time=travel_time,
        )

        # Land penalization
        if land is not None:
            cost += land.penalization(curve, penalty=penalty)

        es.tell(X, cost.tolist())  # update the optimizer
        if verbose:
            es.disp()
    return es


def optimize(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    src: jnp.ndarray,
    dst: jnp.ndarray,
    land: Land | None = None,
    penalty: float = 10,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    K: int = 6,
    L: int = 64,
    num_pieces: int = 1,
    popsize: int = 200,
    sigma0: float | None = None,
    tolfun: float = 1e-4,
    seed: float = jnp.nan,
    verbose: bool = True,
) -> tuple[jnp.ndarray, float]:
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
    land_function : callable, optional
        A function that checks if points on a curve are on land, by default None
    penalty : float, optional
        Penalty for land points, by default 10
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW).
        If set, then `travel_time` must be None. By default None
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this time.
        If set, then `travel_stw` must be None
    K : int, optional
        Number of free Bézier control points. By default 6
    L : int, optional
        Number of points evaluated in each Bézier curve. By default 64
    popsize : int, optional
        Population size for the CMA-ES optimizer. By default 200
    sigma0 : float, optional
        Initial standard deviation to sample new solutions. By default None
    tolfun : float, optional
        Tolerance for the optimizer. By default 1e-4
    seed : int, optional
        Random seed for reproducibility. By default jnp.nan
    verbose : bool, optional
        By default True

    Returns
    -------
    tuple[jnp.ndarray, float]
        The optimized curve (shape L x 2), and the fuel cost
    """
    if K < 3:
        raise ValueError("The number of control points must be at least 3")
    # Initial solution as a straight line
    x0 = jnp.linspace(src, dst, K - 2)
    # Scale it to the range [-1, 1]
    bbox = jnp.vstack([src, dst])
    bbmin, bbmax = jnp.sort(bbox, axis=0)
    x0 = 2 * (x0 - bbmin) / (bbmax - bbmin) - 1
    x0 = x0.flatten()
    # initial standard deviation to sample new solutions
    sigma0 = float(jnp.linalg.norm(dst - src)) if sigma0 is None else float(sigma0)

    start = time.time()
    es = _cma_evolution_strategy(
        vectorfield=vectorfield,
        src=src,
        dst=dst,
        x0=x0,
        land=land,
        penalty=penalty,
        travel_stw=travel_stw,
        travel_time=travel_time,
        L=L,
        num_pieces=num_pieces,
        popsize=popsize,
        sigma0=sigma0,
        tolfun=tolfun,
        seed=seed,
        verbose=verbose,
    )
    if verbose:
        print("Optimization time:", time.time() - start)
        print("Fuel cost:", es.best.f)

    Xbest = es.best.x[None, :]
    curve_best: jnp.ndarray = control_to_curve(
        Xbest, src, dst, L=L, num_pieces=num_pieces
    )[0, ...]
    return curve_best, es.best.f


def optimize_with_increasing_penalization(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    src: jnp.ndarray,
    dst: jnp.ndarray,
    land: Land,
    penalty_init: float = 0,
    penalty_increment: float = 10,
    maxiter: int = 10,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    K: int = 6,
    L: int = 64,
    num_pieces: int = 1,
    popsize: int = 200,
    sigma0: float | None = None,
    tolfun: float = 1e-4,
    seed: float = jnp.nan,
    verbose: bool = True,
) -> tuple[list[jnp.ndarray], list[float]]:
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
    land_function : callable
        A function that checks if points on a curve are on land
    penalty_init : float, optional
        Initial penalty for land points, by default 0
    penalty_increment : float, optional
        Increment in the penalty for land points. By default 10
    maxiter : int, optional
        Maximum number of iterations. By default 10
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW).
        If set, then `travel_time` must be None. By default None
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this time.
        If set, then `travel_stw` must be None
    K : int, optional
        Number of free Bézier control points. By default 6
    L : int, optional
        Number of points evaluated in each Bézier curve. By default 64
    popsize : int, optional
        Population size for the CMA-ES optimizer. By default 200
    sigma0 : float, optional
        Initial standard deviation to sample new solutions. By default None
    tolfun : float, optional
        Tolerance for the optimizer. By default 1e-4
    seed : int, optional
        Random seed for reproducibility. By default jnp.nan
    verbose : bool, optional
        By default True

    Returns
    -------
    tuple[list[jnp.ndarray], list[float]]
        The list of optimized curves (each with shape L x 2), and the list of fuel costs
    """
    if K < 3:
        raise ValueError("The number of control points must be at least 3")
    # Initial solution as a straight line
    x0 = jnp.linspace(src, dst, K - 2)
    # Scale it to the range [-1, 1]
    bbox = jnp.vstack([src, dst])
    bbmin, bbmax = jnp.sort(bbox, axis=0)
    x0 = 2 * (x0 - bbmin) / (bbmax - bbmin) - 1
    x0 = x0.flatten()
    # initial standard deviation to sample new solutions
    sigma0 = float(jnp.linalg.norm(dst - src)) if sigma0 is None else float(sigma0)

    # Initializations
    penalty = penalty_init
    is_land = True
    niter = 1
    ls_curve = []
    ls_cost = []

    start = time.time()
    while is_land and (niter < maxiter):
        es = _cma_evolution_strategy(
            vectorfield=vectorfield,
            src=src,
            dst=dst,
            x0=x0,
            land=land,
            penalty=penalty,
            travel_stw=travel_stw,
            travel_time=travel_time,
            L=L,
            num_pieces=num_pieces,
            popsize=popsize,
            sigma0=sigma0,
            tolfun=tolfun,
            seed=seed,
            verbose=verbose,
        )
        if verbose:
            print("Optimization time:", time.time() - start)
            print("Fuel cost:", es.best.f)

        Xbest = es.best.x[None, :]
        curve: jnp.ndarray = control_to_curve(
            Xbest, src, dst, L=L, num_pieces=num_pieces
        )[0, ...]
        # sigma0 = es.sigma0
        if land(curve).any():
            penalty += penalty_increment
            x0 = es.best.x
        else:
            is_land = False

        niter += 1
        ls_curve.append(curve)
        ls_cost.append(es.best.f)

    return ls_curve, ls_cost


def main(gpu: bool = True, optimize_time: bool = False) -> None:
    """
    Demonstrate usage of the optimization algorithm.

    The vector field is a superposition of four vortices.
    """
    if not gpu:
        jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]

    # Check if JAX is using the GPU
    print("JAX devices:", jax.devices())

    # Create the output folder if needed
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)

    src = jnp.array([0, 0])
    dst = jnp.array([6, 2])

    curve, _ = optimize(
        vectorfield_fourvortices,
        src=src,
        dst=dst,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        K=13,
        L=64,
        num_pieces=3,
        popsize=1000,
        sigma0=5,
        tolfun=1e-6,
    )

    xmin, xmax = curve[:, 0].min(), curve[:, 0].max()
    ymin, ymax = curve[:, 1].min(), curve[:, 1].max()

    x: jnp.ndarray = jnp.arange(xmin, xmax, 0.5)
    y: jnp.ndarray = jnp.arange(ymin, ymax, 0.5)
    X, Y = jnp.meshgrid(x, y)
    U, V = vectorfield_fourvortices(X, Y, None)

    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.plot(curve[:, 0], curve[:, 1], color="red")
    plt.plot(src[0], src[1], "o", color="blue")
    plt.plot(dst[0], dst[1], "o", color="green")
    label = "time" if optimize_time else "speed"
    plt.savefig(output_folder / f"main_cmaes_{label}.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
