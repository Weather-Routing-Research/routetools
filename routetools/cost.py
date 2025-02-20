from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import jit


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

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    sog : jnp.ndarray, optional
        The speed over ground (SOG) of the boat, by default None
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW), by default None
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this time,
        by default None

    Returns
    -------
    jnp.ndarray
        A batch of scalars (vector of shape B)
    """
    cost: jnp.ndarray
    # If the curve is a single point with shape (2)
    # We create the batch dimension B = 1 by expanding it to (1 x 2)
    if curve.ndim == 1:
        curve = curve[None, :]
    # If each curve is a single point with shape (B x 2)
    # We convert it to (B x 2 x 2) by duplicating the point
    if curve.ndim == 2:
        curve = jnp.repeat(curve[:, None, :], 2, axis=1)
    # Choose which cost function to use
    if travel_stw is not None:
        if vectorfield.is_time_variant:  # type: ignore[attr-defined]
            cost = cost_function_constant_speed_time_variant(
                vectorfield, curve, travel_stw
            )
        else:
            cost = cost_function_constant_speed_time_invariant(
                vectorfield, curve, travel_stw
            )
    elif travel_time is not None:
        if vectorfield.is_time_variant:  # type: ignore[attr-defined]
            raise ValueError("Time variant vector fields are not supported.")
        else:
            cost = cost_function_constant_cost_time_invariant(
                vectorfield, curve, travel_time
            )
    else:
        raise ValueError("Either travel_stw or travel_time must be set.")
    # Turn any possible infinite costs into 10x the highest value
    cost = jnp.where(jnp.isinf(cost), jnp.nan, cost)
    cost = jnp.nan_to_num(cost, nan=jnp.nanmax(cost, initial=1e10) * 10)
    return cost


@partial(jit, static_argnums=(0, 2))
def cost_function_constant_speed_time_invariant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_stw: float,
) -> jnp.ndarray:
    """
    Compute the travel time of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_stw : float
        The boat will have this fixed speed through water (STW)

    Returns
    -------
    jnp.ndarray
        A batch of scalars (vector of shape B)
    """
    # Interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
    uinterp, vinterp = vectorfield(curvex, curvey, jnp.array([]))

    # Distances between points in X and Y
    dx = jnp.diff(curve[:, :, 0], axis=1)
    dy = jnp.diff(curve[:, :, 1], axis=1)
    # Power of the distance (segment lengths)
    d2 = jnp.power(dx, 2) + jnp.power(dy, 2)
    # Power of the speed through water
    v2 = travel_stw**2
    # Power of the current speed
    w2 = uinterp**2 + vinterp**2
    dw = dx * uinterp + dy * vinterp
    # Cost is the time to travel the segment
    dt = jnp.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
    # Current > speed -> infeasible path
    dt = jnp.where(v2 <= w2, float("inf"), dt)
    t_total = jnp.sum(dt, axis=1)
    return t_total


@partial(jit, static_argnums=(0, 2))
def cost_function_constant_speed_time_variant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_stw: float,
) -> jnp.ndarray:
    """
    Compute the travel time of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_stw : float
        The boat will have this fixed speed through water (STW)

    Returns
    -------
    jnp.ndarray
        A batch of scalars (vector of shape B)
    """
    # We will interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2

    # Distances between points in X and Y
    dx = jnp.diff(curve[:, :, 0], axis=1)
    dy = jnp.diff(curve[:, :, 1], axis=1)
    # Power of the distance (segment lengths)
    d2 = jnp.power(dx, 2) + jnp.power(dy, 2)
    # Power of the speed through water
    v2 = travel_stw**2

    # Initialize times
    t = jnp.zeros(curve.shape[:-1])
    # Move along the curve one point at a time
    for i in range(1, curve.shape[0]):
        # When sailing from i-1 to i, we interpolate the vector field at the midpoint
        uinterp, vinterp = vectorfield(curvex[:, i - 1], curvey[:, i - 1], t[:, i - 1])
        # Power of the current speed
        w2 = uinterp**2 + vinterp**2
        dw = dx[:, i] * uinterp + dy[:, i] * vinterp
        # Cost is the time to travel the segment
        dt = jnp.sqrt(d2[:, i] / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
        # Current > speed -> infeasible path
        dt = jnp.where(v2 <= w2, 1e6, dt)
        # Compute the time at which we reach i
        t = t.at[:, i].set(t[:, i - 1] + dt)
    return t[:, -1]


@partial(jit, static_argnums=(0, 2))
def cost_function_constant_cost_time_invariant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_time: float,
) -> jnp.ndarray:
    """
    Compute the fuel consumption of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_time : float
        The boat can regulate its STW but must complete the path in exactly this time.

    Returns
    -------
    jnp.ndarray
        A batch of scalars (vector of shape B)
    """
    # Interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
    uinterp, vinterp = vectorfield(curvex, curvey, jnp.array([]))

    # Distances between points
    dx = jnp.diff(curve[:, :, 0], axis=1)
    dy = jnp.diff(curve[:, :, 1], axis=1)
    # Times between points
    dt = travel_time / (curve.shape[1] - 1)
    # We compute the speed over ground from the displacement
    dxdt = dx / dt
    dydt = dy / dt

    # We must navigate the path in a fixed time
    cost = ((dxdt - uinterp) ** 2 + (dydt - vinterp) ** 2) / 2
    total_cost = jnp.sum(cost, axis=1) * dt
    return total_cost
