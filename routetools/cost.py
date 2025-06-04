from collections.abc import Callable

import numpy as np


def cost_function(
    vectorfield: Callable[
        [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ],
    curve: np.ndarray,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    is_time_variant: bool = True,
) -> np.ndarray:
    """
    Compute the fuel consumption of a batch of paths navigating over a vector field.

    Two modes are supported:
    - Fixed travel speed through water (STW)
    - Fixed travel time.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : np.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW), by default None
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this time,
        by default None
    is_time_variant : bool, optional
        Whether the vectorfield has time dependency, by default True.

    Returns
    -------
    np.ndarray
        A batch of scalars (vector of shape B)
    """
    cost: np.ndarray
    # Choose which cost function to use
    if travel_stw is not None:
        if is_time_variant:
            cost = cost_function_constant_speed_time_variant(
                vectorfield, curve, travel_stw
            )
        else:
            cost = cost_function_constant_speed_time_invariant(
                vectorfield, curve, travel_stw
            )
    elif travel_time is not None:
        if is_time_variant:
            # Not supported
            return np.array([np.nan])
        else:
            cost = cost_function_constant_cost_time_invariant(
                vectorfield, curve, travel_time
            )
    else:
        # Arguments missing
        return np.array([np.nan])
    # Turn any possible infinite costs into 10x the highest value
    cost = np.where(np.isinf(cost), np.nan, cost)
    cost = np.nan_to_num(cost, nan=np.nanmax(cost, initial=1e10) * 10)
    return cost


def cost_function_constant_speed_time_invariant(
    vectorfield: Callable[
        [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ],
    curve: np.ndarray,
    travel_stw: float,
) -> np.ndarray:
    """
    Compute the travel time of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : np.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_stw : float
        The boat will have this fixed speed through water (STW)

    Returns
    -------
    np.ndarray
        A batch of scalars (vector of shape B)
    """
    # Interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
    uinterp, vinterp = vectorfield(curvex, curvey, np.array([0.0]))

    # Distances between points in X and Y
    dx = np.diff(curve[:, :, 0], axis=1)
    dy = np.diff(curve[:, :, 1], axis=1)
    # Power of the distance (segment lengths)
    d2 = np.power(dx, 2) + np.power(dy, 2)
    # Power of the speed through water
    v2 = travel_stw**2
    # Power of the current speed
    w2 = uinterp**2 + vinterp**2
    dw = dx * uinterp + dy * vinterp
    # Cost is the time to travel the segment
    dt = np.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
    # Current > speed -> infeasible path
    dt = np.where(v2 <= w2, float("inf"), dt)
    t_total = np.sum(dt, axis=1)
    return t_total


def cost_function_constant_speed_time_variant(
    vectorfield: Callable[
        [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ],
    curve: np.ndarray,
    travel_stw: float,
) -> np.ndarray:
    """
    Compute the travel time of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : np.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_stw : float
        The boat will have this fixed speed through water (STW)

    Returns
    -------
    np.ndarray
        A batch of scalars (vector of shape B)
    """
    # We will interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2

    # Distances between points in X and Y
    dx = np.diff(curve[:, :, 0], axis=1)
    dy = np.diff(curve[:, :, 1], axis=1)
    # Power of the distance (segment lengths)
    d2 = np.power(dx, 2) + np.power(dy, 2)
    # Power of the speed through water
    v2 = travel_stw**2

    # Initialize inputs for the looping construct
    inputs = (curvex.T, curvey.T, dx.T, dy.T, d2.T)
    t_init = np.zeros(curve.shape[0])

    # Calculate travel times for each segment
    t_final = t_init.copy()
    dt_array = np.zeros_like(dx.T)

    for i in range(len(curvex.T)):
        x, y, dx_step, dy_step, d2_step = (arr[i] for arr in inputs)
        # When sailing from i-1 to i, we interpolate the vector field at the midpoint
        uinterp, vinterp = vectorfield(x, y, t_final)
        # Power of the current speed
        w2 = uinterp**2 + vinterp**2
        dw = dx_step * uinterp + dy_step * vinterp
        # Cost is the time to travel the segment
        dt = np.sqrt(d2_step / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
        # Current > speed -> infeasible path
        dt = np.where(v2 <= w2, 1e10, dt)
        # Update the times
        t_final += dt
        dt_array[i] = dt

    return t_final


def cost_function_constant_cost_time_invariant(
    vectorfield: Callable[
        [np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ],
    curve: np.ndarray,
    travel_time: float,
) -> np.ndarray:
    """
    Compute the fuel consumption of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : np.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_time : float
        The boat can regulate its STW but must complete the path in exactly this time.

    Returns
    -------
    np.ndarray
        A batch of scalars (vector of shape B)
    """
    # Interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
    uinterp, vinterp = vectorfield(curvex, curvey, np.array([0.0]))

    # Distances between points
    dx = np.diff(curve[:, :, 0], axis=1)
    dy = np.diff(curve[:, :, 1], axis=1)
    # Times between points
    dt = travel_time / (curve.shape[1] - 1)
    # We compute the speed over ground from the displacement
    dxdt = dx / dt
    dydt = dy / dt

    # We must navigate the path in a fixed time
    cost = ((dxdt - uinterp) ** 2 + (dydt - vinterp) ** 2) / 2
    total_cost = np.sum(cost, axis=1) * dt
    return total_cost
