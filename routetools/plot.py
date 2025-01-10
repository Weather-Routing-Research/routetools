from collections.abc import Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt

from routetools.land import Land


def plot_curve(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]
    ],
    ls_curve: list[jnp.ndarray],
    ls_name: list[str] | None = None,
    ls_cost: list[float] | None = None,
    land: Land | None = None,
    xlim: tuple[float, float] = (jnp.inf, -jnp.inf),
    ylim: tuple[float, float] = (jnp.inf, -jnp.inf),
) -> None:
    """Plot the vectorfield and the curves.

    Parameters
    ----------
    vectorfield : Callable
        Vectorfield function
    ls_curve : list[jnp.ndarray]
        List of curves to plot
    ls_name : list[str] | None, optional
        List of names for each curve, by default None
    ls_cost : list[float] | None, optional
        List of costs for each curve, by default None
    land_array : jnp.ndarray | None, optional
        Array of land, by default None
    xlnd : jnp.ndarray | None, optional
        x values of the land array, by default None
    ylnd : jnp.ndarray | None, optional
        y values of the land array, by default None
    xlim : tuple | None, optional
        x limits, by default None
    ylim : tuple | None, optional
        y limits, by default None
    """
    # Set default parameters
    if ls_name is None:
        ls_name = []
    if ls_cost is None:
        ls_cost = []

    # Generate the vectorfield
    xvf = jnp.arange(xlim[0] - 0.5, xlim[1] + 0.5, 0.25)
    yvf = jnp.arange(ylim[0] - 0.5, ylim[1] + 0.5, 0.25)
    t = 0
    X, Y = jnp.meshgrid(xvf, yvf)
    U, V = vectorfield(X, Y, t)

    plt.figure()

    if land is not None:
        # Land is a boolean array, so we need to use contourf
        plt.contourf(
            land.x,
            land.y,
            land.array.T,
            levels=[0, 0.5, 1],
            colors=["white", "black", "black"],
            origin="lower",
        )

    plt.quiver(X, Y, U, V)
    for idx, curve in enumerate(ls_curve):
        label = ""
        if len(ls_name) == len(ls_curve):
            label = ls_name[idx]
        if len(ls_cost) == len(ls_curve):
            cost = ls_cost[idx]
            label += f" {cost:.6f}"
        plt.plot(
            curve[:, 0],
            curve[:, 1],
            marker="o",
            markersize=2,
            label=label,
        )
        # Update limits according to the curve
        xlim = (
            min(xlim[0], min(curve[:, 0])),
            max(xlim[1], max(curve[:, 0])),
        )
        ylim = (
            min(ylim[0], min(curve[:, 1])),
            max(ylim[1], max(curve[:, 1])),
        )
    # Plot the start and end points
    src = curve[0]
    dst = curve[-1]
    plt.plot(src[0], src[1], "o", color="blue")
    plt.plot(dst[0], dst[1], "o", color="green")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    # Make sure the aspect ratio is correct
    plt.gca().set_aspect("equal", adjustable="box")
