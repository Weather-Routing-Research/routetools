from collections.abc import Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
) -> tuple[Figure, Axes]:
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

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes objects
    """
    # Set default parameters
    if ls_name is None:
        ls_name = []
    if ls_cost is None:
        ls_cost = []

    fig = plt.figure()
    ax = plt.gca()

    # Plot the land
    if land is not None:
        # Land is a boolean array, so we need to use contourf
        ax.contourf(
            land.x,
            land.y,
            land.array.T,
            levels=[0, 0.5, 1],
            colors=["white", "black", "black"],
            origin="lower",
            zorder=0,
        )

    # Plot the curves
    for idx, curve in enumerate(ls_curve):
        label = ""
        if len(ls_name) == len(ls_curve):
            label = ls_name[idx]
        if len(ls_cost) == len(ls_curve):
            cost = ls_cost[idx]
            label += f" {cost:.3f}"
        ax.plot(
            curve[:, 0], curve[:, 1], marker="o", markersize=2, label=label, zorder=2
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
    ax.plot(src[0], src[1], "o", color="blue", zorder=3)
    ax.plot(dst[0], dst[1], "o", color="green", zorder=3)

    # Plot the vectorfield
    xvf = jnp.arange(xlim[0] - 0.5, xlim[1] + 0.5, 0.25)
    yvf = jnp.arange(ylim[0] - 0.5, ylim[1] + 0.5, 0.25)
    t = 0
    X, Y = jnp.meshgrid(xvf, yvf)
    U, V = vectorfield(X, Y, t)
    ax.quiver(X, Y, U, V, zorder=1)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Make sure the aspect ratio is correct
    ax.set_aspect("equal", adjustable="box")

    # Adjust the layout
    fig.tight_layout(pad=2.5)

    return fig, ax


def plot_table_mean_std(
    df: pd.DataFrame,
    value_column: str,
    index_columns: list,
    column_columns: list,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "coolwarm",
    colorbar_label: str = "",
    title: str = "",
):
    """
    Plot a heatmap for a given metric with mean ± standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be visualized.
    mask : np.ndarray
        A boolean mask to filter the DataFrame.
    value_column : str
        The name of the column containing the values to be aggregated.
    index_columns : list
        List of column names to use as row indices (e.g., ["sigma0", "popsize"]).
    column_columns : list
        List of column names to use as column indices (e.g., ["K", "L"]).
    vmin : float, optional
        Minimum value for the heatmap color scale (default is None).
    vmax : float, optional
        Maximum value for the heatmap color scale (default is None).
    cmap : str, optional
        Colormap for the heatmap (default is "coolwarm").
    colorbar_label : str, optional
        Label for the colorbar (default is an empty string).
    title : str, optional
        Title of the heatmap (default is an empty string).

    Returns
    -------
    None
        Displays the heatmap plot.
    """
    # Create pivot tables for mean and standard deviation
    pivot_table_mean = df.pivot_table(
        values=value_column,
        index=index_columns,
        columns=column_columns,
        aggfunc=lambda x: np.nanmean(x),
    ).round(2)

    pivot_table_std = df.pivot_table(
        values=value_column,
        index=index_columns,
        columns=column_columns,
        aggfunc=lambda x: np.nanstd(x),
    ).round(2)

    # Combine mean and std into a single pivot table for annotation
    pivot_table_combined = pivot_table_mean.copy()
    for col in pivot_table_combined.columns:
        pivot_table_combined[col] = (
            pivot_table_mean[col].astype(str)
            + "\n ± "
            + pivot_table_std[col].astype(str)
        )

    # Remove first column level if multi-indexed
    if isinstance(pivot_table_combined.columns, pd.MultiIndex):
        pivot_table_combined.columns = pivot_table_combined.columns.droplevel()
        pivot_table_mean.columns = pivot_table_mean.columns.droplevel()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        pivot_table_mean,
        annot=pivot_table_combined,
        vmin=vmin,
        vmax=vmax,
        fmt="",
        cmap=cmap,
        cbar_kws={"label": colorbar_label},
        annot_kws={"ha": "center", "va": "center"},
        ax=ax,
    )

    # Set labels and title
    ax.set_xlabel(" - ".join(column_columns))
    ax.set_ylabel(" - ".join(index_columns))
    ax.set_title(title)
    return fig, ax
