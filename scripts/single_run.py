import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.land import Land
from routetools.plot import plot_curve


def run_single_simulation(
    vectorfield: str = "fourvortices",
    src: tuple[float, float] = [0, 0],
    dst: tuple[float, float] = [6, 2],
    travel_stw: float = 1,
    travel_time: float = None,
    land_xlim: tuple[float, float] = None,
    land_ylim: tuple[float, float] = None,
    land_waterlevel: float = 0.6,
    land_resolution: int = 5,
    land_seed: int = 2,
    land_penalty: float = 10,
    outbounds_is_land: bool = False,
    cmaes_K: int = 6,
    cmaes_L: int = 128,
    cmaes_numpieces: int = 1,
    cmaes_popsize: int = 2000,
    cmaes_sigma: float = 1,
    cmaes_tolfun: float = 0.1,
    cmaes_maxfevals: int = 20000,
    fms_tolfun: float = 1e-6,
    fms_damping: float = 0.9,
    fms_maxfevals: int = 5000,
    path_img: str = "./output",
):
    """
    Run a single simulation to find an optimal path from source to destination.

    Parameters
    ----------
    vectorfield : str, optional
        The name of the vector field function to use, by default "zero".
    src : tuple[float, float], optional
        The source coordinates, by default (0, 0).
    dst : tuple[float, float], optional
        The destination coordinates, by default (5, 5).
    travel_stw : float, optional
        The speed through water, by default 1.
    travel_time : float, optional
        The travel time for JIT. Overwrites travel_stw, by default None.
    land_xlim : tuple[float, float], optional
        The x-axis limits for the land, by default None.
    land_ylim : tuple[float, float], optional
        The y-axis limits for the land, by default None.
    land_waterlevel : float, optional
        The water level for the land, by default 0.7.
    land_resolution : int, optional
        The resolution of the land grid, by default 3.
    land_seed : int, optional
        The random seed for land generation, by default 0.
    land_penalty : float, optional
        The penalty for traveling over land, by default 10.
    outbounds_is_land : bool, optional
        Whether out-of-bounds areas are considered land, by default False.
    cmaes_K : int, optional
        The number of control points for CMA-ES, by default 6.
    cmaes_L : int, optional
        The number of segments for CMA-ES, by default 64.
    cmaes_numpieces : int, optional
        The number of pieces for CMA-ES, by default 1.
    cmaes_popsize : int, optional
        The population size for CMA-ES, by default 2000.
    cmaes_sigma : float, optional
        The initial standard deviation for CMA-ES, by default 1.
    cmaes_tolfun : float, optional
        The tolerance for the function value in CMA-ES, by default 0.0001.
    fms_tolfun : float, optional
        The tolerance for the function value in FMS, by default 1e-6.
    fms_damping : float, optional
        The damping factor for FMS, by default 0.9.
    fms_maxfevals : int, optional
        The maximum number of iterations for FMS, by default 50000.
    path_img : str, optional
        The path to save output images, by default "./output".
    """
    src = jnp.array(src)
    dst = jnp.array(dst)

    if land_xlim is None:
        land_xlim = (
            jnp.min(jnp.array([src[0], dst[0]]) - 1),
            jnp.max(jnp.array([src[0], dst[0]]) + 1),
        )
    if land_ylim is None:
        land_ylim = (
            jnp.min(jnp.array([src[1], dst[1]]) - 1),
            jnp.max(jnp.array([src[1], dst[1]]) + 1),
        )

    # Load the vectorfield function
    vectorfield_module = __import__(
        "routetools.vectorfield", fromlist=["vectorfield_" + vectorfield]
    )
    vectorfield_fun = getattr(vectorfield_module, "vectorfield_" + vectorfield)

    land = Land(
        land_xlim,
        land_ylim,
        water_level=land_waterlevel,
        resolution=land_resolution,
        random_seed=land_seed,
        outbounds_is_land=outbounds_is_land,
    )

    # Is source or destination on land?
    if land(src) or land(dst):
        print("Source or destination is on land.")
        return

    # CMA-ES optimization algorithm
    start = time.time()

    curve_cmaes, cost_cmaes = optimize(
        vectorfield_fun,
        src,
        dst,
        land=land,
        penalty=land_penalty,
        travel_stw=travel_stw,
        travel_time=travel_time,
        K=cmaes_K,
        L=cmaes_L,
        num_pieces=cmaes_numpieces,
        popsize=cmaes_popsize,
        sigma0=cmaes_sigma,
        tolfun=cmaes_tolfun,
        maxfevals=cmaes_maxfevals,
    )

    if land(curve_cmaes).any():
        print("The curve is on land")
        cost_cmaes = jnp.inf

    print(f"Computation time of CMA-ES: {time.time() - start}")

    # FMS variational algorithm (refinement)
    start = time.time()

    curve_fms, cost_fms = optimize_fms(
        vectorfield_fun,
        curve=curve_cmaes,
        land=land,
        travel_stw=travel_stw,
        travel_time=travel_time,
        tolfun=fms_tolfun,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
        verbose=True,
    )
    # FMS returns an extra dimensions, we ignore that
    curve_fms, cost_fms = curve_fms[0], cost_fms[0]

    if land(curve_fms).any():
        print("The curve is on land")
        cost_fms = jnp.inf

    print(f"Computation time of FMS: {time.time() - start}")

    # Plot them
    fig, ax = plot_curve(
        vectorfield_fun,
        [curve_cmaes, curve_fms],
        ls_name=["CMA-ES", "FMS"],
        ls_cost=[cost_cmaes, cost_fms],
        land=land,
        xlim=land_xlim,
        ylim=land_ylim,
    )
    ax.set_title(f"{vectorfield}")
    fig.savefig(f"{path_img}/single_run.png")
    plt.close(fig)


if __name__ == "__main__":
    typer.run(run_single_simulation)
