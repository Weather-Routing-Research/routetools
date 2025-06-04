import os
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import typer

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.plot import plot_curve


def run_single_simulation(
    vectorfield: str = "zero",
    cmaes_K: int = 7,
    cmaes_L: int = 211,
    cmaes_numpieces: int = 1,
    cmaes_popsize: int = 500,
    cmaes_sigma: float = 1,
    cmaes_tolfun: float = 1e-6,
    cmaes_damping: float = 1.0,
    cmaes_maxfevals: int = 200000,
    cmaes_seed: int = 0,
    fms_tolfun: float = 1e-10,
    fms_damping: float = 0.5,
    fms_maxfevals: int = 50000,
    path_img: str = "./output",
    path_config: str = "config.toml",
):
    """
    Run a single simulation to find an optimal path from source to destination.

    Parameters
    ----------
    vectorfield : str, optional
        The name of the vector field function to use, by default "zero".
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
    # Load the config file as a dictionary
    with open(path_config, "rb") as f:
        config = tomllib.load(f)

    # Extract the vectorfield parameters
    vfparams = config["vectorfield"][vectorfield]
    src = np.array(vfparams["src"])
    dst = np.array(vfparams["dst"])
    travel_stw = vfparams.get("travel_stw", None)
    travel_time = vfparams.get("travel_time", None)
    land_xlim = vfparams.get("xlim", None)
    land_ylim = vfparams.get("ylim", None)

    # Load the vectorfield function
    vectorfield_module = __import__(
        "routetools.vectorfield", fromlist=["vectorfield_" + vectorfield]
    )
    vectorfield_fun = getattr(vectorfield_module, "vectorfield_" + vectorfield)

    # CMA-ES optimization algorithm
    curve_cmaes, cost_cmaes = optimize(
        vectorfield_fun,
        src,
        dst,
        travel_stw=travel_stw,
        travel_time=travel_time,
        K=cmaes_K,
        L=cmaes_L,
        num_pieces=cmaes_numpieces,
        popsize=cmaes_popsize,
        sigma0=cmaes_sigma,
        tolfun=cmaes_tolfun,
        damping=cmaes_damping,
        maxfevals=cmaes_maxfevals,
        seed=cmaes_seed,
    )

    # Create a straight line from source to destination,
    # with as many points as the CMA-ES curve
    num_points = curve_cmaes.shape[0]
    curve_straight = np.linspace(src, dst, num_points)

    # FMS variational algorithm (refinement)
    curve_fms, cost_fms = optimize_fms(
        vectorfield_fun,
        curve=curve_straight,
        travel_stw=travel_stw,
        travel_time=travel_time,
        tolfun=fms_tolfun,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
        verbose=True,
    )
    # FMS returns an extra dimensions, we ignore that
    curve_fms, cost_fms = curve_fms[0], cost_fms[0]

    # Plot them
    fig, ax = plot_curve(
        vectorfield_fun,
        [curve_cmaes, curve_fms],
        ls_name=["CMA-ES", "FMS only"],
        ls_cost=[cost_cmaes, cost_fms],
        xlim=land_xlim,
        ylim=land_ylim,
    )
    ax.set_title(f"{vectorfield}")
    fig.savefig(f"{path_img}/literature_{vectorfield}.png")
    plt.close(fig)


def main(path_output: str = "./output", path_config: str = "config.toml"):
    """Run the simulation."""
    # Make the output directory if it does not exist
    os.makedirs(path_output, exist_ok=True)

    for vectorfield in sorted(
        ["circular", "fourvortices", "doublegyre", "techy", "swirlys"]
    ):
        run_single_simulation(
            vectorfield=vectorfield, path_img=path_output, path_config=path_config
        )


if __name__ == "__main__":
    typer.run(main)
