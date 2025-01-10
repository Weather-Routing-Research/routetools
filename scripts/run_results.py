import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import typer

from routetools.cmaes import optimize
from routetools.config import list_config_combinations
from routetools.fms import optimize_fms
from routetools.land import Land
from routetools.plot import plot_curve


def run_param_configuration(
    params: dict, path_imgs: str = "img", fignum: int = 0
) -> dict:
    """Run the optimization algorithm with the given parameters.

    Parameters
    ----------
    params : dict
        Dictionary with the parameters
    path_imgs : str, optional
        Path to the folder where the images will be saved, by default "img"
    fignum : int, optional
        Figure number, by default 0

    Returns
    -------
    dict
        Dictionary with the results
    """
    src = params["src"]
    dst = params["dst"]
    xlim = params.pop("xlim")
    ylim = params.pop("ylim")

    # Land
    xlnd = jnp.arange(*xlim, 1 / params["resolution"])
    ylnd = jnp.arange(*ylim, 1 / params["resolution"])
    land = Land(
        xlnd,
        ylnd,
        water_level=params.get("water_level", 0.7),
        resolution=params.get("resolution"),
        random_seed=params.get("random_seed"),
    )

    # Vectorfield
    vfname = params["vectorfield"]
    vectorfield_module = __import__(
        "routetools.vectorfield", fromlist=["vectorfield_" + vfname]
    )
    vectorfield = getattr(vectorfield_module, "vectorfield_" + vfname)

    # CMA-ES optimization algorithm
    start = time.time()

    curve, cost = optimize(
        vectorfield,
        src,
        dst,
        land=land,
        penalty=params.get("penalty", 10),
        travel_stw=params.get("travel_stw"),
        travel_time=params.get("travel_time"),
        K=params.get("K", 6),
        L=params.get("L", 64),
        popsize=params.get("popsize", 2000),
        sigma0=params.get("sigma0"),
        tolfun=params.get("tolfun", 0.0001),
    )
    if land(curve).any():
        print("The curve is on land")
        curve = None
        cost = jnp.inf

    comp_time = time.time() - start

    # FMS variational algorithm (refinement)
    start = time.time()
    try:
        curve_fms, cost_fms = optimize_fms(
            vectorfield,
            curve=curve,
            land=land,
            travel_stw=params.get("travel_stw"),
            travel_time=params.get("travel_time"),
        )
        # FMS returns an extra dimensions, we ignore that
        curve_fms, cost_fms = curve_fms[0], cost_fms[0]
    except Exception as e:
        print(e)
        curve_fms = None
        cost_fms = jnp.inf
    comp_time_fms = time.time() - start

    # Store the results
    results = {
        **params,
        "cost": cost,
        "cost_fms": cost_fms,
        "comp_time": comp_time,
        "comp_time_fms": comp_time_fms,
        "image": fignum if curve is not None else None,
    }

    # Plot them
    if curve is not None:
        plot_curve(
            vectorfield,
            [curve, curve_fms],
            ls_name=["CMA-ES", "FMS"],
            ls_cost=[cost, cost_fms],
            land=land,
            xlim=xlim,
            ylim=ylim,
        )
        plt.title(f"{vfname}")
        plt.savefig(f"{path_imgs}/fig{fignum:04d}.png")
        plt.close()

    print("\n------------------------\n")

    return results


def main(path_config: str = "config.toml", path_results: str = "output"):
    """Run the results.

    Parameters
    ----------
    path_config : str, optional
        Path to the configuration file, by default "config.toml"
    path_results : str, optional
        Path to the output folder, by default "output"
    """
    # Generate the list of parameters
    ls_params = list_config_combinations(path_config)

    # Ensure the output folder exists
    os.makedirs(path_results, exist_ok=True)
    path_imgs = path_results + "/img"
    os.makedirs(path_imgs, exist_ok=True)

    # Initialize list of results
    results: list[dict] = []

    for idx, params in enumerate(ls_params):
        results.append(run_param_configuration(params, path_imgs=path_imgs, fignum=idx))

    # Save the results to a csv file using pandas
    df = pd.DataFrame(results)
    df.to_csv(path_results + "/results.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    typer.run(main)
