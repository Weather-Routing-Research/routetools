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

    land = Land(
        xlim,
        ylim,
        water_level=params.get("water_level", 0.7),
        resolution=params.get("resolution"),
        random_seed=params.get("random_seed"),
        outbounds_is_land=params.get("outbounds_is_land", False),
    )

    # Is source or destination on land?
    if land(src) or land(dst):
        print("Source or destination is on land. Skipping...")
        print("\n------------------------\n")
        return {**params}

    # Vectorfield
    vfname = params["vectorfield"]
    vectorfield = params["vectorfield_fun"]

    # Initialize list
    ls_curves = []
    ls_names = []
    ls_costs = []

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
        num_pieces=params.get("num_pieces", 1),
        popsize=params.get("popsize", 2000),
        sigma0=params.get("sigma0"),
        tolfun=params.get("tolfun", 0.0001),
    )
    if land(curve).any():
        print("The curve is on land")
        cost = jnp.inf
    ls_curves.append(curve)
    ls_names.append("CMA-ES")
    ls_costs.append(cost)

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
            tolfun=params.get("refiner_tolfun", 1e-6),
            damping=params.get("refiner_damping", 0.9),
            maxiter=params.get("refiner_maxiter", 50000),
            verbose=True,
        )
        # FMS returns an extra dimensions, we ignore that
        curve_fms, cost_fms = curve_fms[0], cost_fms[0]
        if land(curve_fms).any():
            print("The curve is on land")
            cost_fms = jnp.inf
        ls_curves.append(curve_fms)
        ls_names.append("FMS")
        ls_costs.append(cost_fms)
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
    if len(ls_curves) > 0:
        plot_curve(
            vectorfield,
            ls_curves,
            ls_name=ls_names,
            ls_cost=ls_costs,
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
    # Remove some columns
    df = df.drop(columns=["vectorfield_fun"])
    df.to_csv(path_results + "/results.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    typer.run(main)
