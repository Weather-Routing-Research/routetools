import itertools
import os
import time
import tomllib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import typer

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.land import generate_land_array, generate_land_function
from routetools.plot import plot_curve


def main(path_config: str = "config.toml", path_results: str = "output"):
    """Run the results.

    Parameters
    ----------
    path_config : str, optional
        Path to the configuration file, by default "config.toml"
    path_results : str, optional
        Path to the output folder, by default "output"
    """
    # Open the configuration file
    with open(path_config, "rb") as f:
        config = tomllib.load(f)
    # Extract the dictionaries from inside it
    dict_vectorfield: dict = config["vectorfield"]
    dict_optimizer: dict = config["optimizer"]
    dict_land: dict = config["land"]

    # Ensure the output folder exists
    os.makedirs(path_results, exist_ok=True)
    path_imgs = path_results + "/img"
    os.makedirs(path_imgs, exist_ok=True)

    # Extract the parameters from the optimizers
    for _, optparams in dict_optimizer.items():
        # Some of the keys contain lists of values
        # We need to create a list of dictionaries
        keys, values = zip(*optparams.items(), strict=False)
        ls_optparams = [
            dict(zip(keys, v, strict=False)) for v in itertools.product(*values)
        ]

    # Create a list of dictionaries with the vectorfield parameters
    ls_vfparams = []
    for vfname, vfparams in dict_vectorfield.items():
        vfparams["vectorfield"] = vfname
        # Convert src and dst to jnp.array
        vfparams["src"] = jnp.array(vfparams["src"])
        vfparams["dst"] = jnp.array(vfparams["dst"])
        ls_vfparams.append(vfparams)

    # Finally, do the same for the land
    keys, values = zip(*dict_land.items(), strict=False)
    ls_lndparams = [
        dict(zip(keys, v, strict=False)) for v in itertools.product(*values)
    ]

    # Create all possible combinations of vectorfield and optimizer parameters
    # into a list of dictionaries
    ls_params = [
        {**vfparams, **optparams, **lndparams}
        for vfparams in ls_vfparams
        for optparams in ls_optparams
        for lndparams in ls_lndparams
    ]

    # Initialize list of results
    results: list[dict] = []
    fignum = 0

    for params in ls_params:
        src = params["src"]
        dst = params["dst"]
        xlim = params.pop("xlim")
        ylim = params.pop("ylim")
        xlnd = jnp.arange(*xlim, 1 / params["resolution"])
        ylnd = jnp.arange(*ylim, 1 / params["resolution"])
        land_function = generate_land_function(
            xlnd,
            ylnd,
            water_level=params["water_level"],
            resolution=params["resolution"],
            random_seed=params["random_seed"],
        )
        land_array = generate_land_array(
            xlnd,
            ylnd,
            water_level=params["water_level"],
            resolution=params["resolution"],
            random_seed=params["random_seed"],
        )
        vfname = params["vectorfield"]
        vectorfield_module = __import__(
            "routetools.vectorfield", fromlist=["vectorfield_" + vfname]
        )
        vectorfield = getattr(vectorfield_module, "vectorfield_" + vfname)

        # CMA-ES
        start = time.time()
        try:
            curve, cost = optimize(
                vectorfield,
                src,
                dst,
                land_function=land_function,
                travel_stw=params.get("travel_stw", None),
                travel_time=params.get("travel_time", None),
                K=params["K"],
                L=params["L"],
                popsize=params["popsize"],
                sigma0=params.get("sigma0", None),
                tolfun=params["tolfun"],
                penalty=params.get("penalty", None),
            )
            if cost >= params.get("penalty", jnp.inf):
                raise ValueError("The curve is on land")

        except Exception as e:
            print(e)
            curve = None
            cost = jnp.inf
        comp_time = time.time() - start

        # FMS
        start = time.time()
        try:
            curve_fms, cost_fms = optimize_fms(
                vectorfield,
                curve=curve,
                land_function=land_function,
                travel_stw=params.get("travel_stw", None),
                travel_time=params.get("travel_time", None),
            )
            # FMS returns an extra dimensions, we ignore that
            curve_fms, cost_fms = curve_fms[0], cost_fms[0]
        except Exception as e:
            print(e)
            curve_fms = None
            cost_fms = jnp.inf
        comp_time_fms = time.time() - start

        # Store the results
        results.append(
            {
                **params,
                "cost": cost,
                "cost_fms": cost_fms,
                "comp_time": comp_time,
                "comp_time_fms": comp_time_fms,
                "image": fignum if curve is not None else None,
            }
        )

        # Plot them
        if curve is not None:
            plot_curve(
                vectorfield,
                [curve, curve_fms],
                ls_name=["CMA-ES", "FMS"],
                ls_cost=[cost, cost_fms],
                land_array=land_array,
                xlnd=xlnd,
                ylnd=ylnd,
                xlim=xlim,
                ylim=ylim,
            )
            plt.title(f"{vfname}")
            plt.savefig(f"{path_imgs}/fig{fignum:04d}.png")
            fignum += 1
            plt.close()

    # Save the results to a csv file using pandas
    df = pd.DataFrame(results)
    df.to_csv(path_results + "/results.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    typer.run(main)
