import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp
import pandas as pd
import typer

from routetools.cmaes import optimize
from routetools.config import list_config_combinations
from routetools.fms import optimize_fms
from routetools.land import Land


def run_param_configuration(
    params: dict, path_jsons: str = "json", idx: int = 0
) -> dict:
    """Run the optimization algorithm with the given parameters.

    Parameters
    ----------
    params : dict
        Dictionary with the parameters
    path_jsons : str, optional
        Path to the folder where the JSON files will be saved, by default "json"
    idx : int, optional
        JSON number, by default 0
    """
    # Make a copy to not replace original
    params = params.copy()
    # Path to the JSON file
    path_json = f"{path_jsons}/{idx:04d}.json"
    # If the file already exists, skip
    if os.path.exists(path_json):
        return

    print(f"Running configuration {idx}...")
    src = params["src"]
    dst = params["dst"]
    xlim = params.pop("xlim")
    ylim = params.pop("ylim")

    land = Land(
        xlim,
        ylim,
        water_level=params.get("water_level"),
        resolution=params.get("resolution"),
        random_seed=params.get("random_seed"),
        outbounds_is_land=params.get("outbounds_is_land"),
    )

    # Is source or destination on land?
    if land(src) or land(dst):
        print("Source or destination is on land. Skipping...")
        results = params
    else:
        # Vectorfield
        vectorfield = params["vectorfield_fun"]

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

        comp_time = time.time() - start

        # FMS variational algorithm (refinement)
        start = time.time()

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

        comp_time_fms = time.time() - start

        # Store the results
        results = {
            **params,
            "cost_cmaes": cost,
            "comp_time_cmaes": comp_time,
            "cost_fms": cost_fms,
            "comp_time_fms": comp_time_fms,
            "curve_cmaes": curve,
            "curve_fms": curve_fms,
        }

    # Pop the vectorfield function
    results.pop("vectorfield_fun", None)
    # Any array contained in the dictionary is turned into a list
    for key, value in results.items():
        if isinstance(value, jnp.ndarray):
            results[key] = value.tolist()
    # Save the results in a JSON file
    with open(path_json, "w") as f:
        json.dump(results, f, indent=4)

    print("\n------------------------\n")


def build_dataframe(path_jsons: str = "json") -> pd.DataFrame:
    """Build a dataframe with the results.

    Parameters
    ----------
    path_jsons : str, optional
        Path to the folder where the JSON files are stored, by default "json"

    Returns
    -------
    pd.DataFrame
        Dataframe with the results
    """
    # Read the results as dictionaries and store them in a list
    ls_files = os.listdir(path_jsons)
    ls_results = []
    for file in ls_files:
        with open(f"{path_jsons}/{file}") as f:
            d: dict = json.load(f)
            # Drop curves
            d.pop("curve_cmaes", None)
            d.pop("curve_fms", None)
            # Add the json file number
            d["json"] = int(file.split(".")[0])
            ls_results.append(d)

    # Build the dataframe
    df = pd.DataFrame(ls_results)

    # We need to fill NaNs in resolution and random_seed with -1
    # so we can group by them
    df["resolution"] = df["resolution"].fillna(-1)
    df["random_seed"] = df["random_seed"].fillna(-1)

    # Extra columns:
    df["comp_time"] = df["comp_time_cmaes"] + df["comp_time_fms"]

    # FMS gains w.r.t. CMA-ES
    df["fms_gain"] = 100 * ((df["cost_cmaes"] - df["cost_fms"]) / df["cost_cmaes"])

    # Group by "water_level", "resolution" and "random_seed"
    # Get the lowest "cost_fms" for each group
    df_best = (
        df.sort_values("cost_fms")
        .groupby(["vectorfield", "water_level", "resolution", "random_seed"])
        .first()
        .reset_index()
    )
    # Add that best cost to the original dataframe
    df_best = df_best.rename(columns={"cost_fms": "cost_best"})
    df = df.merge(
        df_best[
            ["vectorfield", "water_level", "resolution", "random_seed", "cost_best"]
        ],
        on=["vectorfield", "water_level", "resolution", "random_seed"],
        how="left",
    )
    return df


def main(
    max_workers: int = 16,
    path_config: str = "config.toml",
    path_results: str = "output",
):
    """Run the results.

    Parameters
    ----------
    max_workers : int, optional
        Number of workers to use, by default 12
    path_config : str, optional
        Path to the configuration file, by default "config.toml"
    path_results : str, optional
        Path to the output folder, by default "output"
    """
    # Generate the list of parameters
    ls_params = list_config_combinations(path_config)

    # Ensure the output folder exists
    os.makedirs(path_results, exist_ok=True)
    path_jsons = path_results + "/json"
    os.makedirs(path_jsons, exist_ok=True)

    # Use ThreadPoolExecutor to parallelize the execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, params in enumerate(ls_params):
            executor.submit(run_param_configuration, params, path_jsons, idx)

    # Build the dataframe
    df = build_dataframe(path_jsons)
    df.to_csv(path_results + "/results.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    typer.run(main)
