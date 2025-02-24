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
    params: dict, path_jsons: str = "json", idx: int = 0, seed_max: int = 0
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
    seed_max : int, optional
        Maximum seed value, by default 0
    """
    # Make a copy to not replace original
    params = params.copy()
    # Path to the JSON file
    path_json = f"{path_jsons}/{idx:06d}.json"
    # If the file already exists, skip
    if os.path.exists(path_json):
        return

    print(f"Running configuration {idx}...")

    land = Land(
        params["xlim"],
        params["ylim"],
        water_level=params.get("water_level"),
        resolution=params.get("resolution"),
        random_seed=params.get("random_seed"),
        outbounds_is_land=params.get("outbounds_is_land"),
    )

    # Is source or destination on land?
    if land(params["src"]) or land(params["dst"]):
        print("Source or destination is on land. We will try another seed.")
        # If this happens, we increase the seed and try again
        params["random_seed"] = int(params.get("random_seed") + seed_max)
        return run_param_configuration(
            params, path_jsons=path_jsons, idx=idx, seed_max=seed_max
        )
    else:
        # Vectorfield
        vectorfield = params["vectorfield_fun"]

        # CMA-ES optimization algorithm
        start = time.time()

        curve, cost = optimize(
            vectorfield,
            params["src"],
            params["dst"],
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
            damping=params.get("damping", 1.0),
            maxfevals=params.get("maxfevals", 50000),
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
            maxfevals=params.get("refiner_maxfevals", 50000),
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

    # Build dataframe and store every 100 iterations
    if idx % 100 == 0:
        path_results = path_jsons.split("/")[0]
        build_dataframe(path_jsons, path_results=path_results)


def build_dataframe(
    path_jsons: str = "json", path_results: str | None = None
) -> pd.DataFrame:
    """Build a dataframe with the results.

    Parameters
    ----------
    path_jsons : str, optional
        Path to the folder where the JSON files are stored, by default "json"
    path_results : str, optional
        Path to the output folder, by default None

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

    # Land generation check: under the same conditions, the land should be the same
    # When the land makes source or destination not reachable, the cost is NaN
    # Thus, when a cost is NaN, it should be NaN for all the rows with the same
    # land configuration
    df["is_nan"] = df["cost_cmaes"].isna()
    col_land = ["vectorfield", "water_level", "resolution", "random_seed"]
    df_group = df.groupby(col_land)["is_nan"].mean()
    mask_wrong = ~((df_group == 0) | (df_group == 1))
    if mask_wrong.any():
        raise ValueError("Land generation is not consistent")
    else:
        # Remove the column
        df = df.drop(columns="is_nan")

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
    if path_results:
        df.to_csv(path_results + "/results.csv", index=False, float_format="%.6f")
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

    # Get the highest seed
    seed_max = max([params.get("random_seed", 0) for params in ls_params] + [1])

    # Ensure the output folder exists
    os.makedirs(path_results, exist_ok=True)
    path_jsons = path_results + "/json"
    os.makedirs(path_jsons, exist_ok=True)

    if max_workers == 1:
        for idx, params in enumerate(ls_params):
            run_param_configuration(params, path_jsons, idx, seed_max)
    else:
        # Use ThreadPoolExecutor to parallelize the execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, params in enumerate(ls_params):
                executor.submit(
                    run_param_configuration, params, path_jsons, idx, seed_max
                )

    # Build the dataframe
    build_dataframe(path_jsons, path_results=path_results)


if __name__ == "__main__":
    typer.run(main)
