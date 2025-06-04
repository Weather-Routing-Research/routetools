import itertools
import tomllib
from typing import Any

import numpy as np


def list_config_combinations(path_config: str) -> list[dict[str, Any]]:
    """Generate a list of dictionaries with all possible combinations of parameters.

    Parameters
    ----------
    config : str
        Path to the TOML file

    Returns
    -------
    list[dict]
        List of dictionaries with all possible combinations of parameters
    """
    # Open the configuration file
    with open(path_config, "rb") as f:
        config = tomllib.load(f)

    # Extract the dictionaries from inside it
    dict_vectorfield: dict[str, Any] = config["vectorfield"]
    dict_land: dict[str, Any] = config["land"]
    dict_optimizer: dict[str, dict[str, Any]] = config["optimizer"]
    dict_refiner: dict[str, dict[str, Any]] = config["refiner"]

    # Extract the parameters from the optimizers
    for _, optparams in dict_optimizer.items():
        # Some of the keys contain lists of values
        # We need to create a list of dictionaries
        keys, values = zip(*optparams.items(), strict=False)
        ls_optparams = [
            dict(zip(keys, v, strict=False)) for v in itertools.product(*values)
        ]

    # Extract the parameters from the refiners
    for _, refparams in dict_refiner.items():
        # Some of the keys contain lists of values
        # We need to create a list of dictionaries
        keys, values = zip(*refparams.items(), strict=False)
        # Add "refiner" to the keys
        keys = tuple(["refiner_" + k for k in keys])
        ls_refparams = [
            dict(zip(keys, v, strict=False)) for v in itertools.product(*values)
        ]

    # Create a list of dictionaries with the vectorfield parameters
    ls_vfparams = []
    vfparams: dict[str, Any]
    for vfname, vfparams in dict_vectorfield.items():
        vfparams["vectorfield"] = vfname
        # Convert src and dst to np.array
        vfparams["src"] = np.array(vfparams["src"])
        vfparams["dst"] = np.array(vfparams["dst"])

        # Include to the list
        ls_vfparams.append(vfparams)

    # Finally, do the same for the land
    keys, values = zip(*dict_land.items(), strict=False)
    ls_lndparams = []
    for v in itertools.product(*values):
        new_dict = dict(zip(keys, v, strict=False))
        # If water_level is 1.0, we don't need to specify resolution or seed
        if new_dict["water_level"] >= 1.0:
            if "resolution" in new_dict:
                new_dict.pop("resolution")
            if "random_seed" in new_dict:
                new_dict.pop("random_seed")
        # Include to the list if is not duplicated
        if new_dict not in ls_lndparams:
            ls_lndparams.append(new_dict)

    # Create all possible combinations of vectorfield and optimizer parameters
    # into a list of dictionaries
    ls_params = [
        {**vfparams, **lndparams, **optparams, **refparams}
        for vfparams in ls_vfparams
        for lndparams in ls_lndparams
        for optparams in ls_optparams
        for refparams in ls_refparams
    ]

    # Sort the list of dictionaries by the keys that will
    # influence the size of arrays
    # This is important for the GPU memory allocation
    ls_sort = ["num_pieces", "K", "L", "popsize"]
    ls_params.sort(key=lambda x: [x[k] for k in ls_sort], reverse=True)
    return ls_params
