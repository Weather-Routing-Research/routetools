import itertools
import tomllib
from typing import Any

import jax.numpy as jnp


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
    dict_optimizer: dict[str, dict[str, Any]] = config["optimizer"]
    dict_land: dict[str, Any] = config["land"]

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
    return ls_params
