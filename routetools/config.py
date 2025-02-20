import itertools
import tomllib
from typing import Any

import jax.numpy as jnp
from jax import jit

# These keys are not used in the vectorfield function
LS_VF_IGNORE = [
    "src",
    "dst",
    "xlim",
    "ylim",
    "travel_stw",
    "travel_time",
    "vectorfield",
]


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
        # Convert src and dst to jnp.array
        vfparams["src"] = jnp.array(vfparams["src"])
        vfparams["dst"] = jnp.array(vfparams["dst"])
        # Load the vectorfield function
        vectorfield_module = __import__(
            "routetools.vectorfield", fromlist=["vectorfield_" + vfname]
        )
        vffun = getattr(vectorfield_module, "vectorfield_" + vfname)

        # We are going to build a vectorfield function using the
        # extra arguments that are not in LS_VF_IGNORE
        vfparams_extra = {k: v for k, v in vfparams.items() if k not in LS_VF_IGNORE}

        def vectorfield(vffun=vffun, vfparams_extra=vfparams_extra):  # type: ignore[no-untyped-def]
            @jit
            def inner(x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
                return vffun(x, y, t, **vfparams_extra)  # type: ignore[no-any-return]

            inner.is_time_variant = vffun.is_time_variant  # type: ignore[attr-defined]
            return inner

        vfparams["vectorfield_fun"] = vectorfield()  # type: ignore[no-untyped-call]

        ls_vfparams.append(vfparams)

    # Finally, do the same for the land
    keys, values = zip(*dict_land.items(), strict=False)
    ls_lndparams = []
    for v in itertools.product(*values):
        new_dict = dict(zip(keys, v, strict=False))
        # If water_level is 1.0, we don't need to specify resolution or seed
        if new_dict["water_level"] >= 1.0:
            new_dict.pop("resolution")
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
    return ls_params
