import itertools
import time
import tomllib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import typer

from routetools.cmaes import optimize
from routetools.land import generate_land_array, generate_land_function


def main(path_config: str = "config.toml", path_results: str = "output"):
    """Run the results.

    Parameters
    ----------
    path_config : str, optional
        Path to the configuration file, by default "config.toml"
    """
    with open(path_config, "rb") as f:
        config = tomllib.load(f)

    dict_vectorfield: dict = config["vectorfield"]
    dict_optimizer: dict = config["optimizer"]
    dict_land: dict = config["land"]

    xlim = dict_land.pop("xlim")
    ylim = dict_land.pop("ylim")
    dict_land["x"] = jnp.linspace(*xlim, 100)
    dict_land["y"] = jnp.linspace(*ylim, 100)

    for _, optparams in dict_optimizer.items():
        # Some of the keys contain lists of values
        # We need to create a list of dictionaries
        keys, values = zip(*optparams.items(), strict=False)
        ls_optparams = [
            dict(zip(keys, v, strict=False)) for v in itertools.product(*values)
        ]

    ls_vfparams = []

    for vfname, vfparams in dict_vectorfield.items():
        vectorfield_module = __import__(
            "routetools.vectorfield", fromlist=["vectorfield_" + vfname]
        )
        vfparams["vectorfield"] = vfname
        # Convert src and dst to jnp.array
        vfparams["src"] = jnp.array(vfparams["src"])
        vfparams["dst"] = jnp.array(vfparams["dst"])
        ls_vfparams.append(vfparams)

    # Create all possible combinations of vectorfield and optimizer parameters
    # into a list of dictionaries
    ls_params = [
        {**vfparams, **optparams}
        for vfparams in ls_vfparams
        for optparams in ls_optparams
    ]

    # Initialize list of results
    results: list[dict] = []
    fignum = 0

    for params in ls_params:
        # We need to pop the vectorfield name to avoid passing it to the optimizer
        vfname = params.pop("vectorfield")
        land_function = generate_land_function(**dict_land)
        vectorfield = getattr(vectorfield_module, "vectorfield_" + vfname)
        start = time.time()
        try:
            curve, cost = optimize(
                vectorfield,
                land_function=land_function,
                **params,
            )
        except Exception as e:
            print(e)
            curve = None
            cost = jnp.inf

        comp_time = time.time() - start

        # Store the results
        results.append(
            {
                "vectorfield": vfname,
                **params,
                "cost": cost,
                "comp_time": comp_time,
            }
        )

        # Plot them
        if curve is not None:
            src = params["src"]
            dst = params["dst"]
            xmin = min([src[0], dst[0]]) - 1
            xmax = max([src[0], dst[0]]) + 1
            ymin = min([src[1], dst[1]]) - 1
            ymax = max([src[1], dst[1]]) + 1
            x = jnp.arange(xmin, xmax, 0.25)
            y = jnp.arange(ymin, ymax, 0.25)
            t = 0
            X, Y = jnp.meshgrid(x, y)
            U, V = vectorfield(X, Y, t)

            plt.figure()
            # Land is a boolean array, so we need to use contourf
            land_array = generate_land_array(**dict_land)
            plt.contourf(
                dict_land["x"],
                dict_land["y"],
                land_array.T,
                cmap="Greys",
                origin="lower",
            )
            plt.quiver(X, Y, U, V)
            plt.plot(curve[:, 0], curve[:, 1], color="red", marker="o")
            plt.plot(src[0], src[1], "o", color="blue")
            plt.plot(dst[0], dst[1], "o", color="green")
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.title(f"{vfname} | Cost: {cost:.6f}")
            plt.savefig(f"{path_results}/fig{fignum:03d}.png")
            fignum += 1
            plt.close()

    # Save the results to a csv file using pandas
    df = pd.DataFrame(results)
    df.to_csv(path_results + "/results.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    typer.run(main)
