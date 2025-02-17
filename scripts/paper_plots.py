import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from routetools.land import Land


def land_configurations(seed: int = 1, fout: str = "output/land_configurations.png"):
    """Generate a grid of land configurations.

    Parameters
    ----------
    seed : int, optional
        Random seed for generating the land, by default 1
    fout : str, optional
        Output file path, by default "output/land_configurations.png"
    """
    # Define the parameters for the grid
    resolutions = [3, 4, 5]
    water_levels = [0.9, 0.8, 0.7]

    # Create a 3x3 grid of plots
    fig, axes = plt.subplots(3, 3, figsize=(5, 5))

    # Generate and plot the land for each combination of resolution and water level
    for i, resolution in enumerate(resolutions):
        for j, water_level in enumerate(water_levels):
            ax: Axes = axes[i, j]
            # Create the land object
            land = Land(
                xlim=(0, 6),
                ylim=(0, 6),
                water_level=water_level,
                resolution=resolution,
                random_seed=seed,
            )

            # Land is a boolean array, so we need to use contourf
            ax.contourf(
                land.x,
                land.y,
                land.array.T,
                levels=[0, 0.5, 1],
                colors=["white", "black", "black"],
                origin="lower",
            )

            # Remove the axis ticks
            ax.set_xticks([])
            ax.set_yticks([])

    # Place the water level and resolution labels
    for i, resolution in enumerate(resolutions):
        axes[i, 0].set_ylabel(f"Resolution: {resolution}")
    for j, water_level in enumerate(water_levels):
        axes[-1, j].set_xlabel(f"Water level: {water_level}")

    # Adjust layout and show the plot
    fig.tight_layout()
    fig.savefig(fout)


def land_avoidance(folder: str = "output"):
    """
    Generate and save plots for land avoidance analysis based on simulation results.

    This function reads simulation results from a CSV file, identifies the worst
    examples based on the cost function, and generates plots to visualize the land
    avoidance behavior for different water levels, resolutions, and random seeds.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file and JSON files,
        by default "output".
    """
    # Read the results CSV file
    df = pd.read_csv(f"{folder}/results.csv")
    # Filter the rows with zero vectorfield
    mask = df["vectorfield"] == "zero"
    # Take the minimum time for each configuration
    df_land = (
        df[mask]
        .groupby(["water_level", "resolution", "random_seed"])["cost_fms"]
        .min()
        .dropna()
        .sort_values(ascending=False)
    )

    # Generate plots for the worst three examples
    for idx in [0, 1, 2]:
        # Extract the configuration parameters
        water_level, resolution, random_seed = df_land.index[idx]
        cost = df_land.iloc[idx]

        # Identify the corresponding row in the dataframe
        mask = (
            (df["water_level"] == water_level)
            & (df["resolution"] == resolution)
            & (df["random_seed"] == random_seed)
            & (df["cost_fms"] == cost)
        )
        row = df[mask].iloc[0]

        # Load the JSON file for the identified example
        json_id = int(row["json"])
        with open(f"{folder}/json/{json_id}.json") as f:
            d: dict = json.load(f)

        # Set up the plot
        fig, ax = plt.subplots(figsize=(4, 4))

        # Plot the land
        resolution = max(int(resolution), 1)
        random_seed = max(int(random_seed), 0)
        xlim = [-1, 6]
        ylim = [-1, 6]
        land = Land(
            xlim=xlim,
            ylim=ylim,
            water_level=water_level,
            resolution=resolution,
            random_seed=random_seed,
        )
        ax.contourf(
            land.x,
            land.y,
            land.array.T,
            levels=[0, 0.5, 1],
            colors=["white", "black", "black"],
            origin="lower",
            zorder=0,
        )

        # Plot the CMA-ES curve
        curve_cmaes = np.array(d["curve_cmaes"])
        ax.plot(
            curve_cmaes[:, 0],
            curve_cmaes[:, 1],
            color="red",
            zorder=1,
            label=f"CMA-ES (dist = {d['cost_cmaes']:.2f})",
        )

        # Plot the FMS curve
        curve_fms = np.array(d["curve_fms"])
        ax.plot(
            curve_fms[:, 0],
            curve_fms[:, 1],
            color="blue",
            zorder=1,
            label=f"FMS (dist = {d['cost_fms']:.2f})",
        )

        # Set the title and save the plot
        ax.set_title(
            f"Water level: {water_level} | Resolution: {resolution} | Seed: {random_seed}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{folder}/land_avoidance_{idx}.png")
        plt.close(fig)


def main():
    """Execute the necessary operations for generating paper plots."""
    land_configurations()
    land_avoidance()


if __name__ == "__main__":
    main()
