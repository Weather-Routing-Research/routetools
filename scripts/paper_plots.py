import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from routetools.land import Land
from routetools.plot import plot_route_from_json, plot_table_aggregated


def land_configurations(seed: int = 4, fout: str = "output/land_configurations.png"):
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
                xlim=(-1, 6),
                ylim=(-1, 6),
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

        # Print what was the CMA-ES configuration
        print(f"Land avoidance {idx}: {json_id}")
        fig, ax = plot_route_from_json(f"{folder}/json/{json_id:06d}.json")
        fig.savefig(f"{folder}/land_avoidance_{idx}.png")
        plt.close(fig)


def plot_parameter_search(folder: str = "output"):
    """Plot the parameter search results as tables.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file, by default "output".
    """
    path_csv = f"{folder}/results.csv"
    df = pd.read_csv(path_csv)

    # --- Land avoidance ---

    # Filter the rows with no vectorfield and land avoidance
    mask_novf = df["vectorfield"] == "zero"
    mask_land = df["water_level"] < 1.0

    # Study the effect of [popsize, L, K, sigma0] on the wrong
    mask = mask_novf & mask_land
    df_filtered = df[mask].copy()

    # Count the unique combinations of the parameters
    cols = ["popsize", "sigma0", "K", "L"]
    n = int(df_filtered.groupby(cols).size().mean())

    fig, ax = plot_table_aggregated(
        df_filtered,
        "isoptimal_fms",
        ["popsize", "sigma0"],
        ["K", "L"],
        agg="sum",
        vmin=n - 10,
        vmax=n,
        round_decimals=0,
        title=f"Number of optimal solutions (out of {n})",
    )
    fig.savefig(f"{folder}/parameter_search_land_avoidance.png")
    plt.close(fig)


def main(folder: str = "output"):
    """Execute the necessary operations for generating paper plots."""
    land_configurations(fout=f"{folder}/land_configurations.png")
    land_avoidance(folder=folder)
    plot_parameter_search(folder=folder)


if __name__ == "__main__":
    main()
