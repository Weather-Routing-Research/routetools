import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer


def experiment_land_complexity(
    path_csv: str = "./output/results_land.csv", folder: str = "./output/"
) -> None:
    """Plot the results of the BERS experiments by land complexity level.

    The land complexity is defined by the resolution and water level:
    - Easy: resolution 3, water level 0.9
    - Medium: resolution 4, water level 0.8
    - Hard: resolution 5, water level 0.7

    Parameters
    ----------
    path_csv : str, optional
        Path to the CSV file containing the results of the experiments,
        by default "./output/results_land.csv"
    """
    # Read the CSV file containing the results of the experiments
    df_land = pd.read_csv(path_csv)

    # Define the land complexity levels based on resolution and water level
    mask_easy = (df_land["resolution"] == 3) & (df_land["water_level"] == 0.9)
    mask_medium = (df_land["resolution"] == 4) & (df_land["water_level"] == 0.8)
    mask_hard = (df_land["resolution"] == 5) & (df_land["water_level"] == 0.7)

    # Create a new column "complexity" to categorize the experiments
    df_land["complexity"] = np.nan
    df_land.loc[mask_easy, "complexity"] = 1
    df_land.loc[mask_medium, "complexity"] = 2
    df_land.loc[mask_hard, "complexity"] = 3

    # Drop the experiments not belonging to any of these categories
    df_land = df_land.dropna(subset=["complexity"])

    # Boxplot: one box for each complexity level
    # Vertical axis: "cost_fms"
    # We also overlay a line showing the average "comp_time" for each complexity level
    # Use a second y-axis for the average "comp_time" line
    plt.figure(figsize=(8, 4))

    ax = plt.gca()  # Get current axes
    ax = df_land.boxplot(
        column="cost_fms", by="complexity", grid=False, showfliers=False, ax=ax
    )
    ax.set_ylim(8.5, 11.5)
    ax.set_xlabel("Land Complexity Level (affects resolution and water level)")
    ax.set_ylabel("Travel time")
    ax.set_title("Results of BERS by Land Complexity Level")

    # Overlay average comp_time
    avg_comp_time = df_land.groupby("complexity")["comp_time"].mean()
    ax2 = ax.twinx()
    avg_comp_time.plot(ax=ax2, color="red", marker="o", linestyle="-")
    ax2.set_ylabel("Average Computation Time (s)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 30)

    # Add a grid
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.suptitle("")
    plt.xticks([1, 2, 3], ["Easy", "Medium", "Hard"])
    plt.tight_layout()
    plt.savefig(folder + "land_complexity.png", dpi=300)


def main(folder: str = "./output/"):
    """Run the experiments and plot the results."""
    experiment_land_complexity(folder=folder)


if __name__ == "__main__":
    typer.run(main)  # Use Typer to run the main function
