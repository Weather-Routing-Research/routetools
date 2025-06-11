import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

COST_LITERATURE = {
    "circular": 1.98,
    "fourvortices": 8.95,
    "doublegyre": 1.01,
    "techy": 1.03,
    "swirlys": 5.73,
}


def experiment_parameter_sensitivity(
    path_csv: str = "./output/results_noland.csv", folder: str = "./output/"
):
    """Plot the results of the BERS experiments by parameter sensitivity.

    Parameters
    ----------
    path_csv : str, optional
        Path to the CSV file containing the results of the experiments,
        by default "./output/results_noland.csv"
    folder : str, optional
        Path to the folder where the plots will be saved, by default "./output/"
    """
    # Read the CSV file containing the results of the experiments
    df_noland = pd.read_csv(path_csv)

    # If any "percterr_cmaes" is equal or higher than 1e10, we warn the user and drop it
    if (df_noland["percterr_cmaes"] >= 1e10).any():
        print(
            "Warning: Some percentage errors for CMA-ES are equal or higher than 1e10. "
            "These will be dropped from the analysis."
        )
        df_noland = df_noland[df_noland["cost_cmaes"] < 1e10]
        # Same with "percterr_fms"
        df_noland = df_noland[df_noland["cost_fms"] < 1e10]

    # Assign literature cost using "vectorfield"
    df_noland["cost_reference"] = df_noland["vectorfield"].map(COST_LITERATURE)

    # Compute percentage errors, clip at 0%
    df_noland["percterr_cmaes"] = (
        df_noland["cost_cmaes"] / df_noland["cost_reference"] * 100 - 100
    ).clip(lower=0)
    df_noland["percterr_fms"] = (
        df_noland["cost_fms"] / df_noland["cost_reference"] * 100 - 100
    ).clip(lower=0)

    df_noland["gain_fms"] = 100 - df_noland["cost_fms"] / df_noland["cost_cmaes"] * 100

    # We will group results by "K", "sigma0" and compute their average "percterr_cmaes"
    df_noland = (
        df_noland.groupby(["K", "sigma0"])
        .agg(
            avg_percterr_cmaes=("percterr_cmaes", "mean"),
            avg_comp_time_cmaes=("comp_time_cmaes", "mean"),
            avg_gain_fms=("gain_fms", "mean"),
            avg_comp_time_fms=("comp_time_fms", "mean"),
            avg_percterr_fms=("percterr_fms", "mean"),
            avg_comp_time=("comp_time", "mean"),
        )
        .reset_index()
    )

    def _helper_experiment_parameter_sensitivity(
        col1: str,
        col2: str,
        cmap: str,
        vmin: float = 0,
        vmax: float = 100,
        title: str = "",
    ):
        # Plot a heatmap where:
        # x-axis: "K" (number of control points for Bézier curve)
        # y-axis: "sigma0" (standard deviation of the CMA-ES distribution)
        # color: col1
        # We place the number on each cell of the heatmap, using white letters
        # Below each number, we add the computation time too (col2)
        plt.figure(figsize=(8, 6))
        ax = plt.gca()  # Get current axes
        heatmap = ax.pcolor(
            df_noland.pivot(index="sigma0", columns="K", values=col1),
            cmap=cmap,
            edgecolors="k",
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,  # Set limits for the color scale
        )
        # Add the numbers in each cell
        for (i, j), val in np.ndenumerate(
            df_noland.pivot(index="sigma0", columns="K", values=col1)
        ):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )
            # Add computation time below the percentage
            ct = df_noland.loc[
                (df_noland["K"] == df_noland["K"].unique()[j])
                & (df_noland["sigma0"] == df_noland["sigma0"].unique()[i]),
                col2,
            ].values[0]
            ax.text(
                j + 0.5,
                i + 0.3,
                f"CT: {ct:.2f}s",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
        # Set the ticks and labels for the axes
        ax.set_xticks(np.arange(len(df_noland["K"].unique())) + 0.5)
        ax.set_xticklabels(df_noland["K"].unique())
        ax.set_yticks(np.arange(len(df_noland["sigma0"].unique())) + 0.5)
        ax.set_yticklabels(df_noland["sigma0"].unique())
        ax.set_xlabel("Number of Control Points (K)")
        ax.set_ylabel("Standard Deviation of CMA-ES (sigma0)")
        ax.set_title(title)
        plt.colorbar(heatmap, label="Average Percentage of Error")
        plt.tight_layout()

    # GRAPH 1
    # color: "avg_percterr_cmaes" (average percentage of error)
    _helper_experiment_parameter_sensitivity(
        col1="avg_percterr_cmaes",
        col2="avg_comp_time_cmaes",
        cmap="coolwarm",
        vmin=0,
        vmax=100,
        title="Parameter Sensitivity of CMA-ES + Bézier",
    )
    plt.savefig(folder + "parameter_sensitivity_cmaes.png", dpi=300)
    plt.close()

    # GRAPH 2
    # color: "gain_fms" (percentage of gain by FMS compared to CMA-ES)
    _helper_experiment_parameter_sensitivity(
        col1="avg_gain_fms",
        col2="avg_comp_time_fms",
        cmap="coolwarm",
        vmin=0,
        vmax=20,
        title="Average Percentage of Gain by FMS Compared to CMA-ES",
    )
    plt.savefig(folder + "parameter_sensitivity_fms.png", dpi=300)
    plt.close()

    # GRAPH 3
    # color: "avg_percterr_fms" (average percentage of error for BERS)
    _helper_experiment_parameter_sensitivity(
        col1="avg_percterr_fms",
        col2="avg_comp_time_fms",
        cmap="coolwarm",
        vmin=0,
        vmax=100,
        title="Average Percentage of Error for BERS",
    )
    plt.savefig(folder + "parameter_sensitivity_bers.png", dpi=300)
    plt.close()


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
    folder : str, optional
        Path to the folder where the plots will be saved, by default "./output/"
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
    plt.close()


def main(folder: str = "./output/"):
    """Run the experiments and plot the results."""
    experiment_parameter_sensitivity(folder=folder)
    experiment_land_complexity(folder=folder)


if __name__ == "__main__":
    typer.run(main)  # Use Typer to run the main function
