import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from routetools.land import Land
from routetools.plot import plot_route_from_json, plot_table_aggregated

LATEX_CORRELATION = r"""
\begin{table}[htbp]
\caption{Pearson correlation coefficient (PCC) between the percentage error (PE)
produced by  \name{} (compared with the minimum distance), its computation time
and the different parameters of this algorithm.}
\label{tab:correlation}
\begin{tabular}{lrr}
\textbf{Configuration Parameters} & \textbf{PE} & \textbf{Compute time} \\
\toprule
Population size, P & $POPLOSS$ & $POPTIME$ \\
Standard deviation, $\sigma_0$ & $SIGMALOSS$ & $SIGMATIME$ \\
Control points, K & $KLOSS$ & $KTIME$ \\
Waypoints, L & $LLOSS$ & $LTIME$ \\ \bottomrule
\end{tabular}
\end{table}
"""

LATEX_COMPUTATION_TIMES = r"""
\begin{table}[htbp]
\centering
\caption{Mean computation times for different land configurations.}
\label{tab:computation_times}
\begin{tabular}{lllrr}
\textbf{Cost} & \textbf{Time} & \textbf{Water}
& \textbf{CMA-ES} & \textbf{FMS}\\
\textbf{function} & \textbf{dependent} & \textbf{level}
& \textbf{time (s)} & \textbf{time (s)}\\
\toprule
Time & No & 1.0 & $10TNC$ & $10TNF$ \\
& & 0.9 & $9TNC$ & $9TNF$ \\
& & 0.8 & $8TNC$ & $8TNF$ \\
& & 0.7 & $7TNC$ & $7TNF$ \\
\midrule
Time & Yes & 1.0 & $10TYC$ & $10TYF$ \\
& & 0.9 & $9TYC$ & $9TYF$ \\
& & 0.8 & $8TYC$ & $8TYF$ \\
& & 0.7 & $7TYC$ & $7TYF$ \\
\midrule
Fuel & & 1.0 & $10FNC$ & $10FNF$ \\
& & 0.9 & $9FNC$ & $9FNF$ \\
& & 0.8 & $8FNC$ & $8FNF$ \\
& & 0.7 & $7FNC$ & $7FNF$ \\
\bottomrule
\end{tabular}
\end{table}
"""


def plot_land_configurations(
    seed: int = 4, fout: str = "output/land_configurations.png"
):
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
    plt.close(fig)
    print(f"Land configurations saved to {fout}")


def plot_land_avoidance(folder: str = "output"):
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
        print(f"Land avoidance: processing {json_id}...")

        # Print what was the CMA-ES configuration
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

    # ---- Land avoidance ----

    mask = (df["vectorfield"] == "zero") & (df["water_level"] < 1.0)

    # Filter the rows with zero vectorfield and water level < 1.0
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
        round_decimals=0,
        title=f"Number of optimal solutions (out of {n})",
        cmap="RdYlGn",
        figsize=(6, 6),
    )
    fig.savefig(f"{folder}/parameter_search_land_avoidance.png")
    plt.close(fig)
    print(f"Land avoidance table saved to {folder}/parameter_search_land_avoidance.png")

    # ---- Vectorfields ----

    mask = df["vectorfield"] != "zero"

    # Filter the rows with zero vectorfield and water level < 1.0
    df_filtered = df[mask].copy()

    # Count the unique combinations of the parameters
    cols = ["popsize", "sigma0", "K", "L"]
    n = int(df_filtered.groupby(cols).size().mean())

    fig, ax = plot_table_aggregated(
        df_filtered,
        "percterr_fms",
        ["popsize", "sigma0"],
        ["K", "L"],
        agg="mean",
        round_decimals=0,
        title=f"Mean percentage error (out of {n} problem instances)",
        cmap="RdYlGn_r",
        figsize=(6, 6),
    )
    fig.savefig(f"{folder}/parameter_search_vectorfields.png")
    plt.close(fig)
    print(f"Vectorfields table saved to {folder}/parameter_search_vectorfields.png")

    # ---- Computation time ----

    # Count the unique combinations of the parameters
    cols = ["popsize", "sigma0", "K", "L"]
    n = int(df.groupby(cols).size().mean())

    fig, ax = plot_table_aggregated(
        df,
        "comp_time",
        ["popsize", "sigma0"],
        ["K", "L"],
        agg="mean",
        round_decimals=0,
        title="Computation time (in seconds)",
        cmap="RdYlGn_r",
        figsize=(6, 6),
    )
    fig.savefig(f"{folder}/parameter_search_time.png")
    plt.close(fig)
    print(f"Computation time table saved to {folder}/parameter_search_time.png")


def table_parameter_search_correlation(folder: str = "output"):
    """Generate a LaTeX table with the correlation between loss and parameters.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file, by default "output".
    """
    path_csv = f"{folder}/results.csv"
    df = pd.read_csv(path_csv)

    dict_masks = {
        "land_avoidance": (df["vectorfield"] == "zero") & (df["water_level"] < 1.0),
        "vectorfields": df["vectorfield"] != "zero",
    }

    for name, mask in dict_masks.items():
        df_filtered = df[mask]

        # Define columns of interest
        cols_param = ["popsize", "sigma0", "K", "L"]
        cols_target = ["percterr_fms", "comp_time"]

        # Compute correlation matrix
        corr: pd.DataFrame = df_filtered[cols_param + cols_target].corr()
        corr = corr.loc[cols_param, cols_target]

        # LaTeX table template
        latex_template = LATEX_CORRELATION

        # Replace placeholders with computed correlation values
        replacements = {
            "$POPLOSS$": f"{corr.loc['popsize', 'percterr_fms']:.3f}",
            "$POPTIME$": f"{corr.loc['popsize', 'comp_time']:.3f}",
            "$SIGMALOSS$": f"{corr.loc['sigma0', 'percterr_fms']:.3f}",
            "$SIGMATIME$": f"{corr.loc['sigma0', 'comp_time']:.3f}",
            "$KLOSS$": f"{corr.loc['K', 'percterr_fms']:.3f}",
            "$KTIME$": f"{corr.loc['K', 'comp_time']:.3f}",
            "$LLOSS$": f"{corr.loc['L', 'percterr_fms']:.3f}",
            "$LTIME$": f"{corr.loc['L', 'comp_time']:.3f}",
        }

        for key, value in replacements.items():
            latex_template = latex_template.replace(key, value)

        # Save to a text file - keep the LaTeX format
        filename = f"{folder}/parameter_search_{name}.tex"
        with open(filename, "w") as file:
            file.write(latex_template)

        print(f"LaTeX table saved to {filename}")


def plot_best_no_land(
    folder: str = "output",
    L: int = 256,
    K: int = 6,
    popsize: int = 500,
    sigma0: int = 1,
    num_pieces: int = 1,
):
    """Generate plots for the best examples without land avoidance.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file and JSON files,
        by default "output".
    """
    path_csv = f"{folder}/results.csv"
    df = pd.read_csv(path_csv)

    mask = (
        (df["water_level"] == 1.0)
        & (df["L"] == L)
        & (df["K"] == K)
        & (df["popsize"] == popsize)
        & (df["sigma0"] == sigma0)
        & (df["num_pieces"] == num_pieces)
    )

    # Filter the rows with highest "gain_fms", grouped by vectorfield
    df_filtered = (
        df[mask]
        .groupby("vectorfield")[["vectorfield", "json", "gain_fms"]]
        .apply(lambda x: x.nlargest(1, "gain_fms"))
        .reset_index(drop=True)
        .sort_values("gain_fms", ascending=False)
    )

    # Plot the top examples
    for idx in df_filtered.index:
        row = df_filtered.iloc[idx]
        vf = row["vectorfield"]
        json_id = int(row["json"])
        print(f"Best without land avoidance: processing {json_id}...")
        fig, ax = plot_route_from_json(f"{folder}/json/{json_id:06d}.json")
        fig.savefig(f"{folder}/best_{vf}.png")
        plt.close(fig)


def plot_biggest_difference(
    folder: str = "output",
    L: int = 256,
    K: int = 6,
    popsize: int = 500,
    sigma0: int = 1,
    num_pieces: int = 1,
):
    """Generate plots for the examples with the biggest FMS savings.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file and JSON files,
        by default "output".
    """
    path_csv = f"{folder}/results.csv"
    df = pd.read_csv(path_csv)

    mask = (
        (df["gain_fms"] < 20)
        & (df["L"] == L)
        & (df["K"] == K)
        & (df["popsize"] == popsize)
        & (df["sigma0"] == sigma0)
        & (df["num_pieces"] == num_pieces)
    )

    # Filter the rows with highest "gain_fms", grouped by vectorfield
    df_filtered = (
        df[mask]
        .groupby("vectorfield")[["json", "vectorfield", "gain_fms"]]
        .apply(lambda x: x.nlargest(2, "gain_fms"))
        .reset_index(drop=True)
        .sort_values("gain_fms", ascending=False)
    )

    # Plot the top examples
    for idx in df_filtered.index:
        row = df_filtered.iloc[idx]
        json_id = int(row["json"])
        vf = row["vectorfield"]
        print(f"Biggest FMS savings: processing {json_id}...")
        fig, ax = plot_route_from_json(f"{folder}/json/{json_id:06d}.json")
        fig.savefig(f"{folder}/biggest_fms_{vf}_{idx}.png")
        plt.close(fig)


def table_computation_times(folder: str = "output"):
    """Generate a LaTeX table with the computation times for different configurations.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file, by default "output".
    """
    path_csv = f"{folder}/results.csv"
    df = pd.read_csv(path_csv)

    # Identify time dependent vector fields
    df["time_dependent"] = df["vectorfield"] == "techy"
    # Identify cost function via the input parameters
    df["cost_function"] = df["travel_stw"].isna().replace({False: "time", True: "fuel"})

    # Initialize dictionary for replacing placeholders
    replacements = {}

    for water_level in [1.0, 0.9, 0.8, 0.7]:
        for cost_function in ["time", "fuel"]:
            mask = (df["water_level"] == water_level) & (
                df["cost_function"] == cost_function
            )

            df_filtered = df[mask]
            mask_td = df_filtered["time_dependent"]

            wl = int(water_level * 10)
            cf = "T" if cost_function == "time" else "F"

            # Compute mean computation time
            df_nc = df_filtered.loc[~mask_td, "comp_time_cmaes"]
            replacements[f"${wl}{cf}NC$"] = rf"{df_nc.mean():.0f} \pm {df_nc.std():.0f}"
            df_yc = df_filtered.loc[mask_td, "comp_time_cmaes"]
            replacements[f"${wl}{cf}YC$"] = rf"{df_yc.mean():.0f} \pm {df_yc.std():.0f}"
            df_nf = df_filtered.loc[~mask_td, "comp_time_fms"]
            replacements[f"${wl}{cf}NF$"] = rf"{df_nf.mean():.0f} \pm {df_nf.std():.0f}"
            df_yf = df_filtered.loc[mask_td, "comp_time_fms"]
            replacements[f"${wl}{cf}YF$"] = rf"{df_yf.mean():.0f} \pm {df_yf.std():.0f}"

    # LaTeX table template
    latex_template = LATEX_COMPUTATION_TIMES

    # Replace placeholders with computed correlation values
    for key, value in replacements.items():
        latex_template = latex_template.replace(key, rf"${value}$")

    # Save to a text file - keep the LaTeX format
    filename = f"{folder}/computation_times.tex"
    with open(filename, "w") as file:
        file.write(latex_template)

    print(f"LaTeX table saved to {filename}")


def main(
    folder: str = "output",
    L: int = 301,
    K: int = 7,
    popsize: int = 500,
    sigma0: int = 1,
    num_pieces: int = 1,
):
    """Execute the necessary operations for generating paper plots."""
    print("--- PLOT LAND CONFIGURATIONS ---")
    plot_land_configurations(fout=f"{folder}/land_configurations.png")
    print("\n--- PLOT LAND AVOIDANCE ---")
    plot_land_avoidance(folder=folder)
    print("\n--- PLOT PARAMETER SEARCH ---")
    plot_parameter_search(folder=folder)
    print("\n--- TABLE PARAMETER SEARCH CORRELATION ---")
    table_parameter_search_correlation(folder=folder)
    print("\n--- PLOT BEST NO LAND ---")
    plot_best_no_land(
        folder=folder, L=L, K=K, popsize=popsize, sigma0=sigma0, num_pieces=num_pieces
    )
    print("\n--- PLOT BIGGEST DIFFERENCE ---")
    plot_biggest_difference(
        folder=folder, L=L, K=K, popsize=popsize, sigma0=sigma0, num_pieces=num_pieces
    )
    print("\n--- TABLE COMPUTATION TIMES ---")
    table_computation_times(folder=folder)


if __name__ == "__main__":
    main()
