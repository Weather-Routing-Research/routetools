import matplotlib.pyplot as plt
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


def main():
    """Execute the necessary operations for generating paper plots."""
    land_configurations()


if __name__ == "__main__":
    main()
