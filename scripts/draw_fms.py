import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.lines import Line2D

from routetools.fms import optimize_fms
from routetools.vectorfield import vectorfield_zero


def main(w: float = 2.0, maxfevals: int = 50, damping: float = 0.1, frames: int = 200):
    """Draw the FMS optimization.

    Parameters
    ----------
    w : float, optional
        The noise level, by default 2.0
    maxfevals : int, optional
        The maximum number of iterations, by default 50
    damping : float, optional
        The damping factor, by default 0.1
    frames : int, optional
        The number of frames, by default 200
    """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    for ax in axs.flatten():
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)

    # Initial route straight from (0,0) to (6,6)
    x = np.linspace(0, 6, 100)
    y = np.linspace(0, 6, 100)
    routes = np.stack([x, y], axis=1)
    # Replicate (100, 2) to (4, 100, 2)
    routes = np.repeat(routes[None, ...], 4, axis=0)

    # Route 0: Add random noise to the X-axis
    noise_x = w * np.random.normal(0, 1, (98,))
    routes[0, 1:99, 0] = routes[0, 1:99, 0] + noise_x
    
    # Route 1: Add random noise to the Y-axis
    noise_y = w * np.random.normal(0, 1, (98,))
    routes[1, 1:99, 1] = routes[1, 1:99, 1] + noise_y
    
    # Route 2: Add random noise to both the X-axis and Y-axis
    noise_xy = w * np.random.normal(0, 1, (98, 2))
    routes[2, 1:99] = routes[2, 1:99] + noise_xy
    
    # Route 3: Add a sinusoidal noise to the X-axis
    sin_noise = w * np.sin(np.pi * np.linspace(0, 2, 100))
    routes[3, :, 0] = routes[3, :, 0] + sin_noise

    # Initialize list of lines
    ls_lines = []
    ls_txt = []

    for idx, ax in enumerate(axs.flatten()):
        (line,) = ax.plot(routes[idx, :, 0], routes[idx, :, 1], "r-", marker="o")
        txt_iter = ax.text(0.5, 5.5, "Iteration: 0", fontsize=12, color="black")
        txt_cost = ax.text(0.5, 5.0, "Cost: ?", fontsize=12, color="black")
        ls_lines.append(line)
        ls_txt.append((txt_iter, txt_cost))

    def animate(frame: int) -> list[Line2D]:
        """Animate the FMS optimization.

        Parameters
        ----------
        frame : int
            The frame number

        Returns
        -------
        list[Line2D]
            List of lines to animate
        """
        nonlocal routes
        # Run the FMS for one step
        routes, costs = optimize_fms(
            vectorfield_zero,
            curve=routes,
            damping=damping,
            travel_stw=1,
            maxfevals=maxfevals,
            verbose=False,
        )
        for idx in range(4):
            ls_lines[idx].set_data(routes[idx, :, 0], routes[idx, :, 1])
            ls_txt[idx][0].set_text(f"Iteration: {frame*maxfevals}")
            ls_txt[idx][1].set_text(f"Cost: {costs[idx]:.2f}")
        return ls_lines + [txt for txt, _ in ls_txt] + [txt for _, txt in ls_txt]

    anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True)

    # Save the animation
    anim.save("output/fms.gif", writer="pillow", fps=10)


if __name__ == "__main__":
    typer.run(main)
