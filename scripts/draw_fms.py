import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from routetools.fms import optimize_fms
from routetools.vectorfield import vectorfield_zero

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

for ax in axs.flatten():
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)

# Initial route straight from (0,0) to (6,6)
x = jnp.linspace(0, 6, 100)
y = jnp.linspace(0, 6, 100)
# Add random noise to the x coordinate
x = x.at[1:99].set(x[1:99] + jax.random.normal(jax.random.PRNGKey(0), (98,)))
route = jnp.stack([x, y], axis=-1)
# Replicate to (4, 100, 100)
routes = jnp.stack([route] * 4)
(line_route,) = axs[0, 0].plot(route[0, 0], route[0, 1], "r-", marker="o")
txt_iter = axs[0, 0].text(0.5, 5.5, "Iteration: 0", fontsize=12, color="black")
txt_cost = axs[0, 0].text(0.5, 5.0, "Cost: ?", fontsize=12, color="black")


def animate(frame: int) -> list[Line2D]:
    global routes
    # Run the FMS for one step
    routes, costs = optimize_fms(
        vectorfield_zero, curve=routes, travel_stw=1, maxiter=1, verbose=False
    )
    for idx in range(4):
        line_route.set_data(route[idx, 0], route[idx, 1])
        txt_iter.set_text(f"Iteration: {frame}")
        txt_cost.set_text(f"Cost: {costs[0]:.2f}")
    return [
        line_route,
    ]


anim = animation.FuncAnimation(fig, animate, frames=10, blit=True)

# Save the animation
anim.save("fms.gif", writer="pillow", fps=30)
