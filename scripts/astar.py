import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import typer

from routetools.cmaes import cost_function
from routetools.fms import optimize_fms
from routetools.vectorfield import vectorfield_swirlys


def main(file: str = "/home/daniprec/weather-routing-research/output/astar_jit.csv"):
    vectorfield = vectorfield_swirlys

    df = pd.read_csv(file)
    x = jnp.asarray(df["lon"])
    y = jnp.asarray(df["lat"])
    t = jnp.asarray(df["time"])
    curve = jnp.stack([x, y], axis=1)
    src = jnp.array([x[0], y[0]]).round(0)
    dst = jnp.array([x[-1], y[-1]]).round(0)
    travel_time = 30  # float(jnp.nansum(t).round(0))
    print("Travel time:", travel_time)

    curve_fms = optimize_fms(
        vectorfield=vectorfield,
        curve=curve,
        travel_stw=None,
        travel_time=travel_time,
        damping=0.9,
        tolfun=1e-6,
    )[0]

    cost_astar = float(
        cost_function(
            vectorfield=vectorfield, curve=curve[None, ...], travel_time=travel_time
        ).sum()
    )
    cost_fms = float(
        cost_function(
            vectorfield=vectorfield, curve=curve_fms[None, ...], travel_time=travel_time
        ).sum()
    )

    xmin, xmax = 0, 6
    ymin, ymax = -1, 6

    x: jnp.ndarray = jnp.arange(xmin, xmax, 0.5)
    y: jnp.ndarray = jnp.arange(ymin, ymax, 0.5)
    X, Y = jnp.meshgrid(x, y)
    U, V = vectorfield(X, Y)

    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.plot(src[0], src[1], "o", color="blue")
    plt.plot(dst[0], dst[1], "o", color="green")
    plt.plot(curve[:, 0], curve[:, 1], color="red", label=f"A*: {cost_astar:.2f}")
    plt.plot(
        curve_fms[:, 0], curve_fms[:, 1], color="orange", label=f"FMS: {cost_fms:.2f}"
    )
    plt.legend(title="Cost")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.savefig("output/astar_jit.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
