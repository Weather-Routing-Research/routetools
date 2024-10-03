import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from demo.cmaes import optimize
from demo.vectorfield import vectorfield_fourvortices


def main(sigma0: float = 5, tolfun: float = 1e-6) -> None:
    """
    Time the optimization algorithm for different population sizes.

    Test for both CPU and GPU.
    """
    src = jnp.array([0, 0])
    dst = jnp.array([6, 2])

    list_gpu = [True, False]
    list_popsizes = [100, 250, 500, 1000, 2500, 5000]
    list_times = []

    plt.figure()

    for gpu in list_gpu:
        if not gpu:
            jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]
        # Check if JAX is using the GPU
        print("JAX devices:", jax.devices())
        for popsize in list_popsizes:
            start = time.time()
            optimize(
                vectorfield_fourvortices,
                src=src,
                dst=dst,
                travel_speed=1,
                travel_time=None,
                popsize=popsize,
                sigma0=sigma0,
                tolfun=tolfun,
            )
            list_times.append(time.time() - start)

        label = "GPU" if gpu else "CPU"
        plt.plot(list_popsizes, list_times, marker="o", label=label)
        list_times.clear()

    plt.xlabel("Population size")
    plt.ylabel("Computation time (s)")
    plt.legend()
    plt.savefig("output/comptime_vs_popsize.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
    typer.run(main)
