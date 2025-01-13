import jax.numpy as jnp
from jax import jit


@jit
def vectorfield_circular(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    intensity: float = -0.9,
    centre: tuple[float, float] = (0, 0),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with a circular pattern.

    Source: Techy 2011
    https://doi.org/10.1007/s11370-011-0092-9
    """
    x0, y0 = centre
    u = -intensity * (y - y0)
    v = intensity * (x - x0)
    return u, v


@jit
def vectorfield_fourvortices(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with four vortices.

    Source: Ferraro 2021
    https://doi.org/10.1016/j.ifacol.2021.11.097
    """

    def Ru(a: float, b: float) -> jnp.ndarray:
        return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * -(y - b)

    u = 1.7 * (-Ru(2, 2) - Ru(4, 4) - Ru(2, 5) + Ru(5, 1))

    def Rv(a: float, b: float) -> jnp.ndarray:
        return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * (x - a)

    v = 1.7 * (-Rv(2, 2) - Rv(4, 4) - Rv(2, 5) + Rv(5, 1))
    return u, v


@jit
def vectorfield_swirlys(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with periodic behaviour.

    Source: Ferraro 2021
    https://doi.org/10.1016/j.ifacol.2021.11.097
    """
    u = jnp.cos(2 * x - y - 6)
    v = 2 / 3 * jnp.sin(y) + x - 3
    return u, v


@jit
def vectorfield_techy(
    x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray, sink: float = -0.3
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with time dependency.

    Time-varying flow that includes a sink < 0, and a vortex that changes vorticity
    linearly during the maneuver.

    Source:
    "Optimal navigation in planar time-varying flow: Zermelo's problem revisited"
    Techy 2011, Fig 12a-c
    "Visir-1.b: ocean surface gravity waves and currents for
    energy-efficient navigation" Mannarini 2019, Fig 2b
    """
    vortex = t - 0.5
    u = sink * x - vortex * y
    v = vortex * x + sink * y
    return u, v


@jit
def vectorfield_zero(
    x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """No currents."""
    return jnp.zeros_like(x), jnp.zeros_like(y)
