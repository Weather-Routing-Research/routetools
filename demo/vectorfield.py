import jax.numpy as jnp


def vectorfield_fourvortices(
    x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def Ru(a, b):
        return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * -(y - b)

    u = 1.7 * (-Ru(2, 2) - Ru(4, 4) - Ru(2, 5) + Ru(5, 1))

    def Rv(a, b):
        return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * (x - a)

    v = 1.7 * (-Rv(2, 2) - Rv(4, 4) - Rv(2, 5) + Rv(5, 1))
    return u, v


def vectorfield_swirlys(
    x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    u = jnp.cos(2 * x - y - 6)
    v = 2 / 3 * jnp.sin(y) + x - 3
    return u, v
