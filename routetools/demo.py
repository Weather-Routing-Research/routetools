import re

import jax.numpy as jnp
import matplotlib.pyplot as plt
import streamlit as st

from routetools.cmaes import optimize
from routetools.fms import optimize_fms

# User can input the vector field formula in the form
# u(x, y) and v(x, y) are the components of the vector field
# x and y are the coordinates
st.write("Enter the vector field formula:")
col1, col2 = st.columns(2)
with col1:
    u = st.text_input("u(x, y)", value="sin(y)")
with col2:
    v = st.text_input("v(x, y)", value="cos(x)")


# Functions are turned into jax.numpy functions
# use regex to find any non-x or y characters and add jnp. to them
# e.g. sin(y) -> jnp.sin(y)
u_regex = re.sub(
    r"(?<![a-zA-Z0-9_])(sin|cos|tan|exp|log|sqrt|abs|arcsin|arccos|arctan|sinh|cosh|tanh|arcsinh|arccosh|arctanh)\b",
    r"jnp.\1",
    u,
)
v_regex = re.sub(
    r"(?<![a-zA-Z0-9_])(sin|cos|tan|exp|log|sqrt|abs|arcsin|arccos|arctan|sinh|cosh|tanh|arcsinh|arccosh|arctanh)\b",
    r"jnp.\1",
    v,
)


# Turn the formula into a JAX-compatible function
def vectorfield(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.array(eval(u_regex)), jnp.array(eval(v_regex))


# Pick start and destination points
st.write("Enter the start and destination points:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    x0 = st.number_input("x0", value=0.0)
with col2:
    y0 = st.number_input("y0", value=0.0)
with col3:
    x1 = st.number_input("x1", value=5.0)
with col4:
    y1 = st.number_input("y1", value=5.0)

xmin, xmax = min(x0, x1) - 5, max(x0, x1) + 5
ymin, ymax = min(y0, y1) - 5, max(y0, y1) + 5

x: jnp.ndarray = jnp.arange(xmin, xmax, 0.5)
y: jnp.ndarray = jnp.arange(ymin, ymax, 0.5)
X, Y = jnp.meshgrid(x, y)
U, V = vectorfield(X, Y)

# Quiver plot of the vector field
fig, ax = plt.subplots()
ax.quiver(X, Y, U, V)


# Run CMA-ES to find the optimal path
col10, col11, col12, col13 = st.columns(4)
with col10:
    optimize_time = (
        st.radio("Optimize", ("Travel time", "Speed through water")) == "Travel time"
    )
with col11:
    if optimize_time:
        # Slider to choose the travel time
        travel_time = st.slider("Travel time", 1, 100)
        travel_stw = None
    else:
        # Slider to choose the speed through water
        stw_min = float(jnp.sqrt(U**2 + V**2).max())
        travel_stw = st.slider("Speed through water", stw_min, stw_min * 10)
        travel_time = None
with col12:
    button = st.button("Run CMA-ES")
with col13:
    button_fms = st.button("Run FMS") if button else False

if button:
    curve = optimize(
        vectorfield,
        jnp.array([x0, y0]),
        jnp.array([x1, y1]),
        travel_stw=travel_stw,
        travel_time=travel_time,
        verbose=False,
    )
elif button_fms:
    curve = optimize_fms(
        vectorfield,
        curve=curve,
        travel_stw=travel_stw,
        travel_time=travel_time,
        damping=0.9 / travel_time if travel_time else 0.9 / travel_stw,
        verbose=False,
    )[0]

if button or button_fms:
    # Plot the optimal path
    ax.plot(curve[:, 0], curve[:, 1], color="red")

ax.plot(x0, y0, "bo")
ax.plot(x1, y1, "ro")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
st.pyplot(fig)
plt.close(fig)
