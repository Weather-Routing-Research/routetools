import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


x_min = 0
x_max = 10
y_min = 0
y_max = 10
# Define epsilon
epsilons = np.arange(0.1, 0.5, 0.05)

# Generate a few points
np.random.seed(2)
points = np.random.rand(7, 2) * 8 + 1

for epsilon in epsilons:
    # Calculate the distance between the closest two points
    distances = distance_matrix(points, points)

    distances = jnp.fill_diagonal(distances, 0, inplace = False)
    ave_distance = jnp.mean(distances)

    # Create circles with radius eps * min_distance
    circles = [Point(p).buffer(epsilon * ave_distance) for p in points]

    # Union the circles
    union_circles = unary_union(circles)

    # Plot the points and the union of circles
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], 'o')


    if isinstance(union_circles, Polygon):
        x, y = union_circles.exterior.xy
        ax.plot(x, y, 'g')
    else:
        for geom in union_circles.geoms:
            x, y = geom.exterior.xy
            ax.plot(x, y, 'g')
    plt.gca().set_aspect('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(f'output/land_gen_with_{epsilon}.png')