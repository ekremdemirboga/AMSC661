import numpy as np
import matplotlib.pyplot as plt
from distmesh import *
import math

# 1. L-Shape
# def fd_lshape(p):
#     d1 = drectangle(p, 0, 2, 0, 1)
#     d2 = drectangle(p, 0, 1, 0, 2)
#     return ddiff(d1, d2)

def fd_lshape(p):
    d1 = drectangle(p, 0, 2, 0, 2)
    d2 = drectangle(p, 0, 1, 0, 1)
    return dintersect(d1,-d2)  # Using intersect

# 2. Pentagon with a hole
def fd_pentagon_hole(p):
    r1 = 1  # Radius of outer pentagon
    r2 = 0.5  # Radius of inner pentagon

    # Vertices of outer pentagon
    outer_points = []
    for i in range(5):
        outer_points.append([r1 * np.cos(i * 2 * np.pi / 5),
                             r1 * np.sin( i * 2 * np.pi / 5)])

    # Distance functions for the edges (more precise)
    outer_segments = []
    for i in range(5):
        p1 = outer_points[i]
        p2 = outer_points[(i + 1) % 5]
        a = p2 - np.array(p1)
        b = p - p1
        d = np.abs(a[0] * b[:, 1] - a[1] * b[:, 0]) / np.linalg.norm(a)
        outer_segments.append(d)
    d_outer = -np.min(outer_segments)  # Inside pentagon is negative distance

    d_inner = dcircle(p, 0, 0, r2)
    return dintersect(d_outer, d_inner)  # Intersection for clean definition

# 3. Half-circle with two holes
def fd_half_circle_holes(p):
    d1 = dcircle(p, 0, 0, 1)
    d2 = dcircle(p, -0.5, 0.5, 0.2)
    d3 = dcircle(p, 0.5, 0.5, 0.2)
    d4 = drectangle(p, -1, 1, 0, 1)
    return dintersect(d4, ddiff(d1, dunion(d2, d3)))

# Example Usage
# You can adjust h0 and bbox as needed

# L-Shape
bbox_lshape = [0, 2, 0, 2]
h0_lshape = 0.1
pfix_lshape = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [0, 2]])  # fixing corner points helps
pts_lshape, tri_lshape = distmesh2D(fd_lshape, huniform, h0_lshape, bbox_lshape, pfix_lshape)

# Pentagon with a hole
bbox_pentagon = [-1.5, 1.5, -1.5, 1.5]
h0_pentagon = 0.1
pts_pentagon, tri_pentagon = distmesh2D(fd_pentagon_hole, huniform, h0_pentagon, bbox_pentagon, [])

# Half-circle with holes
bbox_halfcircle = [-1, 1, 0, 1]
h0_halfcircle = 0.1
pts_halfcircle, tri_halfcircle = distmesh2D(fd_half_circle_holes, huniform, h0_halfcircle, bbox_halfcircle, [])

# Plotting (example for L-shape)
plt.figure(figsize=(6, 6))
plt.gca().set_aspect('equal')
plt.triplot(pts_lshape[:, 0], pts_lshape[:, 1], tri_lshape)
plt.title("L-Shape Mesh")
plt.show()

# Plotting (example for Pentagon)
plt.figure(figsize=(6, 6))
plt.gca().set_aspect('equal')
plt.triplot(pts_pentagon[:, 0], pts_pentagon[:, 1], tri_pentagon)
plt.title("Pentagon with Hole Mesh")
plt.show()

# Plotting (example for Half-circle)
plt.figure(figsize=(6, 6))
plt.gca().set_aspect('equal')
plt.triplot(pts_halfcircle[:, 0], pts_halfcircle[:, 1], tri_halfcircle)
plt.title("Half-circle with Holes Mesh")
plt.show()