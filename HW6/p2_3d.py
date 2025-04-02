import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def solve_heat_distribution_cylinder():
    """
    Solves the heat distribution problem numerically and plots the result
    on the surface of a cylinder.
    """

    nx = 300  # Number of points in x-direction (around the cylinder)
    ny = 300  # Number of points in y-direction (height of the cylinder)
    dx = (2 * np.pi) / nx
    dy = 2 / ny

    x = np.linspace(-np.pi, np.pi, nx)
    y = np.linspace(0, 2, ny)

    # 2. Source term
    f = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            if -np.pi/2 <= x[i] <= np.pi/2:
                f[j, i] = -np.cos(x[i])

    A = lil_matrix((nx * ny, nx * ny))

    for j in range(ny):
        for i in range(nx):
            row = j * nx + i

            # Main coefficients
            A[row, row] = -2 / dx**2 - 2 / dy**2

            # x-direction neighbors (periodic)
            if i > 0:
                A[row, row - 1] = 1 / dx**2
            else:
                A[row, row + nx - 1] = 1 / dx**2

            if i < nx - 1:
                A[row, row + 1] = 1 / dx**2
            else:
                A[row, row - nx + 1] = 1 / dx**2

            # y-direction neighbors
            if j > 0:
                A[row, row - nx] = 1 / dy**2
            if j < ny - 1:
                A[row, row + nx] = 1 / dy**2


    # Bottom boundary (y=0): u(x,0) = 0
    for i in range(nx):
        row = i
        A[row, :] = 0
        A[row, row] = 1

    # Top boundary (y=2): du/dy = 0  (Neumann)
    for i in range(nx):
        row = (ny - 1) * nx + i
        A[row, row - nx] = 1 / dy**2
        A[row, row] = -1 / dy**2
        if i > 0:
          A[row, row-1] = 0
        if i < nx -1:
          A[row, row+1] = 0

    f_vector = f.flatten()

    f_vector[:nx] = 0  # Bottom boundary
    f_vector[(ny - 1) * nx:] = 0  # Neumann BC

    # Solve the linear system
    u_vector = spsolve(A.tocsr(), f_vector)
    u = u_vector.reshape(ny, nx)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    theta = np.linspace(-np.pi, np.pi, nx)
    z = np.linspace(0, 2, ny)
    theta, z = np.meshgrid(theta, z)
    r = 1  # Radius of the cylinder

    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Z = z

    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(u / np.max(u)),
                           rcount=ny, ccount=nx, shade=True)  # Color from u
    #surf = ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, facecolors=plt.cm.jet(u / np.max(u)), shade=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Temperature Distribution on a Cylinder')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == "__main__":
    solve_heat_distribution_cylinder()