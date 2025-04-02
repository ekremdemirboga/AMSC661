import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def solve_heat_distribution():
    """
    Solves the heat distribution problem numerically and plots the result.
    """

    nx = 50  # Number of points in x-direction
    ny = 30  # Number of points in y-direction
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

            A[row, row] = -2 / dx**2 - 2 / dy**2

            if i > 0:
                A[row, row - 1] = 1 / dx**2
            else:
                A[row, row + nx - 1] = 1 / dx**2  # Periodic BC

            if i < nx - 1:
                A[row, row + 1] = 1 / dx**2
            else:
                A[row, row - nx + 1] = 1 / dx**2  # Periodic BC

            if j > 0:
                A[row, row - nx] = 1 / dy**2
            if j < ny - 1:
                A[row, row + nx] = 1 / dy**2


    for i in range(nx):
        row = i
        A[row, :] = 0
        A[row, row] = 1

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

    u_vector = spsolve(A.tocsr(), f_vector)
    u = u_vector.reshape(ny, nx)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, u, levels=14, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Stationary Temperature Distribution')
    plt.colorbar(contour)
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    solve_heat_distribution()