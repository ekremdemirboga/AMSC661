import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
mu_values = [1e2, 1e3, 1e6]  # Different mu values to test
Tmax = 200
h = 1e-3  # Time step

gamma = 1 - 1/np.sqrt(2)  # DIRK2 parameter
tol = 1e-14
itermax = 20

def func(y, mu):
    """Computes the right-hand side of the van der Pol oscillator."""
    x, v = y
    return np.array([v, mu * (1 - x**2) * v - x])

def Jac(y, mu):
    """Computes the Jacobian matrix."""
    x, v = y
    return np.array([[0, 1], [-1 - 2 * mu * x * v, mu * (1 - x**2)]])

def NewtonIterDIRK2(y, h, k, gamma, mu):
    """Performs a Newton iteration for DIRK2."""
    aux = y + h * gamma * k
    F = k - func(aux, mu)
    DF = np.identity(2) - h * gamma * Jac(aux, mu)
    return k - np.linalg.solve(DF, F)

def DIRK2step(y, h, mu):
    """Performs a single step of the DIRK2 method."""
    k1 = func(y, mu)
    for _ in range(itermax):
        k1 = NewtonIterDIRK2(y, h, k1, gamma, mu)
        if np.linalg.norm(k1 - func(y + h * gamma * k1, mu)) < tol:
            break
    
    y = y + h * (1 - gamma) * k1
    k2 = k1
    for _ in range(itermax):
        k2 = NewtonIterDIRK2(y, h, k2, gamma, mu)
        aux = y + h * gamma * k2
        if np.linalg.norm(k2 - func(aux, mu)) < tol:
            break
    
    return aux

# Solve using fixed step DIRK2 for different mu values
for mu in mu_values:
    Nsteps = int(np.ceil(Tmax/h))
    y = np.zeros((Nsteps+1, 2))
    t = np.arange(0, (Nsteps+1)*h, h)
    y[0] = [2.0, 0.0]

    start_time = time.time()
    for j in range(Nsteps):
        y[j+1] = DIRK2step(y[j], h, mu)
    end_time = time.time()

    t_cpu = end_time - start_time
    print(f'DIRK2 fixed step (mu={mu}): CPU time = {t_cpu:.6e} seconds')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(t, y[:, 0], label='x')
    ax[0].plot(t, y[:, 1], label='y')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Values')
    ax[0].legend()
    ax[0].set_title(f'Time Evolution (mu={mu})')
    
    ax[1].plot(y[:, 0], y[:, 1])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title(f'Phase Space (mu={mu})')
    
    plt.suptitle(f'Van der Pol Oscillator for mu={mu}')
    plt.show()
