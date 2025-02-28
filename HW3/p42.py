import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
mu_values = [1e2, 1e3, 1e6]  # Different mu values to test
Tmax = 200
h = 1e-3  # Initial time step
atol = 1e-5
rtol = 1e-5

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

def adaptive_DIRK2(y0, mu, Tmax, h, atol, rtol):
    """Adaptive time stepping for DIRK2 method."""
    t = [0]
    y = [y0]
    
    while t[-1] < Tmax:
        if t[-1] + h > Tmax:
            h = Tmax - t[-1]  # Ensure final step lands exactly at Tmax
        
        y_next = DIRK2step(y[-1], h, mu)
        error_estimate = np.linalg.norm(y_next - y[-1]) / max(np.linalg.norm(y_next), 1e-10)
        tolerance = atol + rtol * max(np.linalg.norm(y_next), 1e-10)
        
        if error_estimate < tolerance:
            t.append(t[-1] + h)
            y.append(y_next)
        
        # Adjust step size
        h *= min(2, max(0.5, 0.9 * (tolerance / (error_estimate + 1e-10)) ** 0.5))
    
    return np.array(t), np.array(y)

# Solve using adaptive DIRK2 for mu = 1e6
mu = 1e6
Tmax = 2e6
h = 1e-3

t, y = adaptive_DIRK2(np.array([2.0, 0.0]), mu, Tmax, h, atol, rtol)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(t, y[:, 0], label='x')
ax[0].plot(t, y[:, 1], label='y')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Values')
ax[0].legend()
ax[0].set_title(f'Adaptive Time Evolution (mu={mu})')

ax[1].plot(y[:, 0], y[:, 1])
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title(f'Adaptive Phase Space (mu={mu})')

plt.suptitle(f'Van der Pol Oscillator with Adaptive Step Size for mu={mu}')
plt.show()