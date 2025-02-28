import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve

a, b, c = 0.04, 1e4, 3e7
tol = 1e-14  # Newton iteration tolerance
itermax = 20  # Maximum iterations
y0 = np.array([1.0, 0.0, 0.0])
def func(y):
    dy = np.zeros(3)
    byz = b * y[1] * y[2]
    cy2 = c * y[1] ** 2
    ax = a * y[0]
    dy[0] = -ax + byz
    dy[1] = ax - byz - cy2
    dy[2] = cy2
    return dy
def Jac(y):
    by = b * y[1]
    bz = b * y[2]
    c2y = 2 * c * y[1]
    J = np.zeros((3, 3))
    J[0, 0] = -a
    J[0, 1] = bz
    J[0, 2] = by
    J[1, 0] = a
    J[1, 1] = -bz - c2y
    J[1, 2] = -by
    J[2, 1] = c2y
    return J

gamma2 = 1.0 - 1.0 / np.sqrt(2)

def NewtonIterDIRK2(y, h, k, gamma):
    aux = y + h * gamma * k
    F = k - func(aux)
    DF = np.identity(3) - h * gamma * Jac(aux)
    return k - np.linalg.solve(DF, F)

def DIRK2_step(y, h):
    k1 = func(y)
    for _ in range(itermax):
        k1 = NewtonIterDIRK2(y, h, k1, gamma2)
        if np.linalg.norm(k1 - func(y + h * gamma2 * k1)) < tol:
            break
    y = y + h * (1 - gamma2) * k1
    k2 = k1
    for _ in range(itermax):
        k2 = NewtonIterDIRK2(y, h, k2, gamma2)
        if np.linalg.norm(k2 - func(y + h * gamma2 * k2)) < tol:
            break
    return y + h * gamma2 * k2

### DIRK3 Solver ###
gamma3 = 0.5 + np.sqrt(3) / 6

def NewtonIterDIRK3(y, h, k, gamma):
    aux = y + h * gamma * k
    F = k - func(aux)
    DF = np.identity(3) - h * gamma * Jac(aux)
    return k - np.linalg.solve(DF, F)

def DIRK3_step(y, h):
    k1 = func(y)
    for _ in range(itermax):
        k1 = NewtonIterDIRK3(y, h, k1, gamma3)
        if np.linalg.norm(k1 - func(y + h * gamma3 * k1)) < tol:
            break
    
    y_stage = y + h * (1 - 2 * gamma3) * k1
    k2 = func(y_stage)
    for _ in range(itermax):
        k2 = NewtonIterDIRK3(y_stage, h, k2, gamma3)
        if np.linalg.norm(k2 - func(y_stage + h * gamma3 * k2)) < tol:
            break

    return y + 0.5 * h * (k1 + k2)

### BDF2 Solver ###
def BDF2_step(y_prev, y, h):
    def G(U):
        return U - (4/3) * y + (1/3) * y_prev - (2/3) * h * func(U)
    
    U_new = fsolve(G, y)  # Solve nonlinear equation
    return U_new

h_values = [1e-3, 1e-2, 1e-1]
cpu_times = {"DIRK2": [], "DIRK3": [], "BDF2": []}

for h in h_values:
    Nsteps = int(np.ceil(100 / h))
    t_values = np.arange(0, (Nsteps + 1) * h, h)

    # DIRK2
    y_DIRK2 = np.zeros((Nsteps + 1, 3))
    y_DIRK2[0] = y0
    start_time = time.time()
    for j in range(Nsteps):
        y_DIRK2[j + 1] = DIRK2_step(y_DIRK2[j], h)
    cpu_times["DIRK2"].append(time.time() - start_time)

    # DIRK3
    y_DIRK3 = np.zeros((Nsteps + 1, 3))
    y_DIRK3[0] = y0
    start_time = time.time()
    for j in range(Nsteps):
        y_DIRK3[j + 1] = DIRK3_step(y_DIRK3[j], h)
    cpu_times["DIRK3"].append(time.time() - start_time)

    # BDF2
    y_BDF2 = np.zeros((Nsteps + 1, 3))
    y_BDF2[0] = y0
    y_BDF2[1] = DIRK2_step(y_BDF2[0], h)  
    start_time = time.time()
    for j in range(1, Nsteps):
        y_BDF2[j + 1] = BDF2_step(y_BDF2[j - 1], y_BDF2[j], h)
    cpu_times["BDF2"].append(time.time() - start_time)

    # Plot solutions in subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    # X Concentration
    axs[0].plot(t_values, y_DIRK2[:, 0], label="x (DIRK2)", linestyle="--")
    axs[0].plot(t_values, y_DIRK3[:, 0], label="x (DIRK3)")
    axs[0].plot(t_values, y_BDF2[:, 0], label="x (BDF2)", linestyle=":")
    axs[0].set_ylabel("Concentration of X")
    axs[0].legend()
    
    # Y Concentration
    axs[1].plot(t_values, y_DIRK2[:, 1], linestyle="--")
    axs[1].plot(t_values, y_DIRK3[:, 1])
    axs[1].plot(t_values, y_BDF2[:, 1], linestyle=":")
    axs[1].set_ylabel("Concentration of Y")
    
    # Z Concentration
    axs[2].plot(t_values, y_DIRK2[:, 2], linestyle="--")
    axs[2].plot(t_values, y_DIRK3[:, 2])
    axs[2].plot(t_values, y_BDF2[:, 2], linestyle=":")
    axs[2].set_ylabel("Concentration of Z")
    axs[2].set_xlabel("Time")

    fig.suptitle(f"Solutions for h={h}")
    plt.show()

# CPU Time Comparison
plt.figure(figsize=(8, 6))
plt.loglog(h_values, cpu_times["DIRK2"], marker="o", label="DIRK2")
plt.loglog(h_values, cpu_times["DIRK3"], marker="s", label="DIRK3")
plt.loglog(h_values, cpu_times["BDF2"], marker="^", label="BDF2")
plt.xlabel("Step Size (h)")
plt.ylabel("CPU Time (seconds)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.title("CPU Time vs Step Size")
plt.show()
