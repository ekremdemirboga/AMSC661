import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

n = 99  # Number of interior grid points
h = 2.0 / (n + 1)
xi = np.linspace(-1, 1, n + 2)
xi_internal = xi[1:-1] # Grid points xi_1 to xi_n

t_final = 1.2
num_eval_points = int(round(t_final / 0.1)) # Calculate number of points (1.2 / 0.1 = 12)
t_eval = np.linspace(0.1, t_final, num=num_eval_points) # [0.1, 0.2, ..., 1.2]

def boussinesq_rhs(t, Y):
    """ RHS function for the ODE system dy/dt = f(t, y) """
    u = Y[:n] # u_1 to u_n
    xL = Y[n]
    xR = Y[n+1]
    xf = (xR - xL) / 2.0
    if xf <= 1e-6: # Avoid division by zero or small xf
        print(f"Warning: xf is very small ({xf}) at t={t}. Stopping integration.")
        return np.zeros_like(Y)
    u_full = np.zeros(n + 2)
    u_full[1:n+1] = u

    # Calculate boundary derivatives D_{-1} and D_1 (using u_full indices)
    D_minus1 = (4 * u_full[1] - u_full[2]) / (2 * h)
    D_1 = (u_full[n-1] - 4 * u_full[n]) / (2 * h)

    dudt = np.zeros(n)
    for j in range(n):
        idx = j + 1 
        xi_j = xi_internal[j] 

        u_j = u_full[idx]
        u_jp1 = u_full[idx+1]
        u_jm1 = u_full[idx-1]

        Du = (u_jp1 - u_jm1) / (2 * h)
        Duu = (u_jp1 - 2 * u_j + u_jm1) / (h**2)

        boundary_term = -0.5 * ((1 + xi_j) * D_1 + (1 - xi_j) * D_minus1) * Du

        dudt[j] = (1.0 / xf**2) * (boundary_term + u_j * Duu + Du**2)
    dxLdt = -D_minus1 / xf
    dxRdt = -D_1 / xf
    return np.concatenate((dudt, [dxLdt, dxRdt]))

# --- Initial Conditions ---
xL0 = -1.0
xR0 = 1.0

# IC 1: u(x,0) = 1 - x^2
u0_ic1 = 1.0 - xi_internal**2
Y0_ic1 = np.concatenate((u0_ic1, [xL0, xR0]))

# IC 2: u(x,0) = 1 - 0.99 * cos(2*pi*x)
u0_ic2 = 1.0 - 0.99 * np.cos(2 * np.pi * xi_internal)
Y0_ic2 = np.concatenate((u0_ic2, [xL0, xR0]))


print("Solving for IC 1...")
sol_ic1 = solve_ivp(boussinesq_rhs, [0, t_final], Y0_ic1, t_eval=t_eval, method='BDF', rtol=1e-5, atol=1e-8)
print("Solving for IC 2...")
sol_ic2 = solve_ivp(boussinesq_rhs, [0, t_final], Y0_ic2, t_eval=t_eval, method='BDF', rtol=1e-5, atol=1e-8)

print("Solution status IC1:", sol_ic1.message)
print("Solution status IC2:", sol_ic2.message)


def generate_plots(sol, ic_label):
    if not sol.success:
        print(f"Solver failed for {ic_label}. Skipping plots.")
        return

    u_sol = sol.y[:n, :] 
    xL_sol = sol.y[n, :]
    xR_sol = sol.y[n+1, :]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i, t in enumerate(sol.t):
        xf = (xR_sol[i] - xL_sol[i]) / 2.0
        x0 = (xL_sol[i] + xR_sol[i]) / 2.0
        x_coords = x0 + xi * xf
        u_full = np.zeros(n + 2)
        u_full[1:n+1] = u_sol[:, i]
        plt.plot(x_coords, u_full, label=f't={t:.1f}')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f'Solution u(x, t) - {ic_label}')
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    xi_plot = xi
    xi_sq = 1 - xi_plot**2 # Reference parabola 1 - xi^2
    plt.plot(xi_plot, xi_sq, 'k--', linewidth=3, label=r'$1-\xi^2$ (Reference)')

    for i, t in enumerate(sol.t):
        u_full = np.zeros(n + 2)
        u_full[1:n+1] = u_sol[:, i]
        u_max = np.max(u_full)
        if u_max > 1e-6: # Avoid division by zero
             plt.plot(xi_plot, u_full / u_max, label=f't={t:.1f}')
        else:
             plt.plot(xi_plot, u_full * 0 , label=f't={t:.1f} (u_max~0)') # Plot zeros if max is tiny
    plt.xlabel(r'$\xi = (x-x_0)/x_f$')
    plt.ylabel(r'$u(\xi, t) / u_{max}(t)$')
    plt.title(f'Renormalized Solution - {ic_label}')
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.ylim(-0.1, 1.1) 

    plt.suptitle(f'Boussinesq Equation - {ic_label}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

generate_plots(sol_ic1, "IC 1: $u(x,0)=1-x^2$")
generate_plots(sol_ic2, "IC 2: $u(x,0)=1-0.99\cos(2\pi x)$")

plt.show()
