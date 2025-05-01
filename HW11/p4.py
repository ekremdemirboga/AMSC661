import numpy as np
import matplotlib.pyplot as plt

# Spatial domain
x_min, x_max = -1.0, 6.0
Nx = 400
x = np.linspace(x_min, x_max, Nx)
dx = x[1] - x[0]

# Initial condition
def initial_condition(x):
    u = np.zeros_like(x)
    u[(x > 0) & (x < 1)] = 2.0
    u[(x > 1) & (x < 2)] = 1.0
    return u

# Flux function
def flux(u):
    return 0.5 * u**2

# Lax-Friedrichs method
def lax_friedrichs(u, dt, dx, nt):
    for _ in range(nt):
        u_p = np.roll(u, -1)
        u_m = np.roll(u, 1)
        u = 0.5*(u_p + u_m) - (dt/(2*dx))*(flux(u_p) - flux(u_m))
        # enforce zero Dirichlet BC
        u[0] = 0
        u[-1] = 0
    return u

# Richtmyer two-step Lax-Wendroff
def richtmyer(u, dt, dx, nt):
    for _ in range(nt):
        u_p = np.roll(u, -1)
        # predictor at midpoints
        u_half = 0.5*(u + u_p) - (dt/(2*dx))*(flux(u_p) - flux(u))
        # compute flux at midpoints
        f_half_p = flux(u_half)
        f_half_m = np.roll(f_half_p, 1)
        u = u - (dt/dx)*(f_half_p - f_half_m)
        u[0] = 0
        u[-1] = 0
    return u

# MacCormack method
def mac_cormack(u, dt, dx, nt):
    for _ in range(nt):
        # predictor
        u_p = np.roll(u, -1)
        u_star = u - (dt/dx)*(flux(u_p) - flux(u))
        # corrector
        f_star_p = flux(u_star)
        f_star_m = np.roll(f_star_p, 1)
        u = 0.5*(u + u_star) - (dt/(2*dx))*(f_star_p - f_star_m)
        u[0] = 0
        u[-1] = 0
    return u

# Helper to run a scheme up to t_final with CFL control
def run_scheme(scheme, u0, dx, t_final, cfl=0.4):
    dt_max = cfl * dx / 2.0
    nt = int(np.ceil(t_final / dt_max))
    dt = t_final / nt
    return scheme(u0.copy(), dt, dx, nt)

# Exact solution piecewise
def u_exact(x, t):
    u = np.zeros_like(x)
    if t <= 1.0:
        xs1 = 1 + 1.5*t
        xs2 = 2 + 0.5*t
        for i, xi in enumerate(x):
            if xi < 0:
                u[i] = 0
            elif xi <= 2*t:
                u[i] = xi / t
            elif xi <= xs1:
                u[i] = 2
            elif xi <= xs2:
                u[i] = 1
            else:
                u[i] = 0
    elif t <= 2.0:
        xs = 1.5 + t
        for i, xi in enumerate(x):
            if xi < 0:
                u[i] = 0
            elif xi <= 2*t:
                u[i] = xi / t
            elif xi <= xs:
                u[i] = 2
            else:
                u[i] = 0
    else:
        C = 3.5 / np.sqrt(2)
        xs = C * np.sqrt(t)
        for i, xi in enumerate(x):
            if xi < 0:
                u[i] = 0
            elif xi <= xs:
                u[i] = xi / t
            else:
                u[i] = 0
    return u

# Times to plot
times = [0.5, 1.5, 2.5, 3.5, 5.0]
u0 = initial_condition(x)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Plot settings
def plot_subplots_all_times(scheme_name, all_solutions):
    fig, axs = plt.subplots(1, len(times), figsize=(20, 4), sharey=True)
    fig.suptitle(f'{scheme_name} Method', fontsize=16)
    
    for i, t in enumerate(times):
        axs[i].plot(x, all_solutions[i], label=f't = {t}')
        axs[i].set_title(f't = {t}')
        axs[i].set_xlabel('x')
        axs[i].grid()
    axs[0].set_ylabel('u')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Store solutions for each method and exact
lf_sols, rm_sols, mc_sols, ex_sols = [], [], [], []

for t in times:
    lf_sols.append(run_scheme(lax_friedrichs, u0, dx, t))
    rm_sols.append(run_scheme(richtmyer, u0, dx, t))
    mc_sols.append(run_scheme(mac_cormack, u0, dx, t))
    ex_sols.append(u_exact(x, t))

# Plot each method separately in subplots
plot_subplots_all_times("Lax-Friedrichs", lf_sols)
plot_subplots_all_times("Richtmyer", rm_sols)
plot_subplots_all_times("MacCormack", mc_sols)
plot_subplots_all_times("Exact Solution", ex_sols)
