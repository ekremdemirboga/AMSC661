import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def van_der_pol(t, y, mu):
    y1, y2 = y
    dy1dt = y2
    dy2dt = mu * ((1 - y1**2) * y2) - y1
    return [dy1dt, dy2dt]
mus = [10, 100, 1000]
t_max = 1000.0
y0 = [2, 0]
epsilons = [1e-6, 1e-9, 1e-12]
solvers = ['RK45', 'LSODA']

fig_phase, axes_phase = plt.subplots(len(mus), len(solvers) * len(epsilons), figsize=(18, 6 * len(mus)))

fig_cpu, axes_cpu = plt.subplots(1, len(mus), figsize=(18, 6))  # One row for CPU time plots

plt.subplots_adjust(hspace=0.4, wspace=0.3)

results = {}

plot_index = 0
for mu_idx, mu in enumerate(mus):
    results[mu] = {}
    print(f"--- Solving for mu = {mu} ---")

    # CPU Time Plot (one per mu)
    ax_cpu = axes_cpu[mu_idx]

    for solver_idx, solver in enumerate(solvers):
        results[mu][solver] = {}
        print(f"  Using solver: {solver}")
        cpu_times = []

        for epsilon_idx, epsilon in enumerate(epsilons):
            results[mu][solver][epsilon]={}
            print(f"    Error tolerance: {epsilon}")

            start_time = time.time()
            sol = solve_ivp(van_der_pol, [0, t_max], y0, method=solver, args=(mu,), atol=epsilon, rtol=epsilon, dense_output=True)
            end_time = time.time()
            cpu_time = end_time - start_time
            results[mu][solver][epsilon] = cpu_time
            cpu_times.append(cpu_time)
            print(f"    CPU time: {cpu_time:.4f} seconds")

            # --- Phase Plane Plot ---
            ax_phase = axes_phase[mu_idx, solver_idx * len(epsilons) + epsilon_idx]
            t_eval = np.linspace(0, t_max, 5000)
            y_eval = sol.sol(t_eval)
            ax_phase.plot(y_eval[0], y_eval[1],'.')
            ax_phase.set_xlabel("y1")
            ax_phase.set_ylabel("y2")
            ax_phase.set_title(f"mu={mu}, {solver}, ε={epsilon}")
            ax_phase.grid(True)
            

        # Add CPU time data to the plot for the current solver
        ax_cpu.loglog(epsilons, cpu_times, marker='o', label=solver)

    ax_cpu.set_xlabel("log(Error Tolerance) (ε)")
    ax_cpu.set_ylabel("log(CPU Time) (seconds)")
    ax_cpu.set_title(f"CPU Time vs. Error Tolerance (mu = {mu})")
    ax_cpu.legend()
    ax_cpu.grid(True)
fig_phase.suptitle("Van der Pol Oscillator - Phase Plane Plots", fontsize=16)
fig_cpu.suptitle("CPU Time Comparison", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()
for mu in mus:
    print(f"--- Results for mu = {mu} ---")
    for solver in solvers:
            print(f" Solver: {solver}")
            for epsilon in epsilons:
                cpu_time = results[mu][solver][epsilon]
                print(f"    epsilon: {epsilon},  CPU Time: {cpu_time:.6f} seconds")