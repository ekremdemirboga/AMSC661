import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Arenstorf Orbit ODE
def arenstorf(t, y, mu):
    x, y, vx, vy = y
    mu_prime = 1 - mu
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - mu_prime)**2 + y**2)

    dvxdt = x + 2 * vy - mu_prime * (x + mu) / r1**3 - mu * (x - mu_prime) / r2**3
    dvydt = y - 2 * vx - mu_prime * y / r1**3 - mu * y / r2**3

    return [vx, vy, dvxdt, dvydt]

mu = 0.012277471
y0 = [0.994, 0, 0, -2.00158510637908252240537862224]  
T_period = 17.0652165601579625588917206249 
t_max_short = T_period  
t_max_long = 100.0      
epsilon = 1e-12        
print("--- Solving for one period (DOPRI5) ---")
sol_period = solve_ivp(
    arenstorf,
    [0, t_max_short],
    y0,
    method='RK45',  
    args=(mu,),
    atol=epsilon,
    rtol=epsilon,
    dense_output=True
)
t_eval_period = np.linspace(0, t_max_short, 5000)
y_eval_period = sol_period.sol(t_eval_period)
plt.figure(figsize=(8, 8))
plt.plot(y_eval_period[0], y_eval_period[1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Arenstorf Orbit (One Period) - DOPRI5")
plt.axis('equal')  
plt.grid(True)
plt.show()
solvers = ['RK45', 'DOP853', 'Radau']
results = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 
for i, solver in enumerate(solvers):
    print(f"--- Solving with {solver} (Tmax = 100) ---")
    start_time = time.time()
    sol_long = solve_ivp(
        arenstorf,
        [0, t_max_long],
        y0,
        method=solver,
        args=(mu,),
        atol=epsilon,
        rtol=epsilon,
        dense_output=True
    )
    end_time = time.time()
    cpu_time = end_time - start_time
    results[solver] = cpu_time
    print(f"  CPU time: {cpu_time:.4f} seconds")
    t_eval = np.linspace(0, t_max_long, 5000)
    y_eval = sol_long.sol(t_eval)
    ax = axes[i]  
    ax.plot(y_eval[0], y_eval[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Arenstorf Orbit - {solver}")
    ax.axis('equal')
    ax.grid(True)
plt.tight_layout()
plt.show()
# Print CPU Time Comparison
print("\n--- CPU Time Comparison ---")
for solver, cpu_time in results.items():
    print(f"{solver}: {cpu_time:.4f} seconds")