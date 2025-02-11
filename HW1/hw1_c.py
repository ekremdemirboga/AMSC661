import numpy as np
import matplotlib.pyplot as plt

def gravity_system(t, state):
    x, y, u, v = state
    r = np.sqrt(x**2 + y**2)
    dxdt = u
    dydt = v
    dudt = -x / r**3
    dvdt = -y / r**3
    return np.array([dxdt, dydt, dudt, dvdt])

def linear_two_step_method(f, t, h, state_prev, state_curr):
    a0, a1, b0, b1 = -4, 5, 4, 2
    f_curr = f(t, state_curr)
    f_prev = f(t - h, state_prev)
    state_next = a0 * state_curr + a1 * state_prev + h * (b0 * f_curr + b1 * f_prev)
    return state_next

state0 = np.array([1.0, 0.0, 0.0, 1.0])  # [x, y, u, v]
T = 4 * np.pi  
N_values = [20, 40, 80] 

for N in N_values:
    h = 2 * np.pi / N  
    num_steps = int(T / h)  
    t_values = np.linspace(0, T, num_steps + 1)
    state_values = np.zeros((num_steps + 1, 4))
    state_values[0] = state0
    state_values[1] = state0 + h * gravity_system(0, state0)
    for i in range(1, num_steps):
        state_values[i + 1] = linear_two_step_method(gravity_system, t_values[i], h, state_values[i - 1], state_values[i])
    
    final_norm = np.linalg.norm(state_values[-1])
    print(f"For N = {N}, h = {h:.4f}, norm at t = 4π: {final_norm:.4f}")
    plt.plot(state_values[:, 0], state_values[:, 1], label=f"N = {N}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gravity Problem: xy-plane Trajectory")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()