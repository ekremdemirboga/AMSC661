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

def midpoint_rule(f, t, h, state):
    state_pred = state + h * f(t, state)
    state_next = state + h * f(t + h/2, (state + state_pred) / 2)
    return state_next

state0 = np.array([1.0, 0.0, 0.0, 1.0])  # [x, y, u, v]

T = 8 * np.pi  
N_values = [20, 40, 80] 
for N in N_values:
    h = 2 * np.pi / N  
    num_steps = int(T / h) 
    t_values = np.linspace(0, T, num_steps + 1)
    state_values = np.zeros((num_steps + 1, 4))
    state_values[0] = state0
    for i in range(num_steps):
        state_values[i + 1] = midpoint_rule(gravity_system, t_values[i], h, state_values[i])
    
    plt.plot(state_values[:, 0], state_values[:, 1], label=f"N = {N}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gravity Problem: Midpoint Rule with Forward Euler Predictor")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()