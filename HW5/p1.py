import numpy as np
import matplotlib.pyplot as plt

def hamiltonian(u, v, x, y):
    return 0.5 * u**2 + 0.5 * v**2 - 1 / np.sqrt(x**2 + y**2)

def equations_of_motion(u, v, x, y):
    dx_dt = u
    dy_dt = v
    du_dt = -x / (x**2 + y**2)**1.5
    dv_dt = -y / (x**2 + y**2)**1.5
    return du_dt, dv_dt, dx_dt, dy_dt

def stoermer_verlet_step(u_n, v_n, x_n, y_n, h):

    du_dt_n, dv_dt_n, _, _ = equations_of_motion(u_n, v_n, x_n, y_n)
    u_half = u_n + 0.5 * h * du_dt_n
    v_half = v_n + 0.5 * h * dv_dt_n

    _, _, dx_dt_half, dy_dt_half = equations_of_motion(u_half, v_half, x_n, y_n)
    x_next = x_n + h * dx_dt_half
    y_next = y_n + h * dy_dt_half

    du_dt_next, dv_dt_next, _, _ = equations_of_motion(u_half, v_half, x_next, y_next)
    u_next = u_half + 0.5 * h * du_dt_next
    v_next = v_half + 0.5 * h * dv_dt_next

    return u_next, v_next, x_next, y_next

h = 0.017  # Time step
t_final = 10 * 9.673596609249161  # 10 revolutions, period from the text [cite: 8]
num_steps = int(t_final / h)

u_0 = 0
v_0 = 0.5
x_0 = 2
y_0 = 0

u = np.zeros(num_steps + 1)
v = np.zeros(num_steps + 1)
x = np.zeros(num_steps + 1)
y = np.zeros(num_steps + 1)
H = np.zeros(num_steps + 1)  # To store Hamiltonian values
t = np.linspace(0, t_final, num_steps + 1)

u[0] = u_0
v[0] = v_0
x[0] = x_0
y[0] = y_0
H[0] = hamiltonian(u_0, v_0, x_0, y_0)

for i in range(num_steps):
    u[i + 1], v[i + 1], x[i + 1], y[i + 1] = stoermer_verlet_step(u[i], v[i], x[i], y[i], h)
    H[i + 1] = hamiltonian(u[i+1], v[i+1], x[i+1], y[i+1])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Orbit in xy-plane')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')  # Ensure circular appearance

plt.subplot(1, 2, 2)
plt.plot(t, H)
plt.title('Hamiltonian over Time')
plt.xlabel('Time')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim(-1,1)

plt.tight_layout() 
plt.show()