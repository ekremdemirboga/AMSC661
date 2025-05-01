import numpy as np
import matplotlib.pyplot as plt

t_max = 3.0
t = np.linspace(0, t_max, 100)

plt.figure(figsize=(8, 6))

num_chars_neg = 5
for x0 in np.linspace(-1, -0.1, num_chars_neg):
     plt.plot(x0 * np.ones_like(t), t, 'g--', linewidth=0.7, label='Char. (u=0)' if x0==np.linspace(-1, -0.1, num_chars_neg)[0] else "") # Label only once

num_chars_01 = 10
for x0 in np.linspace(0.01, 0.99, num_chars_01):
     plt.plot(x0 + 2 * t, t, 'b--', linewidth=0.7, label='Char. (u=2)' if x0==np.linspace(0.01, 0.99, num_chars_01)[0] else "") # Label only once

num_chars_12 = 10
for x0 in np.linspace(1.01, 1.99, num_chars_12):
     plt.plot(x0 + 1 * t, t, 'c--', linewidth=0.7, label='Char. (u=1)' if x0==np.linspace(1.01, 1.99, num_chars_12)[0] else "") # Label only once

num_chars_pos = 5
for x0 in np.linspace(2.1, 3.0, num_chars_pos):
     # Reuse label for u=0 characteristics
     plt.plot(x0 * np.ones_like(t), t, 'g--', linewidth=0.7)

plt.plot(0 * t, t, 'k:', label='Rarefaction Boundary (x=0)')
plt.plot(2 * t, t, 'k:', label='Rarefaction Boundary (x=2t)')

t_shock1 = np.linspace(0, 1, 50)
x_shock1 = 1 + 1.5 * t_shock1
plt.plot(x_shock1, t_shock1, 'r-', linewidth=2, label='Shock (u=2|u=1)')

t_shock2 = np.linspace(0, 1, 50)
x_shock2 = 2 + 0.5 * t_shock2
plt.plot(x_shock2, t_shock2, 'm-', linewidth=2, label='Shock (u=1|u=0)')

t_shock3 = np.linspace(1, t_max, 50)
x_shock3 = 1.5 + t_shock3
plt.plot(x_shock3, t_shock3, 'k-', linewidth=2, label='Merged Shock (u=2|u=0)')

plt.xlabel('x (Position)')
plt.ylabel('t (Time)')
plt.xlim(-1, 5)
plt.ylim(0, t_max)
plt.grid(True)
plt.legend()
plt.show()