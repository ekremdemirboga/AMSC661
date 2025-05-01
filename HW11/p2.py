import numpy as np
import matplotlib.pyplot as plt

# Problem 2(b) parameters and calculated values
rho_L = 0.1
rho_R = 0.9

# Characteristic speeds
# c(rho) = -(1 + log(rho))
c_L = -(1 + np.log(rho_L))
c_R = -(1 + np.log(rho_R))

# Flux function f(rho) = -rho * log(rho)
def flux(rho):
 # Handle rho=0 case if necessary, though not needed for 0.1 and 0.9
 if rho <= 0:
     return 0
 return -rho * np.log(rho)

s = (flux(rho_L) - flux(rho_R)) / (rho_L - rho_R)

print(f"Characteristic speed for rho_L = {rho_L}: c_L = {c_L:.3f}")
print(f"Characteristic speed for rho_R = {rho_R}: c_R = {c_R:.3f}")
print(f"Shock speed: s = {s:.3f}")

t_max = 3.0
t = np.linspace(0, t_max, 100)

plt.figure(figsize=(8, 6))


num_chars_left = 10
for x0 in np.linspace(-2, -0.1, num_chars_left):
     plt.plot(x0 + c_L * t, t, 'b--', linewidth=0.7, label='Char. (rho=0.1)' if x0==np.linspace(-2, -0.1, num_chars_left)[0] else "")

num_chars_right = 10
for x0 in np.linspace(0.1, 2, num_chars_right):
     plt.plot(x0 + c_R * t, t, 'c--', linewidth=0.7, label='Char. (rho=0.9)' if x0==np.linspace(0.1, 2, num_chars_right)[0] else "")

x_shock = s * t
plt.plot(x_shock, t, 'r-', linewidth=2, label=f'Shock (s={s:.3f})')

plt.xlabel('x (Position)')
plt.ylabel('t (Time)')
plt.title('Problem 2(b): Characteristics and Shock for Greenberg Traffic Model')
plt.xlim(-2, 3)
plt.ylim(0, t_max)
plt.grid(True)
plt.legend()
plt.show()