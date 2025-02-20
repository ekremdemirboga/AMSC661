import numpy as np
import matplotlib.pyplot as plt

def stability_euler(z):
    return 1 + z
def stability_midpoint_euler(z):
    return 1 + z + 0.5 * z**2
def stability_kutta3(z):
    return 1 + z + 0.5 * z**2 + (1/6) * z**3
def stability_rk4(z):
    return 1 + z + 0.5 * z**2 + (1/6) * z**3 + (1/24) * z**4
def stability_dopri5(z):
  return 1 + z + (1/2)*z**2 + (1/6)*z**3 + (1/24)*z**4 + (1/120)*z**5 + (1/600)*z**6 + (1/3600)*z**7

x_min, x_max = -4, 4
y_min, y_max = -4, 4
num_points = 500
x = np.linspace(x_min, x_max, num_points)
y = np.linspace(y_min, y_max, num_points)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
methods = {
    "Forward Euler": stability_euler,
    "Midpoint (Euler Predictor)": stability_midpoint_euler,
    "Kutta's 3rd Order": stability_kutta3,
    "RK4": stability_rk4,
    "DOPRI5": stability_dopri5
}
num_methods = len(methods)
fig, axes = plt.subplots(1, num_methods, figsize=(4 * num_methods, 4)) #adjust figure size
if num_methods == 1:
    axes = [axes] 
for i, (name, stability_func) in enumerate(methods.items()):
    ax = axes[i]  
    R = stability_func(Z)
    region = np.abs(R) <= 1
    ax.contourf(X, Y, region, levels=[0, 0.5, 1], colors=['white', 'gray'], alpha=0.3)
    ax.contour(X, Y, region, levels=[0.5], colors='black', linewidths=1)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(name)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
plt.tight_layout()  
plt.suptitle("Regions of Absolute Stability for ERK Methods", fontsize=16)
plt.subplots_adjust(top=0.85)   
plt.show()