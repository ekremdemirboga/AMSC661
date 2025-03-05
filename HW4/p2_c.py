import numpy as np
import matplotlib.pyplot as plt

def compute_ras_boundary(alpha_0):
    theta = np.linspace(0, 2 * np.pi, 300)
    z = np.exp(1j * theta)  
    
    char_eq = lambda z: z**2 + alpha_0 * z + (1 + alpha_0)
    
    boundary = -char_eq(z)  
    return boundary

group1 = np.arange(-1.8, -1.0, 0.1)
group2 = np.arange(-1.0, 0.0, 0.1)

plt.figure(figsize=(8, 6))
for alpha in group1:
    boundary = compute_ras_boundary(alpha)
    plt.plot(boundary.real, boundary.imag, label=f"alpha = {alpha:.1f}")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("RAS Boundaries for alpha in [-1.8, -1.1]")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
for alpha in group2:
    boundary = compute_ras_boundary(alpha)
    plt.plot(boundary.real, boundary.imag, label=f"alpha = {alpha:.1f}")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("RAS Boundaries for alpha in [-1.0, -0.1]")
plt.legend()
plt.grid()
plt.show()
