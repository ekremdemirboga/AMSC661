import numpy as np
import matplotlib.pyplot as plt
a = np.sqrt(2)
h = 0.05
xmin, xmax = -6, 6
x = np.arange(xmin, xmax + h, h)
phi = np.maximum(1 - np.abs(x), 0)
L = xmax - xmin
times = [1/(2*a), 1/a, 2/a, 4/a]
k = 1/(200*a)
rec_steps = [int(t/k) for t in times]
max_steps = rec_steps[-1]
def lax_f(u, nu):
    return 0.5*(np.roll(u, -1) + np.roll(u, 1)) - 0.5*nu*(np.roll(u, -1) - np.roll(u, 1))
def upw(u, nu):
    if nu >= 0:
        return u - nu*(u - np.roll(u, 1))
    return u - nu*(np.roll(u, -1) - u)
def lw(u, nu):
    return u - 0.5*nu*(np.roll(u, -1) - np.roll(u, 1)) + 0.5*nu**2*(np.roll(u, -1) - 2*u + np.roll(u, 1))
def bw(u, nu):
    if nu >= 0:
        return u - nu*(3*u - 4*np.roll(u, 1) + np.roll(u, 2))/2 + 0.5*nu**2*(u - 2*np.roll(u, 1) + np.roll(u, 2))
    return u - nu*(-3*u + 4*np.roll(u, -1) - np.roll(u, -2))/2 + 0.5*nu**2*(np.roll(u, -2) - 2*np.roll(u, -1) + u)
methods = {"Lax-Friedrichs": lax_f, "Upwind": upw, "Lax-Wendroff": lw, "Beam-Warming": bw}
solutions = {m: [] for m in methods}
for name, scheme in methods.items():
    nu1 = a * k / h
    nu2 = -a * k / h
    v = phi.copy()
    w = phi.copy()
    v_rec, w_rec = [], []
    for n in range(1, max_steps + 1):
        v = scheme(v, nu1)
        w = scheme(w, nu2)
        if n in rec_steps:
            v_rec.append(v.copy())
            w_rec.append(w.copy())
    solutions[name] = [0.5*(vr + wr) for vr, wr in zip(v_rec, w_rec)]
def exact_sol(x, t):
    xp = (x + a*t - xmin) % L + xmin
    xm = (x - a*t - xmin) % L + xmin
    return 0.5*(np.interp(xp, x, phi) + np.interp(xm, x, phi))
exact_sols = [exact_sol(x, t) for t in times]
plt.figure(figsize=(12, 8))
for i, t in enumerate(times):
    plt.subplot(2, 2, i+1)
    for name in solutions:
        plt.plot(x, solutions[name][i], label=name)
    plt.plot(x, exact_sols[i], 'k--', label='Exact')
    plt.title(f"t = {t:.3f}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 8))
for i, t in enumerate(times):
    plt.subplot(2, 2, i+1)
    for name in solutions:
        err = np.sqrt((solutions[name][i] - exact_sols[i])**2)
        plt.semilogy(x, err, label=name)
    plt.title(f"Error t = {t:.3f}")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
