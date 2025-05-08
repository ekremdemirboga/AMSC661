import numpy as np
import matplotlib.pyplot as plt

#--- parameters as before ---
Nx    = 400
xmax  = 6.0
x     = np.linspace(0, xmax, Nx, endpoint=False)
dx    = x[1] - x[0]
CFL   = 0.9
dt    = CFL * dx
tmax  = 6.0
nsteps = int(np.ceil(tmax / dt))
t_rec  = np.arange(0, 7)       # 0,1,...,6
nrec   = len(t_rec)


def initial_condition(x):
    u = np.zeros_like(x)
    u[(x >= 0) & (x <= 1)] = 1.0
    return u
#--- corrected exact solution ---
def exact_burgers(x, t):
    u_exact = np.zeros_like(x)
    if t == 0:
        return initial_condition(x)

    if t < 2.0:
        # Region 1: u = 0 for x < 0
        # Region 2: u = x/t for 0 <= x < t
        # Region 3: u = 1 for t <= x < 1 + t/2
        # Region 4: u = 0 for x > 1 + t/2
        u_exact[(x >= 0) & (x < t)] = x[(x >= 0) & (x < t)] / t
        u_exact[(x >= t) & (x < 1 + t/2.0)] = 1.0
    else: # t >= 2.0
        # Region 1: u = 0 for x < 0
        # Region 2: u = x/t for 0 <= x < sqrt(2t)
        # Region 3: u = 0 for x > sqrt(2t)
        shock_pos = np.sqrt(2.0 * t)
        u_exact[(x >= 0) & (x < shock_pos)] = x[(x >= 0) & (x < shock_pos)] / t
    return u_exact



#--- Godunov flux (unchanged) ---
def godunov_flux(u):
    uG = np.concatenate([[0], u, [0]])
    UL, UR = uG[:-1], uG[1:]
    fL, fR = 0.5*UL**2, 0.5*UR**2
    Fr = np.zeros_like(UL)
    # rarefaction case UL<=UR
    mask = UL <= UR
    #   if UR<=0
    sub = mask & (UR<=0)
    Fr[sub] = fR[sub]
    #   if UL>=0
    sub = mask & (UL>=0)
    Fr[sub] = fL[sub]
    # shock case UL>UR
    maskS = ~mask
    s = (fR - fL)/(UR - UL + 1e-16)
    sub = maskS & (s>=0)
    Fr[sub] = fL[sub]
    sub = maskS & (s<0)
    Fr[sub] = fR[sub]
    return Fr

#--- storage and exact ---
Unum = {meth: np.zeros((nrec, Nx)) for meth in ('LF','RM','MC','G')}
Uex  = np.zeros((nrec, Nx))
for j, tt in enumerate(t_rec):
    Uex[j] = exact_burgers(x, tt)

#--- initial bump ---
u0 = np.where((x>=0)&(x<=1), 1.0, 0.0)

#--- time‐integration loop (same as before) ---
for name in Unum:
    u = u0.copy()
    rec = 0
    Unum[name][rec] = u
    t = 0.0
    for _ in range(nsteps):
        f  = 0.5*u**2
        uG = np.concatenate([[0], u, [0]])
        fG = np.concatenate([[0], f, [0]])
        if name == 'LF':
            u = 0.5*(uG[2:] + uG[:-2]) - dt/(2*dx)*(fG[2:] - fG[:-2])
        elif name == 'RM':
            uh = 0.5*(uG[1:-1] + uG[2:]) - dt/(2*dx)*(fG[2:] - fG[1:-1])
            fh = 0.5*uh**2
            u  = u - dt/dx*(fh - np.concatenate([[0], fh[:-1]]))
        elif name == 'MC':
            up = u - dt/dx*(f - np.concatenate([[0], f[:-1]]))
            fp = 0.5*up**2
            u  = 0.5*(u + up) - dt/(2*dx)*(fp - np.concatenate([[0], fp[:-1]]))
        else:  # Godunov
            F = godunov_flux(u)
            u = u - dt/dx*(F[1:] - F[:-1])
        t += dt
        if rec < nrec-1 and t >= t_rec[rec+1] - 1e-9:
            rec += 1
            Unum[name][rec] = u.copy()

#--- plotting ---
for name, Usol in Unum.items():
    fig, axes = plt.subplots(2,4,figsize=(12,6))
    axes = axes.ravel()
    for j in range(nrec):
        ax = axes[j]
        ax.plot(x, Usol[j], 'b-', label='num')
        ax.plot(x, Uex[j],  'k--', label='exact')
        ax.set_ylim(-0.1,1.1), ax.set_title(f"t={t_rec[j]:.0f}")
        if j==0: ax.legend(loc='upper right')
    axes[-1].axis('off')
    fig.suptitle(f"{name} vs exact", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.95])

plt.show()
