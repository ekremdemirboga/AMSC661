from distmesh import *
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.sparse
import scipy.linalg

r_inner = 1.0
r_outer = 2.0

def fd_annulus(p):
    """Signed distance function for annulus 1 < r < 2."""
    d_inner = dcircle(p, 0, 0, r_inner)
    d_outer = dcircle(p, 0, 0, r_outer)
    return ddiff(d_inner, -d_outer)

fh = huniform

bbox = [-r_outer, r_outer, -r_outer, r_outer]
h0 = 0.1

pfix = None

print("Generating mesh for annulus...")
pts, tri = distmesh2D(fd_annulus, fh, h0, bbox, pfix)
print(f"Mesh generated with {pts.shape[0]} points and {tri.shape[0]} triangles.")

fig_mesh, ax_mesh = plt.subplots(figsize=(6,6))
ax_mesh.triplot(pts[:,0], pts[:,1], tri, linewidth=0.5)
ax_mesh.set_title('Annulus Mesh')
ax_mesh.set_aspect('equal')
ax_mesh.set_xlabel('x')
ax_mesh.set_ylabel('y')
plt.show(block=False)

def triarea2(verts):
    """Calculates twice the signed area of a triangle."""
    Aux = np.ones((3,3))
    Aux[1:3,:] = np.transpose(verts)
    return np.linalg.det(Aux)

def stima3(verts):
    """Computes element stiffness (MA) and mass (MB) matrices for a triangle."""
    Aux = np.ones((3,3))
    Aux[1:3,:] = np.transpose(verts)
    rhs = np.zeros((3,2))
    rhs[1,0] = 1
    rhs[2,1] = 1
    try:
        G = np.linalg.solve(Aux,rhs)
    except np.linalg.LinAlgError:
        print(f"Warning: Singular matrix in stima3 for vertices {verts}. Using pseudo-inverse.")
        G = np.linalg.pinv(Aux) @ rhs

    det = np.linalg.det(Aux)
    MA = 0.5*det*np.matmul(G,np.transpose(G))
    MB = det*np.array([[2,1,1],[1,2,1],[1,1,2]])/24
    return MA, MB

Npts = pts.shape[0]
Ntri = tri.shape[0]
print(f"Npts = {Npts}, Ntri = {Ntri}")

radius = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
tol = 1e-3
boundary_nodes = np.where((np.abs(radius - r_inner) < tol) | (np.abs(radius - r_outer) < tol))[0]
free_nodes = np.setdiff1d(np.arange(Npts), boundary_nodes)


A = scipy.sparse.lil_matrix((Npts, Npts), dtype=float)
B = scipy.sparse.lil_matrix((Npts, Npts), dtype=float)
F_glob = np.zeros((Npts, 1))

for j in range(Ntri):
    v = pts[tri[j,:],:]
    ind = tri[j,:]
    MA, MB = stima3(v)
    area = triarea2(v) / 2.0

    for row in range(3):
        for col in range(3):
            A[ind[row], ind[col]] += MA[row, col]
            B[ind[row], ind[col]] += MB[row, col]
        F_glob[ind[row]] += area / 3.0

A = A.tocsr()
B = B.tocsr()
print("Assembly complete.")

U = np.zeros((Npts, 1))
r_nodes = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
r_nodes[r_nodes < 1e-10] = 1e-10
cos_phi_nodes = pts[:, 0] / r_nodes
U[:, 0] = r_nodes + cos_phi_nodes

U[boundary_nodes, 0] = 0.0

fig_ic, ax_ic = plt.subplots(subplot_kw={"projection": "3d"})
ax_ic.plot_trisurf(pts[:, 0], pts[:, 1], U[:, 0], triangles=tri, cmap='viridis')
ax_ic.set_title('Initial Condition u(x, y, 0)')
plt.show(block=False)

Tmax = 1.0
dt = 0.01
Nt = int(round(Tmax / dt))
tvals = np.linspace(0, Tmax, Nt + 1)

print(f"Starting time stepping with dt={dt}, Nt={Nt} steps...")

LHS_matrix = B + (dt / 2.0) * A
RHS_matrix = B - (dt / 2.0) * A
RHS_vector = dt * F_glob

free_nodes_idx = np.ix_(free_nodes, free_nodes)
LHS_ff = LHS_matrix[free_nodes_idx].tocsc()

results = {}
results[0.0] = U.copy()
times_to_plot = [0.1, 1.0]

U_current = U.copy()

for n in range(Nt):
    t_current = tvals[n+1]

    rhs_full = RHS_matrix @ U_current + RHS_vector

    rhs_f = rhs_full[free_nodes]

    try:
        U_f_new = scipy.sparse.linalg.spsolve(LHS_ff, rhs_f)
    except Exception as e:
        print(f"Warning: Sparse solve failed at step {n+1} (t={t_current:.2f}): {e}. Trying dense solver.")
        try:
            U_f_new = np.linalg.solve(LHS_ff.toarray(), rhs_f)
        except np.linalg.LinAlgError:
            print("Error: Dense solver also failed. Matrix might be singular.")
            U_f_new = np.linalg.lstsq(LHS_ff.toarray(), rhs_f, rcond=None)[0]

    U_current[free_nodes, 0] = U_f_new
    U_current[boundary_nodes, 0] = 0.0

    for t_plot in times_to_plot:
        if np.isclose(t_current, t_plot, atol=dt/2.0):
            results[t_plot] = U_current.copy()
            print(f"Stored result at t = {t_current:.2f}")

    if (n + 1) % 10 == 0:
         print(f"Completed step {n+1}/{Nt} (t = {t_current:.2f})")

print("Time stepping finished.")

triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

fig = plt.figure(figsize=(12, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
if 0.1 in results:
    u_01 = results[0.1][:, 0]
    ax1.plot_trisurf(triang, u_01, cmap='viridis', edgecolor='none')
    ax1.set_title('Solution u(x, y) at t=0.1')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
else:
    ax1.set_title('Solution at t=0.1 not available')

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
if 1.0 in results:
    u_10 = results[1.0][:, 0]
    ax2.plot_trisurf(triang, u_10, cmap='viridis', edgecolor='none')
    ax2.set_title('Solution u(x, y) at t=1.0')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
else:
     ax2.set_title('Solution at t=1.0 not available')

ax3 = fig.add_subplot(2, 2, 3)
if 1.0 in results:
    u_10 = results[1.0][:, 0]
    r_nodes = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    sort_indices = np.argsort(r_nodes)
    r_sorted = r_nodes[sort_indices]
    u_sorted = u_10[sort_indices]
    ax3.plot(r_sorted, u_sorted, 'b.', markersize=3, label='FEM Solution at t=1.0 (Nodes)')

    r_exact = np.linspace(r_inner, r_outer, 200)
    log_r = np.log(r_exact)
    u_exact_stationary = (1 - r_exact**2) / 4.0 + (3 * log_r) / (4 * np.log(2.0))
    ax3.plot(r_exact, u_exact_stationary, 'r-', linewidth=2, label='Exact Stationary Solution')

    ax3.set_title('Solution vs Radius at t=1.0')
    ax3.set_xlabel('Radius r')
    ax3.set_ylabel('u(r)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlim(r_inner - 0.05, r_outer + 0.05)
else:
    ax3.set_title('Solution at t=1.0 not available for radial plot')

fig.suptitle('Problem 3')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
