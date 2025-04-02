import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay
from scipy import sparse
import scipy.sparse.linalg as spla



import distmesh




def assemble_stiffness(pts, tri, conductivity_func, tol=1e-12):
    """Assembles the FEM stiffness matrix K for -div(a*grad(u)) = 0."""
    Npts = pts.shape[0]
    Ntri = tri.shape[0]
    K = sparse.lil_matrix((Npts, Npts))
    skipped_triangles = 0

    for i in range(Ntri):
        nodes = tri[i, :]
        if np.any(nodes >= Npts) or np.any(nodes < 0):
             skipped_triangles += 1
             continue

        p = pts[nodes, :] # p[0,:]=(x1,y1), p[1,:]=(x2,y2), p[2,:]=(x3,y3)

        # Calculate signed area * 2
        # (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        twice_area_signed = (p[1,0]-p[0,0])*(p[2,1]-p[0,1]) - (p[2,0]-p[0,0])*(p[1,1]-p[0,1])

        if abs(twice_area_signed) < tol:
             skipped_triangles += 1
             continue # Skip degenerate triangle

        Area = 0.5 * abs(twice_area_signed)

        # --- Calculate Gradients Directly ---
        # grad(eta1) = [y2-y3, x3-x2] / (2*Area_signed)
        # grad(eta2) = [y3-y1, x1-x3] / (2*Area_signed)
        # grad(eta3) = [y1-y2, x2-x1] / (2*Area_signed)
        b = np.zeros(3)
        c = np.zeros(3)
        b[0] = (p[1,1] - p[2,1]) / twice_area_signed
        c[0] = (p[2,0] - p[1,0]) / twice_area_signed
        b[1] = (p[2,1] - p[0,1]) / twice_area_signed
        c[1] = (p[0,0] - p[2,0]) / twice_area_signed
        b[2] = (p[0,1] - p[1,1]) / twice_area_signed
        c[2] = (p[1,0] - p[0,0]) / twice_area_signed

        # G_correct has gradients as rows (shape 3x2)
        G_correct = np.vstack((b, c)).T # Correctly constructs [[b1,c1],[b2,c2],[b3,c3]]

        # Matrix of dot products (shape 3x3)
        D = G_correct @ G_correct.T
        # --- -------------------------- ---

        centroid = np.mean(p, axis=0)
        a_T = conductivity_func(centroid[0], centroid[1])

        # Local stiffness matrix M_ij = a * |T| * (grad(eta_i) . grad(eta_j))
        M_local = a_T * Area * D # D is the matrix of dot products

        # Add local matrix contributions to global matrix K
        for row in range(3):
            for col in range(3):
                # Check indices again just before assignment
                if nodes[row] < Npts and nodes[col] < Npts:
                    K[nodes[row], nodes[col]] += M_local[row, col]
                # else: # Should be caught earlier
                #     print(f"Internal Warning: Invalid node index during assembly {nodes}")


    return K.tocsr()

# --- apply_boundary_conditions, calculate_gradient_and_current, average_to_vertices ---
# --- (These functions remain the same as the previous 'no try-except' version) ---
def apply_boundary_conditions(K, pts, tol=1e-6):
    """Applies Dirichlet BCs u=0 at x=0, u=1 at x=3."""
    Npts = pts.shape[0]
    u = np.zeros(Npts)

    dirichlet0_nodes = np.where(np.abs(pts[:, 0] - 0.0) < tol)[0]
    dirichlet1_nodes = np.where(np.abs(pts[:, 0] - 3.0) < tol)[0]
    dirichlet_nodes = np.unique(np.concatenate((dirichlet0_nodes, dirichlet1_nodes)))
    dirichlet_nodes = dirichlet_nodes[dirichlet_nodes < Npts]

    if len(dirichlet_nodes) == 0: print("Warning: No Dirichlet boundary nodes found.")

    dirichlet0_nodes_final = np.intersect1d(dirichlet_nodes, np.where(np.abs(pts[:, 0] - 0.0) < tol)[0])
    dirichlet1_nodes_final = np.intersect1d(dirichlet_nodes, np.where(np.abs(pts[:, 0] - 3.0) < tol)[0])
    u[dirichlet0_nodes_final] = 0.0
    u[dirichlet1_nodes_final] = 1.0

    all_nodes = np.arange(Npts)
    free_nodes = np.setdiff1d(all_nodes, dirichlet_nodes)
    free_nodes = free_nodes[free_nodes < Npts]

    if K.shape[0] != Npts or K.shape[1] != Npts:
         raise ValueError(f"Stiffness matrix shape {K.shape} inconsistent with Npts {Npts}")

    if len(free_nodes) == 0:
         return u, free_nodes, dirichlet_nodes, None, None

    K_fr_di = K[free_nodes][:, dirichlet_nodes]
    F_bc = -K_fr_di @ u[dirichlet_nodes]
    K_free = K[free_nodes][:, free_nodes]

    return u, free_nodes, dirichlet_nodes, K_free, F_bc


def calculate_gradient_and_current(pts, tri, u, conductivity_func, tol=1e-12):
    Npts = pts.shape[0]
    Ntri = tri.shape[0]
    grad_u = np.zeros((Ntri, 2))
    current_j = np.zeros((Ntri, 2))
    abs_current_centers = np.zeros(Ntri)
    skipped_triangles = 0

    for i in range(Ntri):
        nodes = tri[i, :]
        if np.any(nodes >= Npts) or np.any(nodes < 0):
             skipped_triangles += 1
             continue

        p = pts[nodes, :]
        u_local = u[nodes]

        twice_area_signed = (p[1,0]-p[0,0])*(p[2,1]-p[0,1]) - (p[2,0]-p[0,0])*(p[1,1]-p[0,1])

        if abs(twice_area_signed) < tol:
             grad_u[i,:] = 0.0
             skipped_triangles += 1
        else:
             grad_u_x = (u_local[0]*(p[1,1]-p[2,1]) + u_local[1]*(p[2,1]-p[0,1]) + u_local[2]*(p[0,1]-p[1,1])) / twice_area_signed
             grad_u_y = (u_local[0]*(p[2,0]-p[1,0]) + u_local[1]*(p[0,0]-p[2,0]) + u_local[2]*(p[1,0]-p[0,0])) / twice_area_signed
             grad_u[i, :] = [grad_u_x, grad_u_y]

        centroid = np.mean(p, axis=0)
        a_T = conductivity_func(centroid[0], centroid[1])
        current_j[i, :] = -a_T * grad_u[i, :] # grad_u might be zero if skipped
        abs_current_centers[i] = np.linalg.norm(current_j[i, :])

    return grad_u, current_j, abs_current_centers


def average_to_vertices(tri, Npts, abs_current_centers):
    abs_current_verts = np.zeros(Npts)
    count_tri = np.zeros(Npts, dtype=int)

    Ntri = tri.shape[0]
    skipped_triangles = 0
    for j in range(Ntri):
        nodes = tri[j,:]
        if np.all(nodes < Npts) and np.all(nodes >= 0):
            abs_current_verts[nodes] += abs_current_centers[j]
            count_tri[nodes] += 1
        else:
            skipped_triangles += 1

    valid_nodes = count_tri > 0
    abs_current_verts[~valid_nodes] = 0.0
    abs_current_verts[valid_nodes] = abs_current_verts[valid_nodes] / count_tri[valid_nodes]

    return abs_current_verts

# Main
def solve_problem3(a1, a2, h0=0.1, plot_mesh=False):
    print(f"\n--- Solving for a1 = {a1}, a2 = {a2} ---")

    center_x, center_y, radius = 1.5, 1.5, 1.0
    bbox = [0.0, 3.0, 0.0, 3.0]

    def fd_square(p):
        return distmesh.drectangle(p, bbox[0], bbox[1], bbox[2], bbox[3])
    def fh(p):
        return distmesh.huniform(p)

    pfix_corners = np.array([[0,0], [3,0], [0,3], [3,3]])
    num_circle_pts = 60
    theta = np.linspace(0, 2*np.pi, num_circle_pts, endpoint=False)
    pfix_circle = np.vstack((center_x + radius*np.cos(theta),
                             center_y + radius*np.sin(theta))).T
    pfix = np.vstack((pfix_corners, pfix_circle))

    print("Generating mesh...")
    pts, tri = distmesh.distmesh2D(fd_square, fh, h0, bbox, pfix)

    if pts.shape[0] < 3 or tri.shape[0] < 1:
         print("Error: Mesh generation failed or produced insufficient elements.")
         return None, None, None, None

    if plot_mesh:
        plt.figure(figsize=(6, 6)); plt.triplot(pts[:, 0], pts[:, 1], tri, linewidth=0.5)
        plt.plot(pfix[:,0], pfix[:,1], 'ro', markersize=3, label='Fixed Points')
        plt.title(f'Mesh (a1={a1}, a2={a2})'); plt.xlabel('x'); plt.ylabel('y')
        plt.axis('equal'); plt.legend(); plt.show()

    def conductivity(x, y):
        is_inside = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        return a1 if is_inside else a2

    print("Assembling stiffness matrix...")
    K = assemble_stiffness(pts, tri, conductivity)

    print("Applying boundary conditions...")
    bc_tol = h0 * 0.01
    u, free_nodes, dirichlet_nodes, K_free, F_bc = apply_boundary_conditions(K, pts, tol=bc_tol)

    if K_free is None or K_free.shape[0] == 0:
         print("Error during boundary condition setup or no free nodes.")
         return pts, tri, u, None

    print(f"Solving linear system for {len(free_nodes)} unknowns...")
    # No try-except for solve
    u[free_nodes] = spla.spsolve(K_free, F_bc)

    grad_u, current_j, abs_current_centers = calculate_gradient_and_current(pts, tri, u, conductivity)

    abs_current_verts = average_to_vertices(tri, pts.shape[0], abs_current_centers)

    print("Plotting results...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    tpc_u = plt.tripcolor(pts[:, 0], pts[:, 1], tri, u, cmap='viridis', shading='gouraud')
    plt.colorbar(tpc_u, label='Voltage (u)')
    plt.plot(pfix_circle[:,0], pfix_circle[:,1], 'k--', linewidth=0.8)
    plt.title(f'Voltage Distribution (a1={a1}, a2={a2})')
    plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal'); plt.axis(bbox)
    plt.subplot(1, 2, 2)
    abs_current_verts_plot = np.nan_to_num(abs_current_verts, nan=0.0)
    tpc_j = plt.tripcolor(pts[:, 0], pts[:, 1], tri, abs_current_verts_plot, cmap='inferno', shading='gouraud')
    plt.colorbar(tpc_j, label='Absolute Current Density |j|')
    plt.plot(pfix_circle[:,0], pfix_circle[:,1], 'w--', linewidth=0.8)
    plt.title(f'Absolute Current Density |j| (a1={a1}, a2={a2})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.axis(bbox)
    plt.tight_layout()
    plt.show()

    return pts, tri, u, abs_current_verts

# --- Run the two cases ---
H0_MESH_SIZE = 0.1

results_a = solve_problem3(a1=1.2, a2=1.0, h0=H0_MESH_SIZE, plot_mesh=False)
results_b = solve_problem3(a1=0.8, a2=1.0, h0=H0_MESH_SIZE, plot_mesh=False)
