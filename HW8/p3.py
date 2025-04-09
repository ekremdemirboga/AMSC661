import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

torch.set_default_dtype(torch.float64)
plt.rcParams.update({'font.size': 14})


xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

def exact_solution(x, y):
    return y.pow(2) * torch.sin(torch.pi * x)

def rhs_f(x, y):
    return (2.0 - (torch.pi**2) * y.pow(2)) * torch.sin(torch.pi * x)

def prepare_data(xmin, xmax, nx, ymin, ymax, ny, requires_grad=False, strip_bdry=False):
    """Generates meshgrid and corresponding data matrix."""
    x_lin = torch.linspace(xmin, xmax, steps=nx, requires_grad=requires_grad)
    y_lin = torch.linspace(ymin, ymax, steps=ny, requires_grad=requires_grad)

    if strip_bdry:
        if nx > 2:
             x_lin = x_lin[1:-1]
        else:
             print("Warning: nx must be > 2 to strip boundary.")
        if ny > 2:
             y_lin = y_lin[1:-1]
        else:
             print("Warning: ny must be > 2 to strip boundary.")

    grid_x, grid_y = torch.meshgrid(x_lin, y_lin, indexing='ij')
    data_matrix = torch.stack((torch.flatten(grid_x), torch.flatten(grid_y)), dim=1)
    return grid_x, grid_y, data_matrix

n_pts_train_axis = 9 # Start with 9x9 grid
train_grid_x, train_grid_y, train_data = prepare_data(
    xmin, xmax, n_pts_train_axis, ymin, ymax, n_pts_train_axis,
    requires_grad=True, strip_bdry=True
)
print(f"Number of training points: {train_data.shape[0]}") 

# Test points
n_pts_test_axis = 100
test_grid_x, test_grid_y, test_data = prepare_data(
    xmin, xmax, n_pts_test_axis, ymin, ymax, n_pts_test_axis,
    requires_grad=False, strip_bdry=False
)


sol_exact_test = exact_solution(test_grid_x, test_grid_y)
sol_min = torch.min(sol_exact_test)
sol_max = torch.max(sol_exact_test)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh() # specified sigma(z) = tanh(z)
        self.layer_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.activation(out)
        out = self.layer_2(out)
        return out

N_neurons = 10 
input_dim = 2 
output_dim = 1
model = NeuralNet(input_dim, N_neurons, output_dim)


def A_term(x, y):
    return y.pow(2) * torch.sin(torch.pi * x)

# B(x,y) is zero on Dirichlet boundaries (x=0, x=1, y=0)
# and B_y is zero on Neumann boundary (y=1)
# B = x*(1-x)*y*(1-y)^2
def B_term(x, y):
    return x * (1.0 - x) * y * (1.0 - y).pow(2)

# Trial Solution U = A + B * NN
def trial_solution(x, y, neural_network):
    if x.dim() == 0: x = x.unsqueeze(0)
    if y.dim() == 0: y = y.unsqueeze(0)
    if x.dim() == 1: x = x.unsqueeze(1)
    if y.dim() == 1: y = y.unsqueeze(1)
    coords = torch.cat((x, y), dim=1)
    nn_output = neural_network(coords)
    A = A_term(x, y)
    B = B_term(x, y)
    U = A + B * nn_output
    return U

def loss_function(xy_train, neural_network, rhs_func):
    xy_train = xy_train.detach().clone().requires_grad_(True)
    x_train, y_train = xy_train[:, 0:1], xy_train[:, 1:2]

    U = trial_solution(x_train, y_train, neural_network)

    #First derivatives
    grad_U = torch.autograd.grad(U, xy_train, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    U_x = grad_U[:, 0:1]
    U_y = grad_U[:, 1:2]

    #Second derivatives
    grad_Ux = torch.autograd.grad(U_x, xy_train, grad_outputs=torch.ones_like(U_x), create_graph=True)[0]
    U_xx = grad_Ux[:, 0:1]

    grad_Uy = torch.autograd.grad(U_y, xy_train, grad_outputs=torch.ones_like(U_y), create_graph=True)[0]
    U_yy = grad_Uy[:, 1:2]

    #residual
    residual = U_xx + U_yy - rhs_func(x_train, y_train)

    #Error loss
    loss = torch.mean(residual.pow(2)) # Corresponds to 1/N_tr * sum(residual^2)

    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 10000
log_frequency = 500
loss_history = []
start_time = time.time()
for epoch in range(n_epochs):
    optimizer.zero_grad() # Reset gradients

    loss = loss_function(train_data, model, rhs_f)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % log_frequency == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6e}')

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
print(f"Final Loss: {loss_history[-1]:.6e}")

model.eval() 
with torch.no_grad(): 
    sol_computed_test = trial_solution(
        torch.flatten(test_grid_x),
        torch.flatten(test_grid_y),
        model
    ).reshape(test_grid_x.shape)

error = torch.abs(sol_computed_test - sol_exact_test)
error_max = torch.max(error).item()
error_l2_rel = torch.linalg.norm(sol_computed_test - sol_exact_test) / torch.linalg.norm(sol_exact_test)

print(f"\nMax absolute error on test grid: {error_max:.6e}")
print(f"Relative L2 error on test grid: {error_l2_rel:.6e}")

# Plot Loss History
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.yscale('log')
plt.title('Loss Function vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Plot Computed Solution
plt.figure(figsize=(8, 6))
contour = plt.contourf(test_grid_x.numpy(), test_grid_y.numpy(), sol_computed_test.numpy(), 20, cmap='viridis')
plt.colorbar(contour, label='Computed U(x,y)')
plt.title('Computed Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Plot Exact Solution
plt.figure(figsize=(8, 6))
contour_exact = plt.contourf(test_grid_x.numpy(), test_grid_y.numpy(), sol_exact_test.numpy(), 20, cmap='viridis')
plt.colorbar(contour_exact, label='Exact U(x,y)')
plt.title('Exact Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Plot Absolute Error
plt.figure(figsize=(8, 6))
contour_error = plt.contourf(test_grid_x.numpy(), test_grid_y.numpy(), error.numpy(), 20, cmap='plasma')
plt.colorbar(contour_error, label='|Computed U - Exact U|')
plt.title(f'Absolute Error (Max: {error_max:.2e})')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show()