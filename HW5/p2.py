import numpy as np
import matplotlib.pyplot as plt

def eigenvector_2d(k_x, k_y, J):

    h = 1 / J
    x = np.linspace(h, 1 - h, J - 1)
    y = np.linspace(h, 1 - h, J - 1)
    X, Y = np.meshgrid(x, y)  # Create a meshgrid
    V = np.sin(k_x * X) * np.sin(k_y * Y)
    return V

J = 10  # Mesh points
h = 1 / J

cases = [
    {"k_x": np.pi, "k_y": np.pi, "label": "k_x = π, k_y = π"},
    {"k_x": np.pi, "k_y": 2 * np.pi, "label": "k_x = π, k_y = 2π"},
    {"k_x": (J - 1) * np.pi, "k_y": (J - 1) * np.pi, "label": "k_x = k_y = (J-1)π"},
]

plt.figure(figsize=(10, 5))
for i, case in enumerate(cases):
    V = eigenvector_2d(case["k_x"], case["k_y"], J)
    plt.subplot(1, 3, i + 1)
    img = plt.imshow(V, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    plt.colorbar(img)
    plt.title(case["label"])
    plt.xlabel("x")
    plt.ylabel("y")
plt.tight_layout()
plt.show()