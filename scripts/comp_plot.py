import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_diagram():
    # Set up the figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.5)

    # Left Panel: Ferrero et al.'s Method
    x_sparse = np.random.uniform(1, 9, 15)
    y_sparse = np.random.uniform(1, 9, 15)
    axs[0].scatter(x_sparse, y_sparse, s=100, c='red', label="Sparse Points")
    for i in range(len(x_sparse)):
        axs[0].add_patch(plt.Circle((x_sparse[i], y_sparse[i]), 0.6, color='blue', alpha=0.3, lw=1))
    axs[0].set_title("Ferrero et al.'s Method", fontsize=16)
    axs[0].set_xlim(0, 10)
    axs[0].set_ylim(0, 10)
    axs[0].set_xlabel("X-Axis", fontsize=12)
    axs[0].set_ylabel("Y-Axis", fontsize=12)
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].text(0.5, 9.5, "Descriptors tied to resolution", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.8))

    # Right Panel: Hierarchical Descriptors
    x_dense = np.linspace(1, 9, 10)
    y_dense = np.linspace(1, 9, 10)
    x_grid, y_grid = np.meshgrid(x_dense, y_dense)
    axs[1].scatter(x_grid.flatten(), y_grid.flatten(), s=80, c='green', label="Dense Surface")
    for x, y in zip(x_grid.flatten(), y_grid.flatten()):
        axs[1].add_patch(plt.Circle((x, y), 0.3, color='purple', alpha=0.3, lw=1))
    axs[1].set_title("Enhanced Hierarchical Descriptors", fontsize=16)
    axs[1].set_xlim(0, 10)
    axs[1].set_ylim(0, 10)
    axs[1].set_xlabel("X-Axis", fontsize=12)
    axs[1].set_ylabel("Y-Axis", fontsize=12)
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].text(0.5, 9.5, "Independent of resolution", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.8))

    # Save the diagram
    plt.savefig("Hierarchical_Descriptors_Comparison.png")
    plt.show()

if __name__ == "__main__":
    generate_comparison_diagram()
