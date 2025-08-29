# 3D heat-colored discriminant surface comparison
# Left: Single Gaussian vs Single Gaussian
# Right: Multiple Gaussians vs Multiple Gaussians
# Henry — minimal deps: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection


def gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Multivariate normal (2D) PDF. x shape (..., 2).
    """
    d = mean.shape[0]
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    diff = x - mean
    expo = -0.5 * np.einsum('...i,ij,...j->...', diff, cov_inv, diff)
    norm = 1.0 / (np.sqrt((2 * np.pi) ** d * det) + 1e-12)
    return norm * np.exp(expo)


def mixture_pdf(x: np.ndarray,
                means: np.ndarray,
                covs: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
    """
    Weighted Gaussian mixture PDF. weights should sum to 1.
    """
    pdf = np.zeros(x.shape[:-1])
    for m, S, w in zip(means, covs, weights):
        pdf += w * gaussian_pdf(x, m, S)
    return pdf


def plot_single_gaussian_comparison(ax, X1, X2, X):
    """Plot single Gaussian vs single Gaussian comparison"""
    # Class 1: single Gaussian
    mean1 = np.array([6.0, 6.0])
    cov1 = np.array([[1.5, 0.2], [0.2, 1.5]])
    
    # Class 2: single Gaussian
    mean2 = np.array([4.0, 4.0])
    cov2 = np.array([[1.8, -0.1], [-0.1, 1.8]])
    
    # PDFs & discriminant
    p1 = gaussian_pdf(X, mean1, cov1)
    p2 = gaussian_pdf(X, mean2, cov2)
    eps = 1e-18
    G = np.log(p1 + eps) - np.log(p2 + eps)
    
    # Colors and shading
    norm = Normalize(vmin=np.min(G), vmax=np.max(G))
    cmap = plt.cm.RdYlBu_r
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(norm(G), cmap=cmap, vert_exag=0.7,
                      dx=X1[0,1] - X1[0,0], dy=X2[1,0] - X2[0,0])
    
    colors = cmap(norm(G))
    colors[..., :3] = colors[..., :3] * 0.7 + shaded[..., :3] * 0.3
    
    # Plot surface
    ax.plot_surface(X1, X2, G,
                    facecolors=colors,
                    rstride=1, cstride=1,
                    antialiased=True, linewidth=0)
    
    # Decision boundary
    z0 = G.min() - 0.05 * (G.max() - G.min())
    ax.contour(X1, X2, G, levels=[0.0], zdir='z', offset=z0, linewidths=3, colors='black')
    
    # Labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Discriminant Function')
    ax.set_title('Class 1: Single Gaussian\nClass 2: Single Gaussian')
    ax.set_zlim(z0, G.max())
    
    return G, norm, cmap


def plot_multiple_gaussian_comparison(ax, X1, X2, X):
    """Plot multiple Gaussians vs multiple Gaussians comparison"""
    # Class 1: three Gaussians
    means1 = np.array([
        [3.0, 3.0],
        [7.0, 4.5],
        [5.5, 8.0],
    ])
    covs1 = np.array([
        [[1.2, 0.3], [0.3, 1.0]],
        [[1.0, -0.2], [-0.2, 1.3]],
        [[1.1, 0.4], [0.4, 1.5]],
    ])
    weights1 = np.array([0.35, 0.35, 0.30])

    # Class 2: two Gaussians
    means2 = np.array([
        [2.5, 7.0],
        [8.0, 7.5],
    ])
    covs2 = np.array([
        [[1.6, -0.2], [-0.2, 1.0]],
        [[1.2, 0.2], [0.2, 1.4]],
    ])
    weights2 = np.array([0.55, 0.45])

    # PDFs & discriminant
    p1 = mixture_pdf(X, means1, covs1, weights1)
    p2 = mixture_pdf(X, means2, covs2, weights2)
    eps = 1e-18
    G = np.log(p1 + eps) - np.log(p2 + eps)
    
    # Colors and shading
    norm = Normalize(vmin=np.min(G), vmax=np.max(G))
    cmap = plt.cm.RdYlBu_r
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(norm(G), cmap=cmap, vert_exag=0.7,
                      dx=X1[0,1] - X1[0,0], dy=X2[1,0] - X2[0,0])
    
    colors = cmap(norm(G))
    colors[..., :3] = colors[..., :3] * 0.7 + shaded[..., :3] * 0.3
    
    # Plot surface
    ax.plot_surface(X1, X2, G,
                    facecolors=colors,
                    rstride=1, cstride=1,
                    antialiased=True, linewidth=0)
    
    # Decision boundary
    z0 = G.min() - 0.05 * (G.max() - G.min())
    ax.contour(X1, X2, G, levels=[0.0], zdir='z', offset=z0, linewidths=3, colors='black')
    
    # Labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Discriminant Function')
    ax.set_title('Class 1: Three Gaussians\nClass 2: Two Gaussians')
    ax.set_zlim(z0, G.max())
    
    return G, norm, cmap


def main():
    # ----- Grid -----
    x1 = np.linspace(2, 10, 300)
    x2 = np.linspace(2, 10, 300)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1, X2], axis=-1)

    # ----- Create subplots -----
    fig = plt.figure(figsize=(20, 8))
    
    # Left subplot: Single Gaussian comparison
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=25, azim=45)
    G1, norm1, cmap1 = plot_single_gaussian_comparison(ax1, X1, X2, X)
    
    # Right subplot: Multiple Gaussian comparison
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(elev=25, azim=45)
    G2, norm2, cmap2 = plot_multiple_gaussian_comparison(ax2, X1, X2, X)
    
    # ----- Add colorbars -----
    # Left colorbar
    mappable1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap1)
    mappable1.set_array(G1)
    cbar1 = fig.colorbar(mappable1, ax=ax1, shrink=0.7, pad=0.1)
    cbar1.set_label("Discriminant Function")
    
    # Right colorbar
    mappable2 = plt.cm.ScalarMappable(norm=norm2, cmap=cmap2)
    mappable2.set_array(G2)
    cbar2 = fig.colorbar(mappable2, ax=ax2, shrink=0.7, pad=0.1)
    cbar2.set_label("Discriminant Function")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
