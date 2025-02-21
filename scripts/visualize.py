import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pointcloud(fh, P, color='r'):
  #ax = fh.gca(projection='3d')
  ax = fh.add_subplot(projection='3d')
  ax.scatter3D(P[:,0], P[:,1], P[:,2], '.', color=color, depthshade=False)

  
def plot_semantic_pointcloud(fh, array_list):
    """
    :param array_list: a list of numpy arrays, each array represents a different class
    """
    #ax = fh.gca(projection='3d')
    ax = fh.add_subplot(projection='3d')
    
    for P in array_list:
        P = np.asarray(P)
        ax.scatter3D(P[:,0], P[:,1], P[:,2], '.')

def plot_skeleton(ax, S, color='blue'):
    """ plots the skeleton graph with nodes and edges """
    # plot vertices
    #ax = fh.gca(projection='3d')
     # Plot vertices
    ax.scatter(S.XYZ[:, 0], S.XYZ[:, 1], S.XYZ[:, 2], marker='o', c=color)

    # Plot edges
    N = S.A.shape[0]
    for i in range(N):
        for j in range(N):
            if S.A[i, j] == 1:
                ax.plot([S.XYZ[i, 0], S.XYZ[j, 0]], \
                          [S.XYZ[i, 1], S.XYZ[j, 1]], \
                          [S.XYZ[i, 2], S.XYZ[j, 2]], color)
    
    
def plot_skeleton_correspondences(ax, S1, S2, corres, color='black', corres_filter = False):

    # Delete the correspondences related to virtual nodes
    if corres_filter == True:
        ind_remove = np.where(corres[:, 0] == -1)
        corres = np.delete(corres, ind_remove, axis=0)
        ind_remove = np.where(corres[:, 1] == -1)
        corres = np.delete(corres, ind_remove, axis=0)

    # Plot correspondences
    N = corres.shape[0]
    for i, (src_idx, dst_idx) in enumerate(corres):
        ax.plot(
            [S1.XYZ[corres[i, 0], 0], S2.XYZ[corres[i, 1], 0]],
            [S1.XYZ[corres[i, 0], 1], S2.XYZ[corres[i, 1], 1]],
            [S1.XYZ[corres[i, 0], 2], S2.XYZ[corres[i, 1], 2]],
            color,
        )
        ax.text(S1.XYZ[src_idx, 0], S1.XYZ[src_idx, 1], S1.XYZ[src_idx, 2], str(src_idx), color=color,size=8)
        ax.text(S2.XYZ[dst_idx, 0], S2.XYZ[dst_idx, 1], S2.XYZ[dst_idx, 2], str(dst_idx), color=color,size=8)

    ax.scatter(
        S1.XYZ[corres[:, 0], 0],
        S1.XYZ[corres[:, 0], 1],
        S1.XYZ[corres[:, 0], 2],
        marker='o',
        c=color,
    )

    ax.scatter(
        S2.XYZ[corres[:, 1], 0],
        S2.XYZ[corres[:, 1], 1],
        S2.XYZ[corres[:, 1], 2],
        marker='o',
        c=color,
    )

