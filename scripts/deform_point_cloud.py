#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pathlib
from sklearn.neighbors import NearestNeighbors
import helperfunctions as hf
from scipy.spatial import cKDTree
import non_rigid_registration as nrr
import open3d as o3d
from matplotlib import colors, cm
import matplotlib.pyplot as plt

# def deform_pointcloud(P1, T12, corres, S1, S2):
#   """
#   Deform a pointcloud using deformation parameters computed by registering
#   corresponding skeletons.
#   ALGORITHM USED HERE:
#   - The first neighbour n1 is the nearest node
#   - The second neighbour n2 is a node connected to n1. If n1 has more
#     than one neigbour (the standard case), then the one on the 'right' side
#     is chosen. 'right' side means that the projection lies really on the
#     edge between n1 and n2.
#
#   Parameters
#   ----------
#   P1 : numpy array (Nx3)
#     XYZ coordinates of the first point cloud
#   T12 : list of 4x4 numpy arrays
#     Affine transformation corresponding to each node in S1
#   corres : numpy array (Mx2)
#     correspondence between two skeleton nodes
#   S1, S2 : Skeleton Class
#     Two skeletons for which we compute the non-rigid registration params
#   Returns
#   -------
#   P1_deformed : numpy array (Nx3)
#     P1 after undergoing deformation params given in T12
#
#   """
#   # Find nearest skeleton node
#   num_points = P1.shape[0]
#   # Matrix of nearby info:
#   # col 1: node index of nearest
#   # col 2: distance to nearest
#   nearest= np.zeros((num_points,2))
#   for i in range(num_points):
#     diff = S1.XYZ- P1[i,:]
#     dist = np.sqrt(np.sum(diff**2, axis=1))
#     indices = np.argsort(dist)
#     nearest[i,:] = np.array([[indices[0], dist[indices[0]]]])
#
#   # perform deformation
#   P1_deformed = P1.copy()
#   for i in range(num_points):
#     n1 = int(nearest[i,0])
#     n2_candidates = S1.A[n1,:].nonzero()[0]
#     # in between:
#     # >0 weight f2 for node n2
#     # -1 == is on the other side of n1
#     # -2 == is on the other side of n2
#     in_between = np.zeros(len(n2_candidates))
#     for j in range(len(n2_candidates)):
#       n2 = n2_candidates[j]
#       line_direction = S1.XYZ[n2,:] - S1.XYZ[n1,:]
#       # projection relative to n1
#       line_projection = (((P1[i,:]-S1.XYZ[n1,:]) @ line_direction) / (line_direction@line_direction) ) * line_direction
#       # absolute projection
#       projection = S1.XYZ[n1,:] + line_projection
#
#       f2 = np.linalg.norm(line_projection)/np.linalg.norm(line_direction)
#       in_between[j] = f2
#
#       if np.dot(line_projection, line_direction) < 0:
#         # The projected point does not lie in between the two nodes, it is on
#         # the other side of nearest1
#         # ==> use another node or only nearest1 trafo
#         in_between[j] = -1
#       else:
#         if f2 > 1:
#           # The projected point does not lie in between the two nodes, it is on
#           # the other side of nearest2
#           # ==> use another node or only nearest2 trafo
#           in_between[j] = -2
#
#     # find best candidate
#     J = (in_between>=0).nonzero()[0]
#     if len(J) > 0:
#       # in-between node
#       if len(J) == 1:
#         n2 = n2_candidates[J[0]]
#         f2 = in_between[J[0]]
#       else:
#         # more than one in between: can happen for nodes with degree>=3:
#         # choose the n2, where the projection is nearest to the point
#         dmin = np.inf
#         for j in range(len(J)):
#           # projection absolute
#           projection = S1.XYZ[n1,:] + ( ( (P1[i,:]-S1.XYZ[n1,:]) @ line_direction) / (line_direction @ line_direction) ) * line_direction
#           d = np.linalg.norm(projection - P1[i,:])
#
#           if d < dmin:
#             dmin = d
#             n2 = n2_candidates[J[j]]
#             f2 = in_between[J[j]]
#     else:
#       n2 = n2_candidates[0]
#       if in_between[0] == -1:
#         f2 = 0
#       else:
#         f2 = 1
#
#     # apply transformation to the point
#     f1 = 1 - f2
#     T1 = T12[n1]
#     T2 = T12[n2]
#     Pn1 = hf.hom2euc(T1 @ hf.euc2hom(np.reshape(P1[i,:],(3,1))))
#     Pn2 = hf.hom2euc(T2 @ hf.euc2hom(np.reshape(P1[i,:],(3,1))))
#     P1_deformed[i,:] = (f1*Pn1 + f2*Pn2).flatten()
#
#   return P1_deformed

def is_valid_transformation(T):
    if T.shape == (4, 4) and np.isfinite(T).all():
        return True
    return False

def deform_pointcloud(P1, T12, corres, S1, S2):
    """
    Deform a pointcloud using deformation parameters computed by registering corresponding skeletons.
    """
    num_points = P1.shape[0]

    # Find nearest skeleton node
    nearest = np.zeros((num_points, 2))

    for i in range(num_points):
        diff = S1.XYZ - P1[i, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        indices = np.argsort(dist)
        nearest[i, :] = np.array([indices[0], dist[indices[0]]])

    # Perform deformation
    P1_deformed = P1.copy()

    for i in range(num_points):
        n1 = int(nearest[i, 0])
        n2_candidates = S1.A[n1, :].nonzero()[0]
        best_n2 = None
        best_f2 = None
        line_direction = None

        # Calculate the weight and projection
        for n2 in n2_candidates:
            line_direction = S1.XYZ[n2, :] - S1.XYZ[n1, :]

            if np.linalg.norm(line_direction) == 0:
                continue

            # Projection onto the line
            projection = np.dot(P1[i, :] - S1.XYZ[n1, :], line_direction) / np.dot(line_direction, line_direction)
            projection_clamped = np.clip(projection, 0, 1)  # Ensure within bounds

            f2 = projection_clamped
            if best_f2 is None or abs(f2 - 0.5) < abs(best_f2 - 0.5):
                best_f2 = f2
                best_n2 = n2

        if best_n2 is not None:
            f2 = best_f2
            f1 = 1 - f2

            # Check if transformations are valid
            if is_valid_transformation(T12[n1]) and is_valid_transformation(T12[best_n2]):
                T1 = T12[n1]
                T2 = T12[best_n2]

                # Apply transformation to the point
                Pn1 = hf.hom2euc(T1 @ hf.euc2hom(np.reshape(P1[i, :], (3, 1))))
                Pn2 = hf.hom2euc(T2 @ hf.euc2hom(np.reshape(P1[i, :], (3, 1))))

                P1_deformed[i, :] = (f1 * Pn1 + f2 * Pn2).flatten()
        else:
            # If no valid second neighbor is found, apply T1 directly
            if is_valid_transformation(T12[n1]):
                T1 = T12[n1]
                Pn1 = hf.hom2euc(T1 @ hf.euc2hom(np.reshape(P1[i, :], (3, 1))))
                P1_deformed[i, :] = Pn1.flatten()

    return P1_deformed

def load_pointcloud(pc_file):
  file_ext = pathlib.Path(pc_file).suffix
  if file_ext == '.txt' or file_ext == '.xyz':
    P = np.loadtxt(pc_file)
  elif file_ext == '.npy' or file_ext == '.npz':
    P = np.load(pc_file)
  else:
    print('Pointcloud file type not supported.')

  return P

def downsample_pointcloud(P, ds):
  P_ds = P[::ds,:].copy()
  return P_ds

def pcd2xyz(pcd):
    return np.asarray(pcd.points)

def compute_pc_resolution(pcd):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    sum = 0.0
    pcd_np = pcd2xyz((pcd))
    print("total no. of pts in point cloud:", pcd_np.shape[0])
    for vt in pcd_np:
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(vt, 2)
        sum = sum + dist[1]
    res = sum/pcd_np.shape[0]
    print("pt. cloud resolution:",res)
    return res



def deform_pointcloud_MLS(P1, T12, corres, S1, S2, neighborhood_size=5):
    """
    Deform a point cloud using deformation parameters computed by registering
    corresponding skeletons.

    Parameters
    ----------
    P1 : numpy array (Nx3)
        XYZ coordinates of the first point cloud
    T12 : list of 4x4 numpy arrays
        Affine transformation corresponding to each node in S1
    corres : numpy array (Mx2)
        correspondence between two skeleton nodes
    S1, S2 : Skeleton Class
        Two skeletons for which we compute the non-rigid registration params
    neighborhood_size : int, optional
        Number of nearest neighbors to consider in the local neighborhood

    Returns
    -------
    P1_deformed : numpy array (Nx3)
        P1 after undergoing deformation parameters given in T12
    """

    num_points = P1.shape[0]
    P1_deformed = P1.copy()

    # Compute nearest neighbors for each point
    nn = NearestNeighbors(n_neighbors=neighborhood_size)
    nn.fit(S1.XYZ)
    _, indices = nn.kneighbors(P1)

    for i in range(num_points):
        neighborhood_indices = indices[i]
        neighborhood_points = S1.XYZ[neighborhood_indices]

        # Placeholder for MLS-based surface fitting
        surface_params = fit_surface_to_neighborhood(neighborhood_points)

        # Placeholder for MLS-based deformation computation
        deformation_params = compute_deformation(surface_params, P1[i], T12, S1, P1)

        # Apply deformation to the current point
        P1_deformed[i] = apply_deformation(P1[i], deformation_params)

    return P1_deformed

def fit_surface_to_neighborhood(neighborhood_points):
    """
    # Placeholder for MLS-based surface fitting algorithm.
    #
    # Parameters
    # ----------
    # neighborhood_points : numpy array
    #     XYZ coordinates of the neighborhood points
    #
    # Returns
    # -------
    # surface_params : object
    #     Parameters of the fitted surface (e.g., plane coefficients)
    # """
    # # Implement MLS-based surface fitting algorithm here
    # # Return the surface parameters
    # Compute the centroid of the neighborhood points
    centroid = np.mean(neighborhood_points, axis=0)

    # Compute the covariance matrix of the neighborhood points
    cov_matrix = np.cov(neighborhood_points.T)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Find the eigenvector corresponding to the smallest eigenvalue
    min_eigenvalue_index = np.argmin(eigenvalues)
    normal_vector = eigenvectors[:, min_eigenvalue_index]

    # Compute the distance from the centroid to the plane
    distance = -np.dot(normal_vector, centroid)

    # Construct the surface parameters dictionary
    surface_params = {'normal_vector': normal_vector, 'distance': distance}

    return surface_params


def compute_deformation(surface_params, point, T12, S1, P1):

    # Extract surface parameters
    normal_vector = surface_params['normal_vector']
    distance = surface_params['distance']

    # Find nearest skeleton node
    num_points = P1.shape[0]
    nearest = np.zeros((num_points, 2))
    for i in range(num_points):
        diff = S1.XYZ - P1[i, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        index = np.argmin(dist)
        nearest[i, :] = [index, dist[index]]

    # Find the two nearest nodes for the given point
    nearest_indices = nearest[np.argsort(nearest[:, 1])][:2, 0]
    n1, n2 = nearest_indices.astype(int)


    # Compute the weights for the two nearest nodes
    diff1 = point - S1.XYZ[n1, :]
    diff2 = point - S1.XYZ[n2, :]
    distance1 = np.linalg.norm(diff1)
    distance2 = np.linalg.norm(diff2)
    total_distance = distance1 + distance2
    weight1 = distance2 / total_distance
    weight2 = distance1 / total_distance

    # Apply transformation to the point using the weighted average of node transformations
    T1 = T12[n1]
    T2 = T12[n2]
    Pn1 = hf.hom2euc(T1 @ hf.euc2hom(np.reshape(point, (3, 1))))
    Pn2 = hf.hom2euc(T2 @ hf.euc2hom(np.reshape(point, (3, 1))))
    deformed_point = (weight1 * Pn1 + weight2 * Pn2).flatten()

    return deformed_point

def apply_deformation(point, deformation_params):
    """
    Apply deformation to the given point using the deformation parameters.

    Parameters
    ----------
    point : numpy array
        XYZ coordinates of the point
    deformation_params : numpy array
        Deformation parameters to be applied to the point

    Returns
    -------
    deformed_point : numpy array
        Deformed XYZ coordinates of the point
    """

    # Apply deformation to the point
    deformed_point = point + deformation_params

    return deformed_point

def ransac_weighted_line_projection(P1, T12,  S1, num_iterations=100, threshold=np.inf):
    """
    Deform a point cloud using deformation parameters computed by registering
    corresponding skeletons with RANSAC-based outlier rejection.

    Parameters
    ----------
    P1 : numpy array (Nx3)
        XYZ coordinates of the first point cloud
    T12 : list of 4x4 numpy arrays
        Affine transformation corresponding to each node in S1
    corres : numpy array (Mx2)
        Correspondence between two skeleton nodes
    S1, S2 : Skeleton Class
        Two skeletons for which we compute the non-rigid registration params
    num_iterations : int, optional
        Number of RANSAC iterations (default is 100)
    threshold : float, optional
        Distance threshold for considering points as inliers (default is 0.1)

    Returns
    -------
    P1_deformed : numpy array (Nx3)
        P1 after undergoing deformation params given in T12
    """
    best_inlier_count = 0
    best_P1_deformed = None

    for _ in range(num_iterations):
        # Randomly sample a subset of points
        sample_indices = np.random.choice(len(P1), size=3, replace=False)
        sample_points = P1[sample_indices]

        # Compute the line passing through the sampled points
        line_direction = np.cross(sample_points[1] - sample_points[0], sample_points[2] - sample_points[0])
        line_direction /= np.linalg.norm(line_direction)

        # Compute distances of all points to the line
        distances = np.abs(np.dot(P1 - sample_points[0], line_direction))

        # Count inliers based on threshold
        inliers = distances < threshold
        inlier_count = np.sum(inliers)

        # Update best solution if current solution has more inliers
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_P1_deformed = weighted_line_projection(P1[inliers], T12, S1)

    return best_P1_deformed

def weighted_line_projection(P1, T12, S1):
    """
    Deform a pointcloud using deformation parameters computed by registering
    corresponding skeletons.

    Parameters
    ----------
    P1 : numpy array (Nx3)
        XYZ coordinates of the first point cloud
    T12 : list of 4x4 numpy arrays
        Affine transformation corresponding to each node in S1
    corres : numpy array (Mx2)
        Correspondence between two skeleton nodes
    S1, S2 : Skeleton Class
        Two skeletons for which we compute the non-rigid registration params

    Returns
    -------
    P1_deformed : numpy array (Nx3)
        P1 after undergoing deformation params given in T12
    """
    num_points = P1.shape[0]
    P1_deformed = np.zeros_like(P1)

    for i in range(num_points):
        n1, n2, f2 = find_nearest_nodes(P1[i], S1)
        T1 = T12[n1]
        T2 = T12[n2]
        Pn1 = np.dot(T1, np.hstack((P1[i], 1)))[:3]
        Pn2 = np.dot(T2, np.hstack((P1[i], 1)))[:3]
        P1_deformed[i] = (1 - f2) * Pn1 + f2 * Pn2

    return P1_deformed

def find_nearest_nodes(point, S1):
    diff = S1.XYZ - point
    dist = np.linalg.norm(diff, axis=1)
    n1 = np.argmin(dist)
    n2_candidates = np.where(S1.A[n1] != 0)[0]
    line_direction = S1.XYZ[n2_candidates[0]] - S1.XYZ[n1]
    line_projection = np.dot(point - S1.XYZ[n1], line_direction) / np.dot(line_direction, line_direction) * line_direction
    f2 = np.linalg.norm(line_projection) / np.linalg.norm(line_direction)
    return n1, n2_candidates[0], f2

def mean_square_error(point_cloud_A, point_cloud_B):
    """
    Compute the mean square error between two point clouds.

    Parameters
    ----------
    point_cloud_A : numpy array
        Point cloud A with shape (N, 3).
    point_cloud_B : numpy array
        Point cloud B with shape (M, 3).

    Returns
    -------
    avg_mse : float
        Average mean square error between the two point clouds.
    """
    # Construct KD tree for fast nearest neighbor search
    tree_B = cKDTree(point_cloud_B)

    # Query nearest neighbor for each point in point_cloud_A
    distances, _ = tree_B.query(point_cloud_A)

    # Compute mean square error for each point
    mse = distances ** 2

    # Compute average mean square error
    avg_mse = np.mean(mse)

    return avg_mse, mse

def landmark_Accuracy( S1, S2, corres, Tfs, th, src_pts_1, src_pts_2):
    # for each skeleton node of S1 and S2 get the nearest point set, s1_enarest, s2_nearest using tree_A, and tree_B
    # S1.XYZ gives the array of skeleton pts of shape (n, 3) same for S2
    # Now transform nearest point obtained using S1, i.e s1_nearest and tree_A with transform tfs to get S1_hat
    #compute
    # Initialize an empty array to store transformed skeleton points
    # Construct KD tree for fast nearest neighbor search
    src_pts_A =  np.asarray(src_pts_1.points)
    src_pts_B =  np.asarray(src_pts_2.points)
    tree_A = cKDTree(src_pts_A)
     # Construct KD tree for fast nearest neighbor search
    tree_B = cKDTree(src_pts_B )
    S1_hat = np.empty((len(Tfs), 3))
    # Initialize empty arrays to store nearest points for S1 and S2
    s1_nearest = np.empty((len(Tfs), 3))
    s2_nearest = np.empty((len(Tfs), 3))
    for i in range (corres.shape[0]):
        # Get the nearest point in S1 using tree_A
        nearest_point_index_s1 = tree_A.query(S1.XYZ[corres[i, 0]])[1]
        s1_nearest[i] = src_pts_A[nearest_point_index_s1]

        # Get the nearest point in S2 using tree_B
        nearest_point_index_s2 = tree_B.query(S2.XYZ[corres[i, 1]])[1]
        s2_nearest[i] = src_pts_B[nearest_point_index_s2]
        # Transform nearest point obtained using S1 and tree_A with transform Tfs to get S1_hat
        S1_hat[i] = np.dot(Tfs[corres[i, 0]], np.hstack((s1_nearest[i], 1)))[:3]
    err = np.linalg.norm(S1_hat - s2_nearest, axis = 1)
    lm_precision = np.sum(err < 2 * th) / err.shape[0]
    return lm_precision

def continuity_measure(S1, S2,corres, Tfs, th,point_cloud,sample_proportion = 0.5):
    sample_step = max(int(1 / sample_proportion), 1)
    tree = cKDTree(point_cloud)
    dist, NN_vertices = tree.query(np.asarray(point_cloud), k=2)
    pass_cnt = 0
    total_cnt = 0
    for i in range(0, NN_vertices.shape[0], sample_step):
        pass
    pass


def color_point_cloud_based_on_error(pcd, errors, mse):
    # Convert the point cloud to a NumPy array
    points = np.asarray(pcd.points)

    colors = np.zeros((points.shape[0], 3))

    # Define thresholds
    threshold1 = mse
    threshold2 = 3 * mse

    # Assign colors based on error thresholds
    for i in range(len(errors)):
        if errors[i] <= threshold1:
            colors[i] = [0, 1, 0]  # Green
        elif threshold1 < errors[i] <= threshold2:
            colors[i] = [1, 1, 0]  # Yellow
        else:
            colors[i] = [1, 0, 0]  # Red

    # # Assign colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Normalize errors to range between 0 and 1
    # norm_errors = (errors - min(errors)) / (max(errors) - min(errors))
    #
    # # Get colors from colormap
    # colormap = cm.get_cmap('jet')
    # colors = colormap(norm_errors)[:, :3]  # Get RGB values
    #
    # # Assign colors to the point cloud
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd

def display_vertical_color_bar(errors, mse):
    # Normalize errors to range between 0 and 1 for the color bar
    norm_errors = (errors - min(errors)) / (max(errors) - min(errors))

    # Define the colormap and normalization
    colormap = colors.ListedColormap(['green', 'yellow', 'red'])
    bounds = [0, mse, 3 * mse, max(errors)]
    norm = colors.BoundaryNorm(bounds, colormap.N)

    # Create a figure and color bar
    fig, ax = plt.subplots(figsize=(2, 8))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=ax, orientation='vertical')
    cbar.set_label('Error Value')
    cbar.set_ticks([0, mse, 3 * mse, max(errors)])
    cbar.set_ticklabels(['<= MSE', 'MSE', '3x MSE', '> 3x MSE'])

    plt.show()





