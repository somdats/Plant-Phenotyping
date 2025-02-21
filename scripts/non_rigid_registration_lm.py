import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from collections import namedtuple

import skeleton as skel
import skeleton_matching as skm
import robust_functions as rf
import helperfunctions as hf
import visualize as vis


def check_jacobian_condition(J):
    """
    Computes the condition number of the Jacobian matrix to check for ill-conditioning.
    """
    U, S, Vt = np.linalg.svd(J)  # Singular Value Decomposition (SVD)
    condition_number = S.max() / S.min()  # Compute condition number

    print(f"Condition number of Jacobian: {condition_number}")

    # if condition_number > 1e6:  # Threshold for ill-conditioning
    #     #print("Warning: Jacobian is ill-conditioned! Consider rescaling or adjusting constraints.")
    # else:
    #     print("Jacobian is well-conditioned.")

    return condition_number


# Placeholder for the helper functions module
class hf:
    @staticmethod
    def M(R, t):
        """Construct a homogeneous transformation matrix from rotation R and translation t."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

def register_skeleton(S1, S2, corres, params):
    """
    Registers S1 to S2 using the Levenberg-Marquardt algorithm.
    """
    # Initialize parameters
    print("Beginning of Registration:\n")
    x0 = initialize_parameters(S1, params)
    #print( "x array type:", x0.dtype)

    # Define the objective function for least_squares
    def objective_function(tf):
        res = compute_total_residuals(tf, S1, S2, corres, params)
        if not np.all(np.isfinite(res)):
           print("Objective function returns non-finite values")
        return res

    # Optionally, define a function to compute the Jacobian
    def jacobian_function(tf):
        jacobian = compute_total_jacobian(tf, S1, S2, corres, params)
        if not np.all(np.isfinite(jacobian)):
            print("Jacobian function returns non-finite values")
        return jacobian

    # Perform optimization
    print("start optimization\n")
    result = least_squares(objective_function, x0, jac=jacobian_function, method='lm',
                           ftol= 1e-5, xtol=1e-5, gtol=1e-5,x_scale='jac',max_nfev=50, verbose=2)

    # Extract the optimized parameters
    print("end optimization\n")
    optimized_params = result.x

    # Convert optimized parameters to transformation matrices for each node
    #print("ftol:", result.ftol)
    T12 = convert_params_to_transformations(optimized_params, S1)

    return T12

def initialize_parameters(S1, params):
    # Example: Flatten rotation matrices and translation vectors for all nodes in S1
    x0 = np.array([])
    for node in range (S1.XYZ.shape[0]):
        R_init = np.eye(3)  # Initial rotation (identity matrix)
        t_init = np.zeros(3)  # Initial translation (zero vector)
        # Flatten and concatenate R and t for the node
        node_params = np.concatenate((R_init.flatten(), t_init))
        x0 = np.concatenate((x0, node_params))
    return x0


def compute_feature_constraints(correspondences, principal_directions_S1, principal_directions_S2, leaf_widths_S1, leaf_widths_S2):
    direction_residuals = []
    width_residuals = []

    for (i, j) in correspondences:
        # Compute the direction difference
        direction_diff = np.linalg.norm(principal_directions_S1[i] - principal_directions_S2[j])
        direction_residuals.append(direction_diff)

        # Compute the width difference
        width_diff = np.abs(leaf_widths_S1[i] - leaf_widths_S2[j])
        width_residuals.append(width_diff)

    return np.array(direction_residuals), np.array(width_residuals)

def compute_total_residuals(tf, S1, S2, corres, params):
    """
    Compute the total residuals for all constraints.
    """
    residuals = []

    # Compute point-to-point residuals
    pt2pt_residuals = compute_pt2pt_residuals(tf, S1, S2, corres)
    #print("pt2pt_residuals:", pt2pt_residuals)  # Print the regularization residuals
    residuals.extend( 0.001 * pt2pt_residuals)

    # Compute point-to-plane residuals
    pt2pl_residuals = compute_pt2pl_residuals(tf, S1, S2, corres)
    #print("pt2pl_residuals:", pt2pl_residuals)  # Print the regularization residuals
    residuals.extend( 0.001 * pt2pl_residuals)

    # Compute regularization residuals
    reg_residuals = compute_reg_residuals(tf, S1)
    #print("reg_residuals:", reg_residuals)  # Print the regularization residuals

    residuals.extend(reg_residuals)

    # Compute rotational matrix constraints
    rot, trans = extract_transformation(tf, 0)
    _, r_rot = compute_rotation_matrix_constraints(rot)
    residuals.extend( r_rot.flatten())

  # remove the commment if yo want to use feature and leaf width constraint
    # # Compute feature constraints (leaf width and principal direction)
    # E_feat_direction, E_feat_width = compute_feature_constraints(
    #     corres, S1.normals, S2.normals, S1.leafWidths, S2.leafWidths
    # )
    # weighted_feat_width = 0.0001 * E_feat_width
    # weighted_E_feat_direction = 0.001 * E_feat_direction
    # #print("Regularization residuals:", weighted_feat_width)  # Print the regularization residuals
    # # Ensure E_feat_direction and E_feat_width are flattened if they are arrays
    # residuals.extend(weighted_E_feat_direction.flatten())  # Flatten before adding to residuals
    # residuals.extend(weighted_feat_width.flatten())      # Flatten before adding to residuals

    return np.array(residuals, dtype=float)



def compute_total_jacobian(x, S1, S2, corres, params):
    """
    Compute the total Jacobian matrix for all constraints.
    """
    # Dynamically compute the number of residuals by evaluating the objective function
    # This assumes you have a function to compute the total residuals
    total_residuals = compute_total_residuals(x, S1, S2, corres, params)

    # Now, use the length of the total_residuals array to determine the size of the Jacobian matrix
    J = np.zeros((len(total_residuals), len(x)))  # Correctly initialized Jacobian matrix

    # Fill in the Jacobian matrix with the actual derivatives
    # This part of the code would involve calculating derivatives for each constraint type
    # and populating J accordingly
    ############################################
    # Choose an index to test
    # test_index = 0
    #
    # # Create perturbed versions of x
    # x_test_plus = np.array(x, copy=True)
    # x_test_minus = np.array(x, copy=True)
    # x_test_plus[test_index] += 0.1  # A significant change
    # x_test_minus[test_index] -= 0.1
    #
    # # Compute residuals for perturbed parameters
    # res_test_plus = compute_total_residuals(x_test_plus, S1, S2, corres, params)
    # res_test_minus = compute_total_residuals(x_test_minus, S1, S2, corres, params)
    #
    # # Check if there's a significant difference
    # print("Difference in residuals with significant parameter change:", np.linalg.norm(res_test_plus - res_test_minus))
    #################################################
    eps = 1e-6  # Adjust based on sensitivity analysi
    for i in range(len(x)):
        x_plus_eps = np.array(x, copy=True)
        x_minus_eps = np.array(x, copy=True)
        x_plus_eps[i] += eps
        x_minus_eps[i] -= eps
        residuals_plus_eps = compute_total_residuals(x_plus_eps, S1, S2, corres, params)
        residuals_minus_eps = compute_total_residuals(x_minus_eps, S1, S2, corres, params)

        # Central difference approximation
        # print("Difference in residuals with significant parameter change:",
        #       np.linalg.norm(residuals_plus_eps - residuals_minus_eps))
        J[:, i] = (residuals_plus_eps - residuals_minus_eps) / (2 * eps)

    #check_jacobian_condition(J)

    return J


def convert_params_to_transformations(x, S1):
    """
    Convert the optimized parameters back into transformation matrices for each node.
    """
    T12 = []
    for i in range(S1.XYZ.shape[0]):
        R, t = extract_transformation(x, i)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        T12.append(T)
    return T12



def extract_transformation(x, node_index):
    # Assuming 12 parameters per node (9 for rotation matrix, 3 for translation vector)
    start_idx = node_index * 12
    end_idx = start_idx + 12
    node_params = x[start_idx:end_idx]

    # Extract rotation matrix and translation vector
    R = node_params[:9].reshape((3, 3))
    t = node_params[9:]

    return R, t

def compute_pt2pt_residuals(tf, S1, S2, corres):
    residuals = []
    # Apply the transformation to the point in S1
    for idx_pair in corres:
        idx1, idx2 = idx_pair  # Unpack correspondence indices
        # Extract R and t for the specific correspondence from x
        # This requires knowing the structure of x and how R and t are stored for each node
        R, t = extract_transformation(tf, idx1)  # Placeholder for actual extraction logic

        # Apply the transformation to the point in S1
        transformed_point = np.dot(R, S1.XYZ[idx1, :]) + t

        # Corresponding point in S2
        target_point = S2.XYZ[idx2, :]

        # Compute the residual for this correspondence
        # residual = transformed_point - target_point
        # residuals.extend(residual.flatten())
        # Compute the Euclidean distance (L2 norm) as the residual
        residual = np.linalg.norm(transformed_point - target_point)  # Euclidean distance
        residuals.append(residual)
    return np.array(residuals)

def compute_pt2pl_residuals(tf, S1, S2, corres):
    residuals = []
    # Apply the transformation to the point in S1
    for idx_pair in corres:
        idx1, idx2 = idx_pair  # Unpack correspondence indices
        # Extract R and t for the specific correspondence from x
        # This requires knowing the structure of x and how R and t are stored for each node
        R, t = extract_transformation(tf, idx1)  # Placeholder for actual extraction logic
        # Apply the transformation to the point in S1
        transformed_point = np.dot(R, S1.XYZ[idx1, :]) + t

        # Corresponding point and normal in S2
        target_point = S2.XYZ[idx2, :]
        normal = S2.normals[idx2, :]  # Assuming normals are stored in S2

        # Point-to-plane residual
        residual = np.dot(transformed_point - target_point, normal)
        residuals.extend(residual.flatten())

    return np.array(residuals)


def residual_reg(x1, x2):
    """
    Compute the regularization residual between two sets of transformation parameters.

    Parameters:
    - x1: Transformation parameters (rotation and translation) for node j as a flattened array.
    - x2: Transformation parameters (rotation and translation) for node k as a flattened array.

    Returns:
    - A numpy array containing the residuals.
    """
    # Extract rotation and translation from x1 and x2
    Rj = x1[:9].reshape((3, 3))
    tj = x1[9:]
    Rk = x2[:9].reshape((3, 3))
    tk = x2[9:]

    # Compute transformation matrices
    Tj = hf.M(Rj, tj)
    Tk = hf.M(Rk, tk)

    # Compute the residual based on the difference between transformations
    r_eye = Tj @ np.linalg.inv(Tk)
    r = np.vstack((np.reshape(r_eye[0:3, 0:3] - np.eye(3), (9, 1), order='F'),
                   np.reshape(r_eye[0:3, 3], (3, 1), order='F')))

    return r.flatten()

def compute_reg_residuals(x, S1):
    residuals = []
    J = []  # Placeholder for Jacobian if needed
    m = S1.XYZ.shape[0]  # Number of nodes

    for j in range(m):
        Rj, tj = extract_transformation(x, j)  # Extract transformation for node j
        Tj = create_homogeneous_matrix(Rj, tj)

        # Find indices of adjacent nodes
        adj_indices = np.argwhere(S1.A[j, :] == 1).flatten()

        for k in adj_indices:
            Rk, tk = extract_transformation(x, k)  # Extract transformation for adjacent node k
            Tk = create_homogeneous_matrix(Rk, tk)

            # Compute the residual as before
            r_eye = Tj @ np.linalg.inv(Tk)
            r = np.vstack((np.reshape(r_eye[0:3, 0:3] - np.eye(3), (9, 1), order='F'),
                           np.reshape(r_eye[0:3, 3], (3, 1), order='F')))

            residuals.append(r.flatten())

    residuals = np.array(residuals).flatten()
    return residuals

def compute_rotation_matrix_constraints(R):

  # constraints from rotation matrix entries
  c1 = R[0:3,0].reshape((3,1))
  c2 = R[0:3,1].reshape((3,1))
  c3 = R[0:3,2].reshape((3,1))

  # # Jacobian wrt R (1x9), wrt t (1x3)
  r1 = c1.T @ c2
  Jc_r1 = np.hstack((c2.T, c1.T, np.zeros((1,3))))
  Jt_r1 = np.zeros((1,3))

  r2 = c1.T @ c3
  Jc_r2 = np.hstack((c3.T, np.zeros((1,3)), c1.T))
  Jt_r2 = np.zeros((1,3))

  r3 = c2.T @ c3
  Jc_r3 =  np.hstack((np.zeros((1,3)), c3.T, c2.T))
  Jt_r3 = np.zeros((1,3))

  r4 = c1.T @ c1 -1
  Jc_r4 = np.hstack((2*c1.T, np.zeros((1,3)), np.zeros((1,3))))
  Jt_r4 = np.zeros((1,3))

  r5 = c2.T @ c2 -1
  Jc_r5 = np.hstack((np.zeros((1,3)), 2*c2.T, np.zeros((1,3))))
  Jt_r5 = np.zeros((1,3))

  r6 = c3.T @ c3 -1
  Jc_r6 = np.hstack((np.zeros((1,3)), np.zeros((1,3)), 2*c3.T))
  Jt_r6 = np.zeros((1,3))

  # J:= 6x12, r:= 6x1
  J  = np.vstack((np.hstack((Jc_r1, Jt_r1)),
                  np.hstack((Jc_r2, Jt_r2)),
                  np.hstack((Jc_r3, Jt_r3)),
                  np.hstack((Jc_r4, Jt_r4)),
                  np.hstack((Jc_r5, Jt_r5)),
                  np.hstack((Jc_r6, Jt_r6))))

  r = np.vstack((r1,
                 r2,
                 r3,
                 r4,
                 r5,
                 r6))

  return J, r

def create_homogeneous_matrix(R, t):
    """Construct a homogeneous transformation matrix from rotation R and translation t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
