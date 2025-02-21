import numpy as np
import time
from scipy.spatial import distance
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

from scripts.downsample import compute_pc_resolution
from utils import plot_skeleton, save_graph
from point_cloud_clean import clean_point_cloud
from connection_skeleton import connect_pcd, \
    merge_small_clusters, \
    merge_all_clsuter, \
    clean_cluster, \
    connect_pcd_branch, \
    get_point_label, \
    remove_cycle
from sklearn.neighbors import NearestNeighbors
import open3d
import random
import sys
import time
import os
from  scripts.normal_estimation import *

def compute_principal_direction(points):
    if len(points) < 2:
        return np.array([0, 0, 1])  # Default principal direction if not enough points

    # points = np.array(points).reshape(-1, 3)
    # pca = PCA(n_components=3)
    # pca.fit(points)
    # return pca.components_[0]
    points = np.array(points).reshape(-1, 3)

    # Adjust n_components based on the number of points
    n_components = min(3, len(points), points.shape[1])

    pca = PCA(n_components=n_components)
    pca.fit(points)

    # Return the first principal component or a default if n_components is less than 1
    if n_components >= 1:
        return pca.components_[0]
    else:
        return np.array([0, 0, 1])  # Default direction if PCA can't be computed


def compute_leaf_widths(nodes, principal_directions, neighbors):
    """
    Compute the leaf widths for each node using principal directions.

    Parameters:
    nodes: array-like, shape (n_nodes, 3)
        Coordinates of the nodes.
    principal_directions: array-like, shape (n_nodes, 3)
        Principal directions at each node.
    neighbors: list of lists
        Each sublist contains indices of neighboring points for the corresponding node.

    Returns:
    leaf_widths: array-like, shape (n_nodes,)
        Estimated leaf widths for each node.
    """
    n_nodes = len(nodes)
    leaf_widths = np.zeros(n_nodes)

    for i in range(n_nodes):
        node = nodes[i]
        direction = principal_directions[i]
        direction = direction / np.linalg.norm(direction)  # Ensure the direction is a unit vector

        max_distance = 0
        for j in neighbors[i]:
            if j < n_nodes:  # Ensure the neighbor index is within bounds
                neighbor = nodes[j]
                 # Calculate difference vector
                diff = neighbor - node
                # Compute orthogonal projection
                projection = diff - np.dot(diff, direction) * direction
                distance = np.linalg.norm(projection)
                if distance > max_distance:
                    max_distance = distance

        leaf_widths[i] = max_distance

    return leaf_widths

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_local_points(points, centers, h, maxLocalPoints=50000):
    # Get local_points points around this center point
    local_indices = []
    for center in centers:
        x, y, z = center[:3]

        # 1) first get the square around the center
        where_square = ((points[:, 0] >= (x - h)) & (points[:, 0] <= (x + h)) & (points[:, 1] >= (y - h)) &
                        (points[:, 1] <= (y + h)) & (points[:, 2] >= (z - h)) & (points[:, 2] <= (z + h)))

        square = points[where_square]
        indices_square = np.where(where_square == True)[0]

        # Get points which comply to x^2, y^2, z^2 <= r^2
        square_squared = np.square(square - [x, y, z])
        where_sphere = np.sum(square_squared, axis=1) <= h ** 2
        local_sphere_indices = indices_square[where_sphere]

        local_indices.append(local_sphere_indices)

    return local_indices


def get_thetas(r, h):
    """
    matrix of values r
    """
    thetas = np.exp((-r ** 2) / ((h / 2) ** 2))
    # Clip to JUST not zero
    thetas = np.clip(thetas, 10 ** -323, None)
    return thetas


def get_alphas(x, points, h):
    """
    INPUTS:
        x: 1x3 center we of interest, np.ndarray
        points: Nx3 array of all the points, np.ndarray
        h: size of local neighboorhood, float
    """
    r = np.linalg.norm(x - points, axis=1) + 10 ** -10
    theta = get_thetas(r, h)

    alphas = theta / r
    return alphas


def get_betas(x, points, h):
    """
    INPUTS:
        x: 1x3 center we of interest, np.ndarray
        points: Nx3 array of all the points, np.ndarray
        h: size of local neighboorhood, float
    """
    r = np.linalg.norm(x - points, axis=1) + 10 ** -10
    theta = get_thetas(r, h)

    betas = theta / r ** 2

    return np.array(betas)


def get_density_weights(points, h0, for_center=False, center=[0, 0, 0]):
    """
    INPUTS:
        x: 1x3 center we of interest, np.ndarray
        points: Nx3 array of all the points, np.ndarray
        h: size of local neighboorhood, float
    RETURNS:
        - np.array Nx1 of density weights assoiscated to each point
    """
    density_weights = []

    if for_center:
        r = points - center
        r2 = np.einsum('ij,ij->i', r, r)
        density_weights = np.einsum('i->', np.exp((-r2) / ((h0 / 2) ** 2)))
    else:

        for point in points:
            r = point - points
            r2 = np.einsum('ij,ij->i', r, r)
            # This calculation includes the point itself thus one entry will be zero resultig in the needed + 1 in formula dj = 1+ sum(theta(p_i - p_j))
            density_weight = np.einsum('i->', np.exp((-r2) / ((h0 / 2) ** 2)))
            density_weights.append(density_weight)

    return np.array(density_weights)


def get_term1(center, points, h, density_weights):
    """
    INPUTS:
        center: 1x3 center we of interest, np.ndarray
        points: Nx3 array of all the points, np.ndarray
        h: size of local neighboorhood, float
        h0: size of first local neighboorhood, float
    RETURNS:
        - term1 of the equation as float
    """
    t1_t = time.perf_counter()

    r = points - center[:3]
    r2 = np.einsum('ij,ij->i', r, r)

    thetas = np.exp(-r2 / ((h / 2) ** 2))
    # Clip to JUST not zero
    # thetas =  np.clip(thetas, 10**-323, None)

    # DIFFERS FROM PAPER
    # r_norm = np.sqrt(r_norm, axis = 1)
    # alphas = thetas/r_norm

    alphas = thetas / density_weights

    denom = np.einsum('i->', alphas)
    if denom > 10 ** -20:
        # term1 = np.sum((points.T*alphas).T, axis = 0)/denom
        term1 = np.einsum('j,jk->k', alphas, points) / denom
    else:
        term1 = np.array(False)

    t2_t = time.perf_counter()
    tt = round(t2_t - t1_t, 5)

    return term1, tt


def get_term2(center, centers, h):
    """
    INPUTS:
        center: 1x3 center we of interest, np.ndarray
        centers: Nx3 array of all the centers (excluding the current center), np.ndarray
        h: size of local neighboorhood, float
    RETURNS:
        - term2 of the equation as float
    """
    t1 = time.perf_counter()

    x = center[:3] - centers[:, :3]
    r2 = np.einsum('ij,ij->i', x, x)
    r = 1 / np.sqrt(r2)
    # r3 = np.sum(r**1.2, axis = 1)
    thetas = np.exp((-r2) / ((h / 2) ** 2))

    # r_norm = np.linalg.norm(r,axis = 1)
    # DIFFERS FROM PAPER
    # betas =np.einsum('i,i->i', thetas, density_weights)# / r2
    betas = np.einsum('i,i->i', thetas, r)

    denom = np.einsum('i->', betas)

    if denom > 10 ** -20:
        num = np.einsum('j,jk->k', betas, x)

        term2 = num / denom
    else:
        term2 = np.array(False)

    t2 = time.perf_counter()
    tt = round(t2 - t1, 4)
    return term2, tt


def get_sigma(center, centers, h):
    # These are the weights
    r = centers - center
    r2 = np.einsum('ij,ij->i', r, r)
    thetas = np.exp((-r2) / ((h / 2) ** 2))
    cov = np.einsum('j,jk,jl->kl', thetas, r, r)
    # Get eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov)

    if np.iscomplex(values).any():
        values = np.real(values)
        vectors = np.real(vectors)
        vectors_norm = np.sqrt(np.einsum('ij,ij->i', vectors, vectors))
        vectors = vectors / vectors_norm

    # Argsort always works from low --> to high so taking the negative values will give us high --> low indices
    sorted_indices = np.argsort(-values)

    values_sorted = values[sorted_indices]
    vectors_sorted = vectors[:, sorted_indices]

    sigma = values_sorted[0] / np.sum(values_sorted)

    return sigma, vectors_sorted, 0.0


def get_h0(points):
    x_max = points[:, 0].max()
    x_min = points[:, 0].min()

    y_max = points[:, 1].max()
    y_min = points[:, 1].min()

    z_max = points[:, 2].max()
    z_min = points[:, 2].min()

    diagonal = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) ** .5

    Npoints = len(points)

    return 2 * diagonal / (Npoints ** (1. / 3))


class myCenter:
    def __init__(self, center, h, index):
        self.center = center
        self.h = h
        self.label = "non_branch_point"
        self.index = index
        self.connections = []
        self.bridge_connections = None
        self.closest_neighbours = np.array([])
        self.head_tail = False
        self.branch_number = None
        self.eigen_vectors = np.zeros((3, 3))
        self.sigma = 0.5

    def set_non_branch(self):
        if self.label != 'branch_point' and self.label != 'removed':
            self.set_label('non_branch_point')
            self.connections = []
            self.bridge_connections = None
            self.head_tail = False
            self.branch_number = None

    def set_as_bridge_point(self, key, connection):
        if self.label != 'removed':
            self.set_non_branch()
            self.set_label('bridge_point')
            self.bridge_connections = connection
            self.branch_number = key

    def set_as_branch_point(self, key):

        self.connections = []
        self.bridge_connections = None
        self.head_tail = False
        self.branch_number = None
        self.label = 'branch_point'
        self.branch_number = key

    def set_eigen_vectors(self, eigen_vectors):
        if self.label == "non_branch_point":
            self.eigen_vectors = eigen_vectors

    def set_sigma(self, sigma):
        if self.label != "branch_point":
            self.sigma = sigma

    def set_closest_neighbours(self, closest_neighbours):
        """
        """
        self.closest_neighbours = closest_neighbours

    def set_label(self, label):
        if self.label != 'removed':
            self.label = label

    def set_center(self, center):

        if self.label != "branch_point":
            self.center = center

    def set_h(self, h):
        if self.label != "branch_point":
            self.h = h


class myCenters:

    def set_my_non_branch_centers(self):

        my_non_branch_centers = []

        for center in self.myCenters:
            if center.label == 'non_branch_point' or center.label == 'bridge_point':
                my_non_branch_centers.append(center)
        self.my_non_branch_centers = my_non_branch_centers

    def get_nearest_neighbours(self):

        distances = distance.squareform(distance.pdist(self.centers[:, :3]))
        self.closest = np.argsort(distances, axis=1)

        for center in self.myCenters:
            # center.set_closest_neighbours(self.closest[center.index,1:])
            closest = self.closest[center.index, :].copy()
            sorted_local_distances = distances[center.index, closest] ** 2

            # Returns zero if ALL values are within the range
            in_neighboorhood = np.argmax(sorted_local_distances >= (center.h) ** 2)
            if in_neighboorhood == 0:
                in_neighboorhood = -1

            center.set_closest_neighbours(closest[1:in_neighboorhood])

    def __init__(self, centers, h0, maxPoints):
        self.centers = centers + 10 ** -20  # Making sure centers are never the same as the actual points which can lead to bad things
        self.myCenters = []
        self.my_non_branch_centers = []
        index = 0
        for center in centers:
            self.myCenters.append(myCenter(center, h0, index))
            index += 1
        self.skeleton = {}
        self.closest = []
        self.sigmas = np.array([None] * len(centers))
        self.h0 = h0
        self.h = h0
        self.eigen_vectors = [None] * len(centers)
        self.branch_points = [None] * len(centers)
        self.non_branch_points = [None] * len(centers)
        self.maxPoints = maxPoints
        self.get_nearest_neighbours()
        self.set_my_non_branch_centers()
        self.Nremoved = 0

        # From the official code
        self.search_distance = 4
        self.too_close_threshold = 0.01
        self.allowed_branch_length = 5

    def remove_centers(self, indices):
        """
        Removes a center completely
        """
        if not isinstance(indices, list):
            indices = list([indices])

        for index in sorted(indices, reverse=True):
            center = self.myCenters[index]
            center.set_label("removed")
            self.centers[center.index] = [9999, 9999, 9999]
        self.set_my_non_branch_centers()
        self.Nremoved += len(indices)

    def get_non_branch_points(self):

        non_branch_points = []
        for center in self.myCenters:
            if center.label != "branch_point" and center.label != "removed":
                non_branch_points.append(center.index)

        return non_branch_points

    def get_bridge_points(self):

        bridge_points = []
        for key in self.skeleton:
            head = self.skeleton[key]['head_bridge_connection']
            tail = self.skeleton[key]['tail_bridge_connection']

            if head[0] and head[1] != None:
                if not head[1] in bridge_points:
                    bridge_points.append(head[1])
            if tail[0] and tail[1] != None:
                if not tail[1] in bridge_points:
                    bridge_points.append(tail[1])

        return bridge_points

    def update_sigmas(self):

        k = 5

        new_sigmas = []

        for center in self.my_non_branch_centers:
            index = center.index

            indices = np.array(self.closest[index, :k]).astype(int)

            sigma_nearest_k_neighbours = self.sigmas[indices]

            mean_sigma = np.mean(sigma_nearest_k_neighbours)

            new_sigmas.append(mean_sigma)

        index = 0
        for center in self.my_non_branch_centers:
            center.set_sigma(new_sigmas[index])

            self.sigmas[center.index] = new_sigmas[index]
            index += 1

    def update_properties(self):

        self.set_my_non_branch_centers()

        for center in self.myCenters:
            index = center.index
            self.centers[index] = center.center
            self.eigen_vectors[index] = center.eigen_vectors
            self.sigmas[index] = center.sigma

        self.get_nearest_neighbours()
        self.update_sigmas()

    def update_labels_connections(self):
        """
        Update all the labels of all the centers
            1) goes through all the branches and checks if the head has a bridge connection or a branch connection
                - If bridge connection this is still the head/tail of the branch
                - If it has a branch connection it is simply connected to another branch --> It is no head/tail anymore
            2) Checks if bridges are still bridges
            3) Sets all other points to simple non_branch_points
        """

        updated_centers = []
        for key in self.skeleton:

            branch = self.skeleton[key]

            head = self.myCenters[branch['branch'][0]];
            tail = self.myCenters[branch['branch'][-1]]

            # This is either a None value (for not having found a bridge point / connected branch) or this is an integer index
            head_connection = branch['head_bridge_connection'][1]
            tail_connection = branch['tail_bridge_connection'][1]

            if head_connection != None:

                head_connection = self.myCenters[head_connection]

                if branch['head_bridge_connection'][0]:
                    head_connection.set_as_bridge_point(key, head.index)
                    head.head_tail = True
                else:
                    head_connection.set_as_branch_point(key)
                    head.head_tail = False

                head.set_as_branch_point(key)
                head.connections = [head_connection.index, branch['branch'][1]]

                updated_centers.append(head_connection.index)
                updated_centers.append(head.index)
            else:
                head.set_as_branch_point(key)
                head.head_tail = True

            if tail_connection != None:

                tail_connection = self.myCenters[tail_connection]

                if branch['tail_bridge_connection'][0]:
                    tail.head_tail = True
                    tail_connection.set_as_bridge_point(key, tail.index)
                else:
                    tail.head_tail = False
                    tail_connection.set_as_branch_point(key)

                tail.set_as_branch_point(key)
                tail.connections = [tail_connection.index, branch['branch'][-2]]
                updated_centers.append(tail_connection.index)
                updated_centers.append(tail.index)
            else:
                tail.set_as_branch_point(key)
                tail.head_tail = True

            # 1) Go through the branch list and set each center t branch_point and set the head_tail value appropriately
            # 2) Set the connections
            index = 1
            for center in branch['branch'][1:-1]:
                center = self.myCenters[center]

                center.set_as_branch_point(key)

                center.connections.append(branch['branch'][index - 1])
                center.connections.append(branch['branch'][index + 1])
                center.head_tail = False

                updated_centers.append(center.index)
                index += 1

        for center in self.myCenters:

            if center.index in updated_centers:
                continue
            center.set_non_branch()

        for key in self.skeleton:
            branch = self.skeleton[key]

            for index in branch['branch']:
                if branch['branch'].count(index) > 1:
                    print("ERROR: This branch has multiple counts of 1 index...", branch['branch'])
                    break

    def contract(self, points, local_indices, h, density_weights, mu=0.35):
        """
        Updates the centers by the algorithm suggested in "L1-medial skeleton of Point Cloud 2010"

        INPUT:
            - Centers
            - points belonging to centers
            - local neighbourhood h0
            - mu factor for force between centers (preventing them from clustering)
        OUTPUT:
            - New centers
            - Sigmas (indicator for the strength of dominant direction)
            - The eigenvectors of the points belonging to the centers
        """
        self.h = h

        t1_total = time.perf_counter();
        term1_t = 0;
        term2_t = 0;
        sigma_t = 0
        t_pre = 0;
        t_post = 0

        error_center = 0;
        N = 0;
        for myCenter in self.myCenters:

            t1 = time.perf_counter()
            # Get the closest 50 centers to do calculations with
            centers_indices = myCenter.closest_neighbours
            # Get the density weight of these centers
            centers_in = np.array(self.centers[centers_indices])

            my_local_indices = local_indices[myCenter.index]
            local_points = points[my_local_indices]

            t2 = time.perf_counter()
            t_pre += t2 - t1
            # Check if we have enough points and centers
            shape = local_points.shape
            if len(shape) == 1:
                continue
            elif shape[0] > 2 and len(centers_in) > 1:

                density_weights_points = density_weights[my_local_indices]

                term1, delta_t1 = get_term1(myCenter.center, local_points, h, density_weights_points)

                term2, delta_t2 = get_term2(myCenter.center, centers_in, h)

                term1_t += delta_t1;
                term2_t += delta_t2

                if term1.any() and term2.any():
                    sigma, vecs, delta_ts = get_sigma(myCenter.center, centers_in, h)
                    # sigma = np.clip(sigma, 0 ,1.)
                    sigma_t += delta_ts

                    # DIFFERS FROM PAPER
                    # mu = mu_length/sigma_length * (sigma - min_sigma)
                    # if mu < 0:
                    #     continue

                    # mu_average +=mu

                    t1 = time.perf_counter()
                    new_center = term1 + mu * sigma * term2

                    error_center += np.linalg.norm(myCenter.center[:3] - new_center[:3]);
                    N += 1

                    # Update this center object
                    #new_center normal
                    new_center = np.hstack((new_center, myCenter.center[3:]))
                    myCenter.set_center(new_center)
                    myCenter.set_eigen_vectors(vecs)
                    myCenter.set_sigma(sigma)
                    myCenter.set_h(h)
                    t2 = time.perf_counter()

                    t_post += t2 - t1
        t2_total = time.perf_counter();
        total_time = round(t2_total - t1_total, 4);

        if N == 0: N += 1

        # CURSOR_UP_ONE = '\x1b[1A'; ERASE_LINE = '\x1b[2K'
        # first_line = CURSOR_UP_ONE + ERASE_LINE + "\tTotal Contract time = {} secs, Average movement of centers={}          \n".format(total_time, round(error_center/N,6))
        # second_line = "\tTime per step: prep={} secs, term1={} secs, term2={} secs, sigma={} secs, post = {} secs           \r".format(round(t_pre,4),round(term1_t,4), round(term2_t,4), round(sigma_t,4), round(t_post,4))

        # sys.stdout.write(first_line)
        # sys.stdout.write(second_line)
        # sys.stdout.flush()

        return error_center / N

    def bridge_2_branch(self, bridge_point, requesting_branch_number):
        """
        Change a bridge to a branch.

        1) finds a branch with this bridge_point
        2) changes the boolean indicating bridge/branch to False
        3) Changes the head/tail label of the head/tail of this branch
        4) When the whole skeleton is checked it changes the bridge_label to branch_label
        """

        for key in self.skeleton:
            head_bridge_connection = self.skeleton[key]['head_bridge_connection']
            tail_bridge_connection = self.skeleton[key]['tail_bridge_connection']

            # 1)
            if bridge_point == head_bridge_connection[1]:
                # 2)
                self.skeleton[key]['head_bridge_connection'][0] = False
                # 3)
                head = self.skeleton[key]['branch'][0]
                self.myCenters[head].head_tail = False

            if bridge_point == tail_bridge_connection[1]:
                self.skeleton[key]['tail_bridge_connection'][0] = False
                tail = self.skeleton[key]['branch'][-1]
                self.myCenters[tail].head_tail = False

        # 4)
        self.myCenters[bridge_point].set_as_branch_point(requesting_branch_number)

    def find_bridge_point(self, index, connection_vector):
        """
        Finds the bridging points of a branch
        These briding points are used to couple different branches at places where we have conjunctions
        INPUT:v
            - Index of the tail/head of the branch
            - the vector connecting this head/tail point to the branch
        OUTPUT:
            - If bridge_point found:
                index of bridge_point
            else:
                none
        ACTIONS:
            1) find points in the neighboorhood of this point
            2) Check if they are non_branching_points (i.e. not already in a branch)
            3) Are they 'close'? We defined close as 5*(distance_to_closest_neighbour)
            5) Angle of line end_of_branch to point and connection_vector < 90?
            6) return branch_point_index
        """

        myCenter = self.myCenters[index]

        success = False
        bridge_point = None
        for neighbour in myCenter.closest_neighbours:

            neighbour = self.myCenters[neighbour]

            if neighbour.label == "branch_point" or neighbour.label == 'removed':
                continue

            # If current neighbour is too far away we break
            if sum((neighbour.center - myCenter.center) ** 2) > self.h ** 2:
                break

            branch_2_bridge_u = unit_vector(neighbour.center - myCenter.center)
            connection_vector_u = unit_vector(connection_vector)

            cos_theta = np.dot(branch_2_bridge_u, connection_vector_u)
            # cos_theta >0 --> theta < 100 degrees
            if cos_theta >= 0:
                bridge_point = neighbour.index
                success = True
                break

        return bridge_point, success

    def connect_bridge_points_in_h(self):
        # Connects bridge points which are within the same neighboorhood
        for center in self.myCenters:
            if center.label != 'bridge_point':
                continue
            # Check the local neighboorhood for any other bridge_points
            for neighbour in center.closest_neighbours:

                neighbour = self.myCenters[neighbour]

                # Is it a bridge point?
                if neighbour.label != 'bridge_point':
                    continue

                # Is it still in the local neighboorhood?
                if sum((neighbour.center - center.center) ** 2) > (2 * center.h) ** 2:
                    break

                # If here we have two bridge points in 1 local nneighboorhood:
                # So we merge them:
                branch1 = center.branch_number;
                branch2 = neighbour.branch_number;

                # Check if we are connected to the head or tail of the branch
                if self.skeleton[branch1]['head_bridge_connection'][1] == center.index:
                    index_branch1_connection = 0
                elif self.skeleton[branch1]['tail_bridge_connection'][1] == center.index:
                    index_branch1_connection = -1
                else:
                    raise Exception(
                        "ERROR in 'merge_bridge_points': COULDNT FIND THE BRIDGE INDEX IN THE BRIDGE_CONNECTIONS OF THE SPECIFIED BRANCH")
                if self.skeleton[branch2]['head_bridge_connection'][1] == neighbour.index:
                    index_branch2_connection = 0
                elif self.skeleton[branch2]['tail_bridge_connection'][1] == neighbour.index:
                    index_branch2_connection = -1
                else:
                    raise Exception(
                        "ERROR in 'merge_bridge_points': COULDNT FIND THE BRIDGE INDEX IN THE BRIDGE_CONNECTIONS OF THE SPECIFIED BRANCH")

                # Change the conenctions and boolenas accordingly:
                if index_branch1_connection == 0:
                    # Add the bridge point to the branch
                    self.skeleton[branch1]['branch'].insert(0, center.index)
                    # Update the head_conenction such that it does not have any bridge connection anymore
                    self.skeleton[branch1]['head_bridge_connection'][0] = False
                    # And connect it to the otehr branch, i.e. the neighboor
                    self.skeleton[branch1]['head_bridge_connection'][1] = neighbour.index
                else:
                    self.skeleton[branch1]['branch'].extend([center.index])
                    self.skeleton[branch1]['tail_bridge_connection'][0] = False
                    self.skeleton[branch1]['tail_bridge_connection'][1] = neighbour.index

                if index_branch2_connection == 0:
                    self.skeleton[branch2]['branch'].insert(0, neighbour.index)
                    self.skeleton[branch2]['head_bridge_connection'][0] = False
                    self.skeleton[branch2]['head_bridge_connection'][1] = center.index
                else:
                    self.skeleton[branch2]['branch'].extend([neighbour.index])
                    self.skeleton[branch2]['tail_bridge_connection'][0] = False
                    self.skeleton[branch2]['tail_bridge_connection'][1] = center.index

                # Now they are branch points:
                center.set_as_branch_point(branch1)
                neighbour.set_as_branch_point(branch2)

    def connect_identical_bridge_points(self):
        """
        Connectes branches which are connected to an identical bridge point
        1) Makes a list with the connection values of all the heads and tails. The value is None if it is connected to another branch
        2) Finds a similar index
        3) Connects these branches
        4) Replaces the value by None in the list and start at (2) again
        """

        # 1)
        bridge_points = []
        for key in self.skeleton:
            branch = self.skeleton[key]

            bridges_of_branch = []
            if branch['head_bridge_connection'][0]:
                bridges_of_branch.append(branch['head_bridge_connection'][1])
            else:
                bridges_of_branch.append(None)
            if branch['tail_bridge_connection'][0]:
                bridges_of_branch.append(branch['tail_bridge_connection'][1])
            else:
                bridges_of_branch.append(None)

            bridge_points.append(bridges_of_branch)

        bridge_points = np.array(bridge_points)
        success = True
        while success:
            success = False
            for points in bridge_points:
                bridge_head = points[0]
                bridge_tail = points[1]

                # If not None check how man y instances we ahve of this bridge point
                if bridge_head != None:
                    # 2)
                    count_head = len(np.argwhere(bridge_points == bridge_head))
                    if count_head > 1:
                        # 3) #If mroe then 1 we get all the indices (row, column wise) where the rows are branch numbers and the columns indicate if its at the head or tail
                        indices = np.where(bridge_points == bridge_head)
                        # We choose the first banch as the 'parent'  it will adopt this bridge_point as branch point. All other branches with this bridge_point will simply loose it.
                        branch1 = indices[0][0]
                        # Set these values to False as after this we do not have a bridge point anymore
                        # self.skeleton[branch1]['head_bridge_connection'][0] = False
                        # self.skeleton[branch1]['head_bridge_connection'][1] = bridge_head
                        # Sets all branches with this bridge_point to False as well
                        self.bridge_2_branch(bridge_head, branch1)
                        # 4) Set all the indices with this bridge_point to None and start over
                        bridge_points[indices] = None
                        success = True
                        break

                if bridge_tail != None:
                    count_tail = len(np.argwhere(bridge_points == bridge_tail))
                    if count_tail > 1:
                        indices = np.where(bridge_points == bridge_tail)
                        branch1 = indices[0][0]  # Becomes part of the branch
                        # self.skeleton[branch1]['tail_bridge_connection'][0] = False
                        # self.skeleton[branch1]['tail_bridge_connection'][1] = bridge_tail

                        self.bridge_2_branch(bridge_tail, branch1)
                        bridge_points[indices] = None
                        success = True
                        break

    def merge_bridge_points(self):
        """
        1) Connects bridge points which are within the same neighboorhood
        2) Connectes branches which are connected to an identical bridge point
        """

        # 1)
        self.connect_bridge_points_in_h()
        # 2)
        self.connect_identical_bridge_points()

    def set_bridge_points(self, key, branch):
        """
        First finds then sets bridge_points of this branch
            1) checks if head/tail is connected to a branch
            2) Checks if we can find a bridge point
            3) If we find bridge, set the old bridge(if we had it) to non_branch_point and set new bridge label to bridge_point and update the branch
        """

        # 1)
        if branch['head_bridge_connection'][0]:
            head = branch['branch'][0]
            head_1 = branch['branch'][1]
            head_bridge_vector = self.centers[head] - self.centers[head_1]
            # 2)
            bridge_point, success = self.find_bridge_point(head, head_bridge_vector)
            # 3) Update old bridge_point
            if success:
                old_bridge_point = branch['head_bridge_connection'][1]
                if old_bridge_point != None:
                    old_bridge_point = self.myCenters[old_bridge_point]
                    old_bridge_point.set_non_branch()

                branch['head_bridge_connection'][1] = bridge_point
                self.myCenters[bridge_point].set_as_bridge_point(key, head)

        if branch['tail_bridge_connection'][0]:
            tail = branch['branch'][-1]
            tail_1 = branch['branch'][-2]
            tail_bridge_vector = self.centers[tail] - self.centers[tail_1]

            bridge_point, success = self.find_bridge_point(tail, tail_bridge_vector)

            if success:
                # Update old bridge_point
                old_bridge_point = branch['tail_bridge_connection'][1]
                if old_bridge_point != None:
                    old_bridge_point = self.myCenters[old_bridge_point]
                    old_bridge_point.set_non_branch()

                branch['tail_bridge_connection'][1] = bridge_point
                self.myCenters[bridge_point].set_as_bridge_point(key, tail)

        self.skeleton[key] = branch

    def add_new_branch(self, branch_list):
        """
        A branch: {'branch': [list of branch points], 'head connection':[Bool denoting if its a bridge/branch point True/False, index of conenction], tail_bridge_connection:[same stuff]}
        For each new branch a few checks:
            1) were there bridge points? If so they need to be connected
                - If they are and the head / tail of the branch this mean the branch is connected to another branch
            2) Finds the potential bridge points
            3) sets the labels of the centers
            4) adds the branch to the skeleon list of branches
        """
        head_bridge_connection = [True, None]
        tail_bridge_connection = [True, None]

        key = len(self.skeleton) + 1

        # Check for bridge points:
        for index in branch_list:

            center = self.myCenters[index]
            # Do we have a bridge point?
            if center.label != 'bridge_point':
                continue

            # Our head is connected to a bridge point of another branch. Thus our head has NO bridge point and we need to change this in the branch from which this is the bridge_point
            if index == branch_list[0]:
                head_bridge_connection[0] = False
                head_bridge_connection[1] = center.bridge_connections

            # same stuff
            elif index == branch_list[-1]:
                tail_bridge_connection[0] = False
                tail_bridge_connection[1] = center.bridge_connections

            # Now make this bridge_point a branch
            self.bridge_2_branch(center.index, requesting_branch_number=key)

        branch = {'branch': branch_list, 'head_bridge_connection': head_bridge_connection,
                  'tail_bridge_connection': tail_bridge_connection}

        # Set labels
        for index in branch_list:

            self.myCenters[index].set_as_branch_point(key)

            if (index == branch_list[0] and head_bridge_connection[0]) or (
                    index == branch_list[-1] and tail_bridge_connection[0]):
                self.myCenters[index].head_tail = True
            else:
                self.myCenters[index].head_tail = False

        self.skeleton[key] = branch

    def update_branch(self, key, new_branch):
        """
        Checks wheter the updated branch contains a bridge point of another branch. If so it updates the label of the bridge point and the branch head/tail connection values
        INPUTS:
            - key of the branch
            - the new branch
        """

        # Go through the new_branch list
        for index in [new_branch['branch'][0], new_branch['branch'][-1]]:
            # Check if this point is a bridge_point not from this branch
            center = self.myCenters[index]
            # Set the label of this center to branch_point
            center.set_as_branch_point(key)

            # Set head/tail label
            if index == new_branch['branch'][0] and new_branch['head_bridge_connection'][0]:
                center.head_tail = True
            elif index == new_branch['branch'][-1] and new_branch['tail_bridge_connection'][0]:
                center.head_tail = True
            else:
                center.head_tail = False

        # Actually update branch
        self.skeleton[key] = new_branch

    def find_extension_point(self, center_index, vector):
        """
        INPUT:
            - The neighbours
            - The center which this is about
            - the connection vector of this center to the skeleton
        OUTPUT:
            - Boolean indicating connection yes or
            - index of connection
        ACTIONS:
            1) Check if neighbour is too far
            2) Check if too close
            3) check if in he right direcion
            4) if checks 1,2,3 we check if we meet the requirement. Then we stop
        """
        myCenter = self.myCenters[center_index]
        vector_u = unit_vector(vector)

        connection = False
        for neighbour in myCenter.closest_neighbours:

            neighbour = self.myCenters[neighbour]
            if neighbour.label == 'branch_point' or neighbour.label == 'removed':
                continue

            # #Check if inside local neighbourhood
            r = neighbour.center - myCenter.center
            r2 = np.einsum('i,i->i', r, r)
            r2_sum = np.einsum('i->', r2)

            # 1)
            if r2_sum > (self.search_distance) ** 2:
                break
            # 2)
            elif r2_sum <= (self.too_close_threshold) ** 2:
                self.remove_centers(neighbour.index)
                continue

            # make unit vector:
            center_2_neighbour_u = unit_vector(
                neighbour.center - myCenter.center)  # From front of skeleton TOWARDS the new direction

            # Check skeleton angle condition
            # cos(theta) = dot(u,v)/(norm(u)*norm(v)) <= -0.9
            cos_theta = np.dot(center_2_neighbour_u, vector_u)

            # 3)
            if cos_theta > 0:
                continue

            # 4)
            if cos_theta <= -0.9:
                connection = True
            break
        return connection, neighbour.index

    def try_extend_branch(self, branch, head_bridge_connection=True, tail_bridge_connection=True):
        """
        Tries to extend this branch from the head and the tail onwards.

        INPUTS:
            - the branch as list of idnices
            - head/tail conenction boolean indicating if the head/tail is connected to a bridge point (T) or branch point (F)
        OUTPUTS:
            - The extended branch ( as far as possible)
            - Boolean indicating if a branch was etended in any way
        """

        found_connection = False

        # head =! tail --> which mean skeleton is a full circle AND head is nto conencted to another branch
        if head_bridge_connection:
            # Get index of head connection
            head = branch[0]
            # Get vector conencted head to the rest of the skeleton
            head_bridge_connection_vector = self.centers[branch[1]] - self.centers[head]

            # find a possibleextensions of this connection
            connection, index = self.find_extension_point(head, head_bridge_connection_vector)
            # Inserts it
            if connection:
                if not connection == branch[-1]:
                    branch.insert(0, index)
                    found_connection = True

        if tail_bridge_connection:
            tail = branch[-1]
            tail_bridge_connection_vector = self.centers[branch[-2]] - self.centers[tail]

            connection, index = self.find_extension_point(tail, tail_bridge_connection_vector)
            if connection:
                if not connection == branch[0]:
                    branch.extend([index])
                    found_connection = True

        return branch, found_connection

    def try_to_make_new_branch(self, myCenter):
        """
        Tries to form a new branch
        """

        found_branch = False
        for neighbour in myCenter.closest_neighbours:
            neighbour_center = self.centers[neighbour]
            # Check if inside local neighbourhood
            if sum((neighbour_center - myCenter.center) ** 2) > (self.search_distance) ** 2:
                break

            center_2_neighbour_u = unit_vector(neighbour_center - myCenter.center)

            # Check if this neighbour is in the direction of the dominant eigen_vector:
            # So is the angle  > 155 or < 25

            if abs(np.dot(myCenter.eigen_vectors[:, 0], center_2_neighbour_u)) < 0.9:
                continue

            branch = [myCenter.index, neighbour]
            found_connection = True
            while found_connection:
                branch, found_connection = self.try_extend_branch(branch)

            # We ackknowledge new branches if they exist of 5 or more centers:
            branch_length = self.allowed_branch_length * int(self.h / self.h0)
            if len(branch) > 5:
                # If inside this branch are centers which were bridge points we will connect them up
                self.add_new_branch(branch)
                found_branch = True
                break

        return found_branch

    def try_extend_skeleton(self):
        """
        Tries to extend each already existing branch from the head and the tail onwards
         - If other bridge points are encountered the branch will stop extending this way and connect to the branch of this aprticular bridge point
        """

        for key in self.skeleton:

            branch = self.skeleton[key]
            branch_list = branch['branch']

            success = True
            had_one_success = False
            # Tries to extend the head and tail by one center
            while success:

                # Go through the new_branch list, skip the already known indices
                branch_list, success = self.try_extend_branch(branch_list, branch['head_bridge_connection'][0],
                                                              branch['tail_bridge_connection'][0])

                # If newly found tail/head are a bridge point from another branch we stop extending this way
                if success:
                    new_head = self.myCenters[branch_list[0]]
                    new_tail = self.myCenters[branch_list[-1]]

                    # If we encounter a bridge from a different branch we are connected to this branch and thus do not have a bridge point anymore and need to adjust that particular branch as well
                    if new_head.label == 'bridge_point' and new_head.index != branch['head_bridge_connection'][1]:
                        # Update this branch head bridge connection
                        branch['head_bridge_connection'][0] = False
                        branch['head_bridge_connection'][1] = new_head.bridge_connections
                        # Update the branch from which this bridge_point originated
                        self.bridge_2_branch(new_head.index, key)

                    elif new_tail.label == 'bridge_point' and new_tail.index != branch['tail_bridge_connection'][1]:

                        branch['tail_bridge_connection'][0] = False
                        branch['tail_bridge_connection'][1] = new_tail.bridge_connections
                        # Update the branch from which this bridge_point originated
                        self.bridge_2_branch(new_tail.index, key)

                # If we extended the branch we update it.
                if success:
                    had_one_success = True
                    branch['branch'] = branch_list
                    self.update_branch(key, branch)

            self.set_bridge_points(key, branch)
            self.merge_bridge_points()
            self.clean_points_around_branch(branch)

    def find_connections(self):
        """
        1) Tries to extend the existing skeleton:
        2) Tries to find new skeletons
        3) Merges skeletons if possible
        """

        # self.try_extend_skeleton()

        non_branch_points = np.array(self.get_non_branch_points())
        if non_branch_points.any():
            # non_branch_points = non_branch_points)
            sigma_candidates = self.sigmas[non_branch_points]

            seed_points_to_check = non_branch_points[np.where(sigma_candidates > 0.9)]

            # bridge_points = self.get_bridge_points()

            # seed_points_to_check = list(bridge_points) + list(seed_points_to_check)
            # print("top 5 sigmas:", sorted(sigma_candidates, reverse =True)[:5])
            for seed_point_index in seed_points_to_check:

                myCenter = self.myCenters[seed_point_index]

                old_skeleton = len(self.skeleton)
                succes = self.try_to_make_new_branch(myCenter)
                if succes:
                    new_branch_number = len(self.skeleton)
                    new_branch = self.skeleton[new_branch_number]
                    self.set_bridge_points(new_branch_number, new_branch)
                    self.merge_bridge_points()
                    self.clean_points_around_branch(new_branch)

                    # input("adding new branch...")

            self.update_labels_connections()
            self.clean_points()

    def clean_points(self):
        """
        Cleans points which
        1) have no poins in their neighboorhood
        2) Are bridge_points, with no non_branch_points around. Makes them part of the branch
        3) More then half your neighbours are branch_points
        AFTER the first 2 branches are formed
        """

        if len(self.skeleton) > 1:
            remove_centers = []
            for center in self.myCenters:
                if center.label == 'removed' or center.label == 'branch_point':
                    continue

                if center.closest_neighbours.any():
                    neighbour = center.closest_neighbours[0]
                    neighbour = self.myCenters[neighbour]
                # 1) If no neighbours:
                else:
                    # If a bridge point we make it a branch
                    if center.label == 'bridge_point':
                        self.bridge_2_branch(center.index, center.branch_number)
                    elif center.label == 'non_branch_point':
                        remove_centers.append(center.index)
                    # Skip the other checks
                    continue
                # 2)
                if center.label == 'bridge_point':
                    has_non_branch_point_neighbours = False
                    # Check if all close neighbours are branch_points:
                    for neighbour in center.closest_neighbours:
                        neighbour = self.myCenters[neighbour]

                        # Check till we leave the neighboorhood
                        if sum((neighbour.center - center.center) ** 2) > (2 * center.h) ** 2:
                            break

                        if neighbour.label != 'branch_point' and neighbour.label != 'removed':
                            has_non_branch_point_neighbours = True
                            break
                    if not has_non_branch_point_neighbours:
                        self.bridge_2_branch(center.index, center.branch_number)

                # 3)
                if center.label == 'non_branch_point':
                    N_branch_points = 0
                    N_non_branch_points = 0
                    # Check if all close neighbours are branch_points:
                    for neighbour in center.closest_neighbours:
                        neighbour = self.myCenters[neighbour]

                        # Check till we leave the neighboorhood
                        if np.sum((neighbour.center - center.center) ** 2) > (center.h) ** 2:
                            break

                        if neighbour.label == 'branch_point':
                            N_branch_points += 1
                        elif neighbour.label == 'non_branch_point' or neighbour.label == 'bridge_point':
                            N_non_branch_points += 1

                    if N_branch_points > N_non_branch_points:
                        remove_centers.append(center.index)

            # Remove all the centers
            if remove_centers:
                self.remove_centers(remove_centers)
                print("removed", len(remove_centers), 'points!')

    def clean_points_around_branch(self, branch):
        """
        Removes points which:
            1) are within h of any point in the branch_list
            2) Excludes the head and tail IF they are connected to a bridge
            3) are non_branch_points

        """

        remove_centers = []
        for center in branch['branch']:

            # 2) If the head/tail is connected to a bridge do not remove any points
            if center == branch['branch'][0] and branch['head_bridge_connection'][0]:
                continue
            elif center == branch['branch'][-1] and branch['tail_bridge_connection'][0]:
                continue

            center = self.myCenters[center]

            for neighbour in center.closest_neighbours:

                neighbour = self.myCenters[neighbour]

                if sum((neighbour.center - center.center) ** 2) > self.too_close_threshold ** 2:
                    break

                if neighbour.label != 'non_branch_point':
                    continue

                remove_centers.append(neighbour.index)

        if remove_centers:
            self.remove_centers(remove_centers)
            print("removed", len(remove_centers), 'points!')


def get_skeleton_pcd(pcd, voxel_size, try_make_skeleton=False, plot_iter_pcd=False, iter=3, verbose=False):
    h0 = get_h0(np.asarray(pcd.points)) / 5
    h = h0
    if verbose:
        print("::h0:", h0)

    random.seed(16)
    center_voxel = 2.75 * voxel_size

    center_pcd = open3d.geometry.PointCloud()
    center_pcd.points = open3d.utility.Vector3dVector(pcd.points)
    center_pcd.normals = open3d.utility.Vector3dVector(pcd.normals)
    center_pcd = center_pcd.voxel_down_sample(center_voxel)
    centers = np.asarray(center_pcd.points)
    normals = np.asarray(center_pcd.normals)
    centers = np.hstack((centers, normals))
    my_Centers = myCenters(centers, h, 4000)
    density_weights = get_density_weights(np.asarray(pcd.points), h0)

    iters = iter
    if verbose:
        print("Max iterations: {}, Number points: {}, Number centers: {}".format(iters, len(pcd.points), len(centers)))
    time1 = time2 = 0
    for i in range(iters):

        bridge_points = 0
        non_branch_points = 0
        for center in my_Centers.myCenters:
            if center.label == 'bridge_point':
                bridge_points += 1
            if center.label == 'non_branch_point':
                non_branch_points += 1
        if verbose:
            print("::Iteration:{}, h:{}, bridge_points:{}".format(i, round(h, 3), bridge_points))

        centers = my_Centers.centers
        #tp = centers[:, :3]
        center_pcd.points = open3d.utility.Vector3dVector(centers[:, :3])
        if plot_iter_pcd:
            center_pcd.colors = open3d.utility.Vector3dVector( [0.2, 0.9, 0.1]*np.ones(centers.shape) )
            open3d.visualization.draw_geometries([center_pcd])

        for j in range(100):
            local_indices = get_local_points(np.asarray(pcd.points), centers, h)
            error = my_Centers.contract(np.asarray(pcd.points), local_indices, h, density_weights)
            my_Centers.update_properties()

        if try_make_skeleton:
            my_Centers.find_connections()

        h = h + (i + 1) * h0 / 2
        #h = h0 / (1 + i/2)
    final_centers = []
    #####added new prinicipal directions and leaf width computation#######################################
    principal_directions = []
    leaf_widths = []
    neighbors_list = []
    for c in my_Centers.myCenters:
        final_centers.append(c.center)
        #######  prinicpal direction compute#################
        local_indices = get_local_points(np.asarray(pcd.points), np.array([c.center]), h)[0]
        local_points = np.asarray(pcd.points)[local_indices]
        principal_direction = compute_principal_direction(local_points)
        principal_directions.append(principal_direction)

        # Get neighbors for each node
        cc = c.center[:3]
        neighbors = [index for index in local_indices if np.linalg.norm(cc - np.asarray(pcd.points)[index]) < h]
        neighbors_list.append(neighbors)

    xyz_centers = np.array(final_centers)[:,:3]
    principal_directions = np.array(principal_directions)

    # Compute leaf widths
    leaf_widths = compute_leaf_widths(xyz_centers, principal_directions, neighbors_list)

    return np.array(final_centers), principal_directions, leaf_widths
    # Post-process to remove boundary points
    # medial_axis_centers, medial_axis_directions, medial_axis_widths = post_process_skeleton(xyz_centers, principal_directions, leaf_widths, neighbors_list, h)
    #
    # return medial_axis_centers, medial_axis_directions, medial_axis_widths


# def post_process_skeleton(centers, directions, widths, neighbors_list, h):
#     # Remove points likely on the boundary
#     filtered_centers = []
#     filtered_directions = []
#     filtered_widths = []
#
#     for i in range(len(centers)):
#         if widths[i] > h / 10:  # Heuristic to remove boundary points
#             filtered_centers.append(centers[i])
#             filtered_directions.append(directions[i])
#             filtered_widths.append(widths[i])
#
#     return np.array(filtered_centers), np.array(filtered_directions), np.array(filtered_widths)

def post_process_skeleton(centers, directions, widths, neighbors_list, h):
    filtered_centers = []
    filtered_directions = []
    filtered_widths = []

    # Parameters (start with more lenient values)
    angular_threshold = 0.3  # Less strict on direction
    width_threshold = h / 20  # Less strict on width, narrower filter

    for i in range(len(centers)):
        if widths[i] > width_threshold:  # First, check the width
            # Get valid neighbors
            valid_neighbors = [idx for idx in neighbors_list[i] if idx < len(directions)]
            if not valid_neighbors:
                continue

            # Compute the average direction of neighbors
            neighbor_directions = directions[valid_neighbors]
            avg_neighbor_direction = np.mean(neighbor_directions, axis=0)
            avg_neighbor_direction /= np.linalg.norm(avg_neighbor_direction)

            # Compute angular difference
            angular_diff = np.dot(directions[i], avg_neighbor_direction)

            # Filter point if it meets both criteria
            if angular_diff > angular_threshold:
                filtered_centers.append(centers[i])
                filtered_directions.append(directions[i])
                filtered_widths.append(widths[i])
            else:
                # If it fails the direction test but has a larger width, still keep it
                if widths[i] > h / 15:  # Adaptive width threshold
                    filtered_centers.append(centers[i])
                    filtered_directions.append(directions[i])
                    filtered_widths.append(widths[i])

    return np.array(filtered_centers), np.array(filtered_directions), np.array(filtered_widths)




# # Define the transformation matrices for the RGB and violet cameras
# T_rgb = np.array([
#     [-0.43207212, 0.00809352, 0.90180274, -7.77181101],
#     [-0.04041172, -0.99912904, -0.01039507, -1.55595863],
#     [0.90093317, -0.04093482, 0.43202287, -10.77304866],
#     [0, 0, 0, 1]
# ])
#
# T_violet = np.array([
#     [-0.44994605, 0.07443196, 0.88994856, -7.41619033],
#     [-0.02950328, -0.99721567, 0.06848692, -0.13398110],
#     [0.89256827, 0.00455902, 0.45088924, -10.40321196],
#     [0, 0, 0, 1]
# ])
#
# # Calculate the inverse of the violet camera transformation matrix
# T_violet_inv = np.linalg.inv(T_violet)
#
# # Calculate the relative transformation matrix
# T_rel = np.dot(T_rgb, T_violet_inv)
#
# # Extract the rotation matrix and translation vector
# R_rel = T_rel[:3, :3]
# t_rel = T_rel[:3, 3]
#
# # Calculate the angle of rotation in degrees
# rotation = R.from_matrix(R_rel)
# angle_of_rotation = rotation.as_euler('xyz', degrees=True)
#
# # Calculate the translation vector magnitude
# translation_magnitude = np.linalg.norm(t_rel)
#
# print("Angle of rotation (degrees):", angle_of_rotation)
# print("Translation magnitude:", translation_magnitude)




day = '03-22_AM'
pc_path = '../data/lyon2/processed/{}_segmented.ply'
pcd_clean_option = {'downsample_rate': 80,
                    'voxel_size': 1.3,
                    'crop': False,
                    'crop_low_bound': 0.0,
                    'crop_up_bound': 1}
point_per_center = 20
verbose = True

if __name__ == "__main__":
    try_make_skeleton = False
    calculate_pcd = True
    downsample_adapt = True
    dirPath = "D:/Pheno4D/Pheno4D/tobacco" # "D:/4d_plant_analysis_data/data/maize/processed"
    outputpath = "D:/4d_plant_registration_data/tomato/plant3/with_normals"
    ply_file_name = "03-13.ply"
    pcd = open3d.io.read_point_cloud(os.path.join(outputpath,ply_file_name))
    norms = np.asarray(pcd.normals)

    pc_res = compute_pc_resolution(pcd)
    if len(pcd.normals) == 0:
    # b_min = pcd.get_min_bound()
    # b_max = pcd.get_max_bound()
    # len_diag = np.linalg.norm(b_max-b_min)
        estimate_normals(pcd, 2.0, pc_res, 30)
        open3d.io.write_point_cloud(os.path.join(outputpath, ply_file_name), pcd, write_ascii=True)

    print("Clean Point Cloud")
    print(64 * "=")
    if downsample_adapt:
        b_min = pcd.get_min_bound()
        b_max = pcd.get_max_bound()
        # dimns = b_max - b_min-
        # vol = dimns[0] * dimns[1] * dimns[2]
        # avg_density = len(np.asarray(pcd.points))/vol
        len_diag = np.linalg.norm(b_max-b_min)
        pc_res = compute_pc_resolution(pcd)
        pcd_clean_option['voxel_size'] = 0.020 * len_diag #0.042
    t_start = time.time()
    pcd, description, raw_mean = clean_point_cloud(pcd, option=pcd_clean_option, verbose=verbose)
    t_end = time.time()
    # if len(pcd.normals) == 0:
    #     estimate_normals(pcd,0.02,pc_res,10)
    # open3d.io.write_point_cloud(r"D:/Plant-PCR/skeleton-from-plant4d/downsampled_maize_03_21.ply",pcd,write_ascii=True)
    print(64 * "-")
    print("point cloud 1: {} points ".format(description['point_number']))
    print("used {:.3f}s".format(t_end - t_start))
    voxel_size = description['voxel_size']
    #open3d.visualization.draw_geometries([pcd])

    if calculate_pcd:
        t_start = time.time()
        #points = np.asarray(pcd.points)
        final_centers, principal_directions, leaf_widths = get_skeleton_pcd(pcd, voxel_size,  plot_iter_pcd=False, iter=2)
        print(final_centers.shape)
        #np.savetxt("../data/lyon2/skeleton_{}.csv".format(day), final_centers)
        #dde-mean centers
        pcd.points = open3d.utility.Vector3dVector(final_centers[:, :3])
        pcd.normals = open3d.utility.Vector3dVector(principal_directions)
        #pcd.normals = open3d.utility.Vector3dVector(final_centers[:, 3:])
        pcd.translate(raw_mean)
        fname = ply_file_name + ".csv"
        te_f_name = os.path.splitext(ply_file_name)[0] + "_new_centers.ply"
        csvpath = os.path.join(outputpath, fname)
        # pcd is translated and final centers are now in local coordinate system
        #final_centers[:, :3] = np.asarray(pcd.points)
        #final_centers[:, 3:] = np.asarray(pcd.normals)
        np.savetxt(csvpath, final_centers)
        open3d.io.write_point_cloud(os.path.join(outputpath, te_f_name), pcd, write_ascii=True)
    else:
        #final_centers = np.loadtxt("../data/lyon2/skeleton_{}.csv".format(day))
        final_centers = np.loadtxt(r"D:/Plant-PCR/skeleton-from-plant4d/skeleton_maize_03_21.csv")

    # pcd.points = open3d.utility.Vector3dVector(final_centers)
    # pcd = pcd.voxel_down_sample(0.8 * voxel_size)
    # open3d.visualization.draw_geometries([pcd])

    print("")
    print("Try Establishing Connections")
    print(64 * "=")
    xyz = np.asarray(pcd.points)




    xyz_normals = np.asarray(pcd.normals)
    print("::center number: {}".format(xyz.shape[0]))
    NNbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(xyz)
    distance, indices = NNbrs.kneighbors(xyz)

    threshold = 15 * voxel_size
    labels, vector_orientation = get_point_label(xyz, indices, distance, number_NN=15, sigma_threshold=0.5,
                                                 visualize_label=True)
    clusters, clusters_connection = \
        connect_pcd_branch(xyz, labels, vector_orientation, threshold, indices, distance, number_NN=15,
                           plot_branch=False)

    # t_end = time.time()
    # print("::using {:.2f}s".format(t_end - t_start))
    print("::discovering {} clusters".format(len(clusters)))
    connection = []
    for c in clusters_connection:
        connection += c
    print("::discovering {} edges".format(len(connection)))
    #plot_skeleton(xyz, np.array(connection), plot_point=True)

    clusters = [list(c) for c in clusters]
    clusters_ = []
    connection = []
    for i, cl in enumerate(clusters):
        if len(cl) > 2:
            clusters_.append(cl)
            connection += clusters_connection[i]
    clusters = clusters_

    connection, clusters = merge_all_clsuter(xyz, connection, clusters)

    connection = remove_cycle(xyz, connection, clusters)

    print(len(connection))

    plot_skeleton(xyz, np.array(connection), plot_point=True)
    fname = ply_file_name + "_new_graph.txt"
    skelgraph = os.path.join(outputpath, fname)
    print("Number of skeleton graph vertices: {}".format(xyz.shape[0]))
    t_end = time.time()
    print("::using {:.2f}s".format(t_end - t_start))
    save_graph(skelgraph, xyz, connection, xyz_normals, leaf_widths)
    # save the skeleton
    # if not os.path.exists("../data/lyon2/{}/".format(day)):
    #     os.makedirs("../data/lyon2/{}/".format(day))
    # save_graph("../data/lyon2/{}/skeleton_{}_connected.txt".format(day, day), xyz, connection)
    print(" ")
