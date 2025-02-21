import numpy as np
import matplotlib.pyplot as plt
from dijkstar import Graph, find_path
import skeleton as skel
from datetime import datetime
import  time
import scripts.maximal_clique as mcq
import networkx as nx

Mod = True

def to_prob_matrix(matrix):
    # Ensure the matrix is non-zero to avoid division by zero
    zero_indices = matrix == 0
    matrix[zero_indices] = np.nan  # Set 0 values to NaN temporarily
    prob_matrix = 1.0 / (1.0 + matrix)  # Map all values to [0, 1]
    prob_matrix[zero_indices] = 0.0  # Set NaN values back to 0
    return prob_matrix

def normalize_emission_matrix(E):
    total_likelihoods = np.sum(E, axis=1)
    # Step 1: Calculate the sum of emission probabilities for each observation
    E_normalized = E / total_likelihoods[:, np.newaxis]

    return E_normalized


def skeleton_matching(S1, S2, params):
    """
  Computes the correspondences given for a skeleton pair using a HMM formulation.

  Parameters
  ----------
  S1, S2 : Skeleton Class
    Two skeletons for which we compute the correspondences
  params : Dictionary
    Can be an empty dict (All params have some default values)

    weight_e: Used in builing the emissions matrix E.It weighss the difference between the
              skeleton point vs. the difference in the degree of a graph node.
              Set to low number if you want more matches end-points to end-points
              default: 10.0
    match_ends_to_ends: if false, an endpoint to a middlepoint gets equal weights for E
                        as middlepoint to middlepoint.
                        default:false
    use_labels:         If true, then labels are used for the emmision cost.
                        Labels must be given in S.labels for every node.
                        default: false
    label_penalty:      penalty for emmision if labels are not the same.
                        default: 1 (1 means: same cost as node degree with a difference of 1)
    debug:              show plots and other info (true or false)
                        default: false
  Returns
  -------
  corres : numpy array [Mx2]
    column 0 has node ids from S1, and column 1 has corresponding node ids from S2.

  """
    print("Computing matches.")
    # set default params is not provided
    # in emissions E  : weights the difference between the
    # skeleton point vs. the difference in the degree of a graph node
    if 'weight_e' not in params:
        params['weight_e'] = 10.0

    # use semantic labels
    if 'use_labels' not in params:
        params['use_labels'] = False

    # apply label penalty or not
    if 'label_penalty' not in params:
        params['label_penalty'] = False

    # show debug msgs/vis
    if 'debug' not in params:
        params['debug'] = False

    # define HMM S1->S2
    print("HMM: Computing Transition and Emission probabilities")
    T1, E1, statenames1 = define_skeleton_matching_hmm(S1, S2, params)
    #E1 = normalize_emission_matrix(E1)

    # Transform T and E to probability
    to_prob = lambda x: 1 / x
    T1 = to_prob(T1)
    E1 = to_prob(E1)

    # compute correspondence pairs using viterbi
    V = np.array(S1.get_sequence())
    #best_seq = viterbi(V, T1, E1, statenames1)
    # v_time_s = time.time()
    #best_seq = viterbi_algorithm(V, T1, E1, statenames1)
    # v_time_e = time.time()
    #print("elapsed_time(sec) in viterbi search comp:", (v_time_e-v_time_s))
    #best_seq = viterbi_algorithm_sparse(V, T1, E1, statenames1,1e5)
    #best_seq = viterbi(V, T1, E1, statenames1)
    #best_seq = viterbi_sparse_with_invalid(V, T1, E1, statenames1)

    #best_seq = viterbi_sparse_with_gpu(V, T1, E1, statenames1)
    #best_seq = viterbi_parallel(V, T1, E1, statenames1)
    #best_seq = beam_search(V, T1, E1, statenames1,V.shape[0])
    v_time_s = time.time()
    best_seq = iterative_beam_search(V, T1, E1, statenames1, 4 * V.shape[0],0.1,2,-10) #0.70 or around 0.5
    #best_seq = iterative_beam_search_with_mcq(V, T1, E1, statenames1,  2 * V.shape[0],0.5,2,-10)
    v_time_e = time.time()
    print("elapsed_time(sec) in beam search comp:", (v_time_e-v_time_s))
    corres = get_correspondences_from_seq(best_seq)


    # remove all matchings to virtual 'nothing' node
    ind_remove = np.where(corres[:, 1] == -1)
    corres = np.delete(corres, ind_remove, 0)

    # post process
    corres,distances = remove_double_matches_in_skeleton_pair(S1, S2, corres)
    print("Number of unique correspondences:", corres.shape[0])
    # v_time_s = time.time()
    # result_corres  = mcq.maximal_clique_community(corres,distances, 10)
    # v_time_e = time.time()
    # print("elapsed_time(sec) in maximal clique:", (v_time_e-v_time_s))

    # v_time_s = time.time()
    # filtered_corres, filter_dist = filter_correspondences(corres, distances, 30)
    # v_time_e = time.time()
    # print("elapsed_time(sec) in distance filtering:", (v_time_e-v_time_s))

    # visualize matching results
    if params['debug']:
        fh_debug = plt.figure()
        skel.plot_skeleton(fh_debug, S1, 'b')
        skel.plot_skeleton(fh_debug, S2, 'r')
        plot_skeleton_correspondences(fh_debug, S1, S2, corres)

    return corres

def filter_correspondences(correspondences, distances, threshold):
    filtered_correspondences = []
    filtered_distances = []

    for correspondence, distance in zip(correspondences, distances):
        if distance < threshold:
            filtered_correspondences.append(correspondence)
            filtered_distances.append(distance)

    filtered_correspondences = np.array(filtered_correspondences)
    filtered_distances = np.array(filtered_distances)

    return filtered_correspondences, filtered_distances

def define_skeleton_matching_hmm(S1, S2, params):
    """
  Define cost matrices for Hidden Markov Model for matching skeletons

  Parameters
  ----------
  S1, S2 :  Skeleton Class
            Two skeletons for which we compute the correspondences
  params :  Dictionary
            see skeleton_matching function for details

  Returns
  -------
  T : numpy array
      Defines the cost from one pair to another pair
  E : numpy array
      emmision cost matrixd efines the cost for observing one pair as a match
  statenames : list
               names for each state used in the HMM (all correspondence pairs)

  """
    # define statenames
    statenames = define_statenames(S1, S2)

    # Precompute geodesic distances
    GD1, NBR1 = compute_geodesic_distance_on_skeleton(S1)
    GD2, NBR2 = compute_geodesic_distance_on_skeleton(S2)

    # Precompute euclidean distances between all pairs
    ED = compute_euclidean_distance_between_skeletons(S1, S2)

    # compute transition matrix
    if Mod:
        T = compute_transition_matrix_optim(S1, S2, GD1, NBR1, GD2, NBR2, ED)
    else:
        T = compute_transition_matrix(S1, S2, GD1, NBR1, GD2, NBR2, ED)

    # compute emission matrix
    if Mod:
        E = compute_emission_matrix_optim(S1, S2, ED, params)
    else:
        E = compute_emission_matrix(S1, S2, ED, params)


    return T, E, statenames


def define_statenames(S1, S2):
    # number of nodes in each skeleton
    N = S1.XYZ.shape[0]
    M = S2.XYZ.shape[0]

    statenames = []
    # starting state
    statenames.append([-2, -2])
    for n1 in range(N):
        for m1 in range(M):
            statenames.append([n1, m1])
    for n1 in range(N):
        statenames.append([n1, -1])

    return statenames


def get_correspondences_from_seq(seq):
    N = len(seq)
    corres = np.zeros((N, 2), dtype=np.int64)
    for i in range(N):
        corres[i, 0] = seq[i][0]
        corres[i, 1] = seq[i][1]

    return corres


def compute_geodesic_distance_on_skeleton(S):

    G = Graph(undirected=True)
    edges = S.edges
    for e in edges:
        edge_length = euclidean_distance(S.XYZ[e[0], :], S.XYZ[e[1], :])
        G.add_edge(e[0], e[1], edge_length)

    # edges = S.edges
    # edge_lengths = cdist(S.XYZ, S.XYZ)  # Compute all pairwise Euclidean distances
    # for e in edges:
    #     i, j = e
    #     edge_length = edge_lengths[i, j]
    #     S.A[i, j] = edge_length
    #     S.A[j, i] = edge_length

    # t1_loop = time.time()
    N = S.XYZ.shape[0]
    # GD1 = np.zeros([N, N])
    # NBR1 = np.zeros([N, N])
    # for i in range(N):
    #     for j in range(i + 1, N):
    #         gdist, nbr = geodesic_distance_no_branches(G, S.A, i, j)
    #         GD1[i, j] = gdist
    #         GD1[j, i] = gdist
    #         NBR1[i, j] = nbr
    #         NBR1[j, i] = nbr
    # t2_loop = time.time()
    # print("elapsed_time(sec) in nested geodesic comp:", (t2_loop-t1_loop))
    #vectorized impl
    # Create a meshgrid of indices for all pairs of nodes
    t11_loop = time.time()
    ix1, ix2 = np.triu_indices(N, k=1)  # Generate only upper triangular indices to avoid duplicates

    # Calculate geodesic distances between all pairs of nodes
    all_shortest_paths = np.array([geodesic_distance_no_branches(G, S.A,  i, j) for i, j in zip(ix1, ix2)])

    GD = np.zeros((N, N))
    NBR = np.zeros((N, N))

    # Fill in the GD and NBR matrices using the computed geodesic distances and number of branches
    GD[ix1, ix2] = GD[ix2, ix1] = all_shortest_paths[:, 0]
    NBR[ix1, ix2] = NBR[ix2, ix1] = all_shortest_paths[:, 1]
    t22_loop = time.time()
    print("elapsed_time(sec) in vectorized geodesic comp:", (t22_loop-t11_loop))

    return GD, NBR


def geodesic_distance_no_branches(G, A, i, j):
    path = find_path(G, i, j)
    gdist = path.total_cost

    nbr = 0
    for i in range(1, len(path.nodes)):
        degree = sum(A[path.nodes[i], :])
        nbr = nbr + (degree - 2)  # A normal inner node has degree 2, thus additional branches are degree-2

    return gdist, nbr


def mean_distance_direct_neighbors(S, GD):
    N = S.XYZ.shape[0]
    sumd = 0
    no = 0
    for i in range(N):
        for j in range(N):
            if S.A[i, j]:
                sumd = sumd + GD[i, j]
                no = no + 1

    mean_dist = sumd / no

    return mean_dist


def euclidean_distance(x1, x2):
    dist = np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2 + (x1[2] - x2[2]) ** 2)
    return dist


def compute_euclidean_distance_between_skeletons(S1, S2):
    N = S1.XYZ.shape[0]
    M = S2.XYZ.shape[0]

    # D = np.zeros([N, M])
    # t1_loop = datetime.now()
    # for n1 in range(N):
    #     for m1 in range(M):
    #         D[n1, m1] = euclidean_distance(S1.XYZ[n1, :], S2.XYZ[m1, :])
    # t2_loop = datetime.now()
    # print("elapsed_time in nested-loops-dist:", (t2_loop-t1_loop).total_seconds())
    # # # Reshape the input arrays to enable broadcasting
    t1_loop = datetime.now()
    S1_reshaped = S1.XYZ[:, np.newaxis, :]
    S2_reshaped = S2.XYZ[np.newaxis, :, :]

    # Calculate the squared Euclidean distance
    squared_distance = np.sum((S1_reshaped - S2_reshaped) ** 2, axis=-1)

    # Take the square root to get the Euclidean distance
    distance = np.sqrt(squared_distance)
    #normalized_dist = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
    t2_loop = datetime.now()
    print("elapsed_time(sec) in vectorized-dist:", (t2_loop-t1_loop).total_seconds())

    return distance
    #return D


def remove_double_matches_in_skeleton_pair(S1, S2, corres):
    """
  Remove double matches of two skeletons and keeps only the one with the smallest distance.

  Parameters
  ----------
  S1, S2 : Skeleton Class
           Two skeletons for which we compute the correspondences
  corres : numpy array (Mx2)
           correspondence between two skeleton nodes pontetially with one-to-many matches

  Returns
  -------
  corres: numpy array
          one-to-one correspondences between the skeleton nodes.

  """
    num_corres = corres.shape[0]
    distances = np.zeros((num_corres, 1))
    for i in range(num_corres):
        distances[i] = euclidean_distance(S1.XYZ[corres[i, 0], :], S2.XYZ[corres[i, 1], :])

    # Remove repeated corres 1 -> 2
    corres12, counts12 = np.unique(corres[:, 0], return_counts=True)
    ind_remove12 = []
    for i in range(len(corres12)):
        if counts12[i] > 1:
            ind_repeat = np.argwhere(corres[:, 0] == corres12[i]).flatten()
            dist_repeat = distances[ind_repeat].flatten()
            ind_ = np.argsort(dist_repeat)[1:]
            ind_remove12.extend(ind_repeat[ind_])

    # Remove repeated corres 2 -> 1
    corres21, counts21 = np.unique(corres[:, 1], return_counts=True)
    ind_remove21 = []
    for i in range(len(corres21)):
        if counts21[i] > 1:
            ind_repeat = np.argwhere(corres[:, 1] == corres21[i]).flatten()
            dist_repeat = distances[ind_repeat].flatten()
            ind_ = np.argsort(dist_repeat)[1:]
            ind_remove21.extend(ind_repeat[ind_])

    ind_remove12 = np.unique(ind_remove12)
    ind_remove21 = np.unique(ind_remove21)

    # Exclude rows with indices from ind_remove12 and ind_remove21
    ind_to_remove = np.union1d(ind_remove12, ind_remove21).astype(dtype= int)
    corres = np.delete(corres, ind_to_remove, axis=0)
    distances = np.delete(distances, ind_to_remove, axis=0)

    return corres, distances


def compute_transition_matrix(S1, S2, GD1, NBR1, GD2, NBR2, ED):
    N = S1.XYZ.shape[0]
    M = S2.XYZ.shape[0]
    T = 1e6 * np.ones([N * M + N, N * M + N], dtype=np.float64)
    v_reshaped_arr = np.zeros([N,M,N,M],dtype=np.float64)
    dp_arr = 1e6 * np.ones([N,M,N,M],dtype=np.float64)
    mean_dist_S1 = mean_distance_direct_neighbors(S1, GD1)
    max_cost_normal_pairs = 0

    # normal pair to normal pair:
    # (n1,m1) first pair, (n2,m2) second pair
    max_cost_normal_pairs = 0
    for n1 in range(N):
        for m1 in range(M):
            for n2 in range(N):
                for m2 in range(M):
                    # Avoid going in different directions on the skeleton. Then the
                    # geodesic difference can be small, but actually one would assume a
                    # large cost
                    v_inS1 = S1.XYZ[n2, :] - S1.XYZ[n1, :]
                    v_inS2 = S2.XYZ[m2, :] - S2.XYZ[m1, :]

                    # angle between vectors smaller 90 degrees
                    dp = np.dot(v_inS1, v_inS2)
                    dp_arr[n1,m1,n2,m2] = dp
                    #print("Value of dp: {:.3f}".format(dp))
                    if dp >= 0:
                        # geodesic distance and difference in number of branches along the way
                        g1 = GD1[n1, n2]
                        g2 = GD2[m1, m2]
                        br1 = NBR1[n1, n2]
                        br2 = NBR2[m1, m2]
                        v = np.abs(br1 - br2) * np.max(GD1) + np.abs(g1 - g2) + mean_dist_S1
                        v_reshaped_arr[n1][m1][n2][m2] = v
                        T[n1 * M + m1, n2 * M + m2] = v

                        if v > max_cost_normal_pairs:
                            max_cost_normal_pairs = v

    #np.savetxt("dp-old.txt", dp_arr, fmt='%.3f', delimiter=' ,')
    v_reshaped_arr = v_reshaped_arr.reshape(N, M, N, M)
    #print("Value of T: {:.3f}".format(T[1, 1]))
    # Main diagonal should be large
    for i in range(N * M):
        T[i, i] = max_cost_normal_pairs

    # Normal pair -> not present
    for n1 in range(N):
        for m1 in range(M):
            for n2 in range(N):
                if n2 != n1:
                    T[n1 * M + m1, N * M + n2] = np.max(GD1) / 2

    print("Value of T: {:.3f}".format(T[1, 1]))
    # Not present -> normal pair
    for n2 in range(N):
        for m1 in range(M):
            for n1 in range(N):
                T[N * M + n2, n1 * M + m1] = max_cost_normal_pairs

    print("Value of T: {:.3f}".format(T[1, 1]))
    # Not present -> not present
    S1_seq = S1.get_sequence()
    for n1 in range(N):
        for n2 in range(N):
            pos_n1 = S1_seq.index(n1) if n1 in S1_seq else -1
            pos_n2 = S1_seq.index(n2) if n2 in S1_seq else -1
            if pos_n2 > pos_n1:
                v = GD1[n1, n2]
                T[N * M + n1, N * M + n2] = v + mean_dist_S1 + np.min(ED[n1, :])

    print("Value of T: {:.3f}".format(T[1, 1]))
    # Add starting state (which will not be reported in hmmviterbi)
    sizeT = N * M + N;
    T1 = np.hstack((1e6 * np.ones((1, 1)), np.ones((1, N * M)), 1e6 * np.ones((1, N))))
    T2 = np.hstack((1e6 * np.ones((sizeT, 1)), T))
    T = np.vstack((T1, T2))
    zero_indices = np.argwhere(T == 0)
    print("Value of T: {:.3f}".format(T[1, 1]))
    #np.savetxt("transition_matrix_ori.txt", T, fmt='%1.4f', delimiter=' ,')
    return T


def compute_transition_matrix_optim(S1, S2, GD1, NBR1, GD2, NBR2, ED):
    N = S1.XYZ.shape[0]
    M = S2.XYZ.shape[0]
    sizeT = N * M + N

     # Vectorized version
    n1_vals = np.arange(N)
    m1_vals = np.arange(M)
    n2_vals = np.arange(N)
    m2_vals = np.arange(M)

    n1, m1, n2, m2 = np.ix_(n1_vals, m1_vals, n2_vals, m2_vals)

    T = 1e6 * np.ones([sizeT, sizeT], dtype=np.float64)

    mean_dist_S1 = mean_distance_direct_neighbors(S1, GD1)

    S1_XYZ_reshaped = S1.XYZ[:, np.newaxis, np.newaxis, :]
    S2_XYZ_reshaped = S2.XYZ[np.newaxis, np.newaxis, :, :]
    v_inS1 = S1_XYZ_reshaped[:, np.newaxis, :, :] - S1_XYZ_reshaped[np.newaxis, :, :, :]
    v_inS2 = S2_XYZ_reshaped[:, :, :, np.newaxis] - S2_XYZ_reshaped[:, :, np.newaxis, :]
    #dot_product_vectorized = np.einsum('ijkl,ijkl->ijk', v_inS1, v_inS2)
    #dot_product_vectorized = np.einsum('ijkl,mnop->ijmn', v_inS1, v_inS2)
    dot_product_vectorized = np.sum(v_inS1 * v_inS2, axis=-1)
    #dot_product_vectorized = np.sum((S1_XYZ_reshaped[:, np.newaxis, :, :] - S2_XYZ_reshaped[np.newaxis, :, :, :]) ** 2, axis=-1)
    #
    # # # Compute dot product
    # dot_product = np.dot(v_inS1_flat, v_inS2_flat.T).reshape(N, M, N, M)
    # #np.savetxt("dp-modified.txt", dot_product, fmt='%.3f', delimiter=' ,')
    #
    # Reshape dot product result to original shape
    #dp = np.dot(v_inS1_flat, v_inS2_flat.T).reshape(N, M, N, M)
    #print(dp)
    dot_product_vectorized = np.swapaxes(dot_product_vectorized, 1, 2)
    valid_pairs = dot_product_vectorized >= 0

    # v_inS1_flat = v_inS1.reshape(N, M,3)  # Reshape to (N, 1, M, 1, 3)
    # v_inS2_flat = v_inS2.reshape(N,M,3 )  # Reshape to (1, N, 1, M, 3)

    #dot_product = np.sum(v_inS1[..., np.newaxis, :] * v_inS2.transpose((0, 2, 1, 3)), axis=-1)
    #cosine similarity
    cos_sim = np.dot(S1.normals, S2.normals.T)
    cos_sim_min = np.min(cos_sim)
    cos_sim_max = np.max(cos_sim)

    if cos_sim_min == cos_sim_max:
        # Handle the case where all values are the same to avoid division by zero
        normalized_cos_sim = np.ones_like(cos_sim)
    else:
        # Normalize the cosine similarity values
        normalized_cos_sim = (cos_sim - cos_sim_min) / (cos_sim_max - cos_sim_min)

    # You can choose a small constant relative to the data range
    # Here, we use 1% of the data range as a small constant
    small_constant = 0.05 * (cos_sim_max - cos_sim_min)
    normalized_cos_sim += small_constant
    g1 = GD1[n1, n2]
    g2 = GD2[m1, m2]
    br1 = NBR1[n1, n2]
    br2 = NBR2[m1, m2]

    v = np.abs(br1 - br2) * np.max(GD1) + np.abs(g1 - g2) + mean_dist_S1
    v[~valid_pairs] = 1e6
    #max_cost_normal_pairs = np.max(v)
    # Combine cosine similarity with other factors in the transition matrix calculation
    v_with_cos_sim = v  * normalized_cos_sim
    T[:N * M, :N * M] = v_with_cos_sim.reshape(N * M, N * M)
    #print("Value of T: {:.3f}".format(T[1, 1]))
    # Main diagonal should be large
    #T[np.arange(N * M), np.arange(N * M)] = max_cost_normal_pairs
    #temp arr copy
    V_temp = v
    inv_value = 1e6
    V_temp[V_temp == inv_value] = np.nan
    max_val = np.nanmax(V_temp)
    np.fill_diagonal(T[:N * M, :N * M], max_val)

    # Normal pair -> not present
    T[:N * M, N * M:] = np.max(GD1) / 2
    print("Value of T: {:.3f}".format(T[1, 1]))

    # Not present -> normal pair
    T[N * M:, :N * M] = max_val
    zero_indices = np.argwhere(T == 0)
    #print("Value of T: {:.3f}".format(T[1, 1]))

    #working as well
    # S1_seq = S1.get_sequence()
    # S1_seq_indices = [S1_seq.index(n1) if n1 in S1_seq else -1 for n1 in range(N)]
    #
    # for n1 in range(N):
    #     for n2 in range(N):
    #         pos_n1 = S1_seq_indices[n1]
    #         pos_n2 = S1_seq_indices[n2]
    #         if pos_n2 > pos_n1:
    #             v = GD1[n1, n2]
    #             T[N * M + n1, N * M + n2] = v + mean_dist_S1 + np.min(ED[n1, :])
    S1_seq = S1.get_sequence()
    S1_seq_indices = np.where(np.isin(np.arange(N), S1_seq), np.arange(N), -1)
    min_ED = np.min(ED, axis=1)
    mask = S1_seq_indices[:, np.newaxis] < S1_seq_indices
    v = GD1[mask]
    n1_indices, n2_indices = np.where(mask)
    T[N * M + n1_indices, N * M + n2_indices] = v + mean_dist_S1 + min_ED[n1_indices]
    #print("Value of T: {:.3f}".format(T[1, 1]))
    T1 = np.hstack((1e6 * np.ones((1, 1)), np.ones((1, N * M)), 1e6 * np.ones((1, N))))
    T2 = np.hstack((1e6 * np.ones((sizeT, 1)), T))
    T = np.vstack((T1, T2))
    #print("Value of T: {:.3f}".format(T[1, 1]))
    #np.savetxt("transition_matrix_modified.txt", T, fmt='%1.4f', delimiter=' ,')
    return T


def compute_emission_matrix(S1, S2, ED, params):
     # emmision matrix (state to sequence)
    # Here: degree difference + euclidean difference inside a pair
    N = S1.XYZ.shape[0]
    M = S2.XYZ.shape[0]
    E = 1e05 * np.ones((N * M + N, N))
    abs_degee = 1e05 * np.ones((N, M))
    for n1 in range(N):
        for m1 in range(M):

            degree1 = float(np.sum(S1.A[n1, :]))
            degree2 = float(np.sum(S2.A[m1, :]))

            # Do not penalize end node against middle node
            if not params['match_ends_to_ends'] and (
                    (degree1 == 1 and degree2 == 2) or (degree1 == 2 and degree2 == 1)):
                E[n1 * M + m1, n1] = params['weight_e'] * (ED[n1, m1] + 10e-10)
            else:
                abs_degee[n1][m1] = np.abs(degree1 - degree2)
                E[n1 * M + m1, n1] = np.abs(degree1 - degree2) + params['weight_e'] * (ED[n1, m1] + 10e-10)

    #print("values of ed_val:", ed_val)
    # Add penalty if labels are not consistent
    if params['use_labels']:
        for n1 in range(N):
            for m1 in range(M):
                if S1.labels[n1] != S2.labels[m1]:
                    E[n1 * M + m1, n1] = E[n1 * M + m1, n1] + params['label_penalty']

    print("E-values-old:", E)
    # No match
    for n1 in range(N):
        # Take the  best
        I = np.argsort(E[0:N * M, n1])
        E[N * M + n1, n1] = E[I[0], n1]

    # Add starting state (which will not be reported in hmmviterbi)
    E = np.vstack((1e10 * np.ones((1, N)), E))
    #np.savetxt("emission-matrix.txt", E, fmt='%.3f', delimiter=' ')

    return E

def compute_emission_matrix_optim(S1, S2, ED, params):
    N = S1.XYZ.shape[0]
    M = S2.XYZ.shape[0]
    E = 1e05 * np.ones((N * M + N, N))

    degree1 = np.sum(S1.A, axis=1).astype(np.int32)
    degree2 = np.sum(S2.A, axis=1).astype(np.int32)

    # Reshape degrees for broadcasting
    degree1 = degree1.reshape(N, 1)
    degree2 = degree2.reshape(1, M)

    # Do not penalize end node against middle node
    mask = ~params['match_ends_to_ends'] & (
        ((degree1 == 1) & (degree2 == 2)) | ((degree1 == 2) & (degree2 == 1))
    )

    # Base emission cost using degree difference and Euclidean distance
    weight_e_times_ed = params['weight_e'] * (ED + 10e-10)
    abs_degree_diff = np.abs(degree1 - degree2)

    # Compute angle differences using principal directions
    angles1 = np.array([compute_angle_principal(i, np.where(S1.A[i] > 0)[0], S1.normals) for i in range(N)])
    angles2 = np.array([compute_angle_principal(j, np.where(S2.A[j] > 0)[0], S2.normals) for j in range(M)])

    # Reshape for broadcasting
    angles1 = angles1.reshape(N, 1)
    angles2 = angles2.reshape(1, M)
    angle_diff = np.abs(angles1 - angles2)

    # Incorporate angle difference into the emission cost
    angle_diff = angle_diff.reshape(N * M, 1)
    abs_degree_diff = abs_degree_diff.reshape(N * M, 1)
    weight_e_times_ed = weight_e_times_ed.reshape(N * M, 1)

    # Combine all components into the emission matrix
    total_emission_cost = abs_degree_diff + weight_e_times_ed + angle_diff * 0.1 #* params['weight_angle'] *
    E[:N*M, :1] = total_emission_cost

    # Apply mask to avoid penalizing certain node matches
    # E[:N*M, :1][mask] = weight_e_times_ed[mask]  # This line can be commented or uncommented depending on specific behavior

    # Shift elements for proper emission matrix format
    E = shift_elements_emission(E, N, M)

    for n1 in range(N):
        # Select the best match for each node in S1
        I = np.argsort(E[0:N * M, n1])
        E[N * M + n1, n1] = E[I[0], n1]

    # Add starting state (which will not be reported in HMM Viterbi)
    E = np.vstack((1e10 * np.ones((1, N)), E))
    zero_indices = np.argwhere(np.isclose(E, 0.0, atol=1e-8))
    if zero_indices.shape[0] > 0:
        E[zero_indices[:, 0], zero_indices[:, 1]] += 1e5

    return E

#############################################################commented on 29.08.2024, the function works####################
# def compute_emission_matrix_optim(S1, S2, ED, params):
#
#     N = S1.XYZ.shape[0]
#     M = S2.XYZ.shape[0]
#     E = 1e05 * np.ones((N * M + N, N))
#
#     degree1 = np.sum(S1.A, axis=1).astype(np.int32)
#     degree2 = np.sum(S2.A, axis=1).astype(np.int32)
#     #cos_sim = np.dot(S1.normals, S2.normals.T)
#
#     #normalized_cos_sim = (cos_sim - np.min(cos_sim)) / (np.max(cos_sim) - np.min(cos_sim))
#     # zero_indices = np.argwhere(normalized_cos_sim == 0)
#     # if zero_indices.shape[0] > 0:
#     #     normalized_cos_sim[zero_indices[:, 0], zero_indices[:, 1]] += 1e-5
#
#     #this works as well
#     # for n1 in range(N):
#     # # Do not penalize end node against middle node
#     #     is_special_case = (degree1[n1] == 1) & (degree2 == 2) | (degree1[n1] == 2) & (degree2 == 1)
#     #     E[n1 * M : (n1 + 1) * M, n1] = np.where(
#     #         is_special_case,
#     #         params["weight_e"] * (ED[n1, :] + 1e-10),
#     #         np.abs(degree1[n1] - degree2) + params["weight_e"] * (ED[n1, :] + 1e-10),
#     #     )
#
#     # Reshape the arrays to have the same shape for broadcasting
#     degree1 = degree1.reshape(N, 1)
#     # print(degree1)
#     # print("\n")
#     degree2 = degree2.reshape(1, M)
#     # print(degree2)
#
#     # Do not penalize end node against middle node
#     mask = ~params['match_ends_to_ends'] & (
#         ((degree1 == 1) & (degree2 == 2)) | ((degree1 == 2) & (degree2 == 1))
#     )
#
#     weight_e_times_ed = params['weight_e'] * (ED + 10e-10) #+ cos_sim
#     abs_degree_diff = np.abs(degree1 - degree2)
#     #print(abs_degree_diff)
#
#     # Reshape arrays to have the same shape before addition
#     abs_degree_diff = abs_degree_diff.reshape(N * M, 1)
#     # print(abs_degree_diff)
#     # print()
#     weight_e_times_ed = weight_e_times_ed.reshape(N * M, 1)
#     # print(weight_e_times_ed)
#     test_arr = abs_degree_diff + weight_e_times_ed
#     E[:N*M, :1] = abs_degree_diff + weight_e_times_ed
#     #E[:N*M, :1][mask] = weight_e_times_ed[mask]
#     E = shift_elements_emission(E, N, M)
#
#     # # Convert labels to numpy arrays
#     # S1_labels = np.array(S1.labels)
#     # S2_labels = np.array(S2.labels)
#     #
#     # if params['use_labels']:
#     #     # mismatch_labels = (S1_labels[:, np.newaxis] != S2_labels)
#     #     # mismatch_labels = mismatch_labels.reshape(N * M, 1)  # Reshape to match E[:N*M, :]
#     #     # val = mismatch_labels[29,0]
#     #     # tt = E[29,1]
#     #     mismatch_labels = create_mismatch_labels(S1_labels,S2_labels, params['label_penalty'])
#     #     E += mismatch_labels
#     #     #E[:N*M, :1] += mismatch_labels * params['label_penalty']
#     #     #print("E-values-new:", E)
#
#     for n1 in range(N):
#         # Take the  best
#         I = np.argsort(E[0:N * M, n1])
#         E[N * M + n1, n1] = E[I[0], n1]
#
#     # Add starting state (which will not be reported in hmmviterbi)
#     E = np.vstack((1e10 * np.ones((1, N)), E))
#     zero_indices = np.argwhere(np.isclose(E, 0.0, atol=1e-8))
#     if zero_indices.shape[0] > 0:
#         E[zero_indices[:, 0], zero_indices[:, 1]] += 1e5
#     return E
###############################################
def shift_elements_emission(input_array, N, M):
  # Initialize a new array to store the shifted values
    shifted_arr = np.full_like(input_array, 1e5)

    # Perform the shifting for each block of rows (M rows at a time)
    start_row = M
    for i in range(1, N):
        end_row = start_row + M

        # Copy the first column to the shifted position in the new array
        col_offset = i  # Calculate the offset for the first column based on i
        shifted_arr[start_row:end_row, col_offset] = input_array[start_row:end_row, 0]
        start_row = end_row
    start_row = 0
    end_row = M
    shifted_arr[start_row:end_row, 0] = input_array[start_row:end_row, 0]
    return shifted_arr

def create_mismatch_labels(S1_labels, S2_labels, labels_penalty):
    N, M = len(S1_labels), len(S2_labels)

    # Broadcast label arrays for comparison
    labels_S1 = np.array(S1_labels)
    labels_S2 = np.array(S2_labels)
    mismatch_labels = (labels_S1[:, np.newaxis] != labels_S2)

    # Create the mismatch_labels_all array and fill in the values
    mismatch_labels_all = np.zeros((N * M + N, N), dtype=bool)

    # Fill in the mismatch_labels for elements between S1 and S2
    mismatch_labels_all[:N * M, :] = mismatch_labels.ravel()[:, np.newaxis]

    # Fill in the mismatch_labels for elements within S1
    mismatch_labels_all[N * M:, :] = mismatch_labels_all[N * M:, :] * labels_penalty


    return mismatch_labels_all


def viterbi_algorithm(V, T, E, StateNames):
    M = V.shape[0]
    N = T.shape[0]

    T_log = np.log(np.where(T >= 1e6, 1e-20, T))  # Use log probabilities to avoid underflow for invalid elements
    E_log = np.log(np.where(E >= 1e5, 1e-20, E))  # Use log probabilities for observations and handle invalid elements

    V_log = np.log(V + 1e-20)  # Use log probabilities for observations

    # Initialize the Viterbi matrix and backtrace matrix
    Viterbi = np.zeros((N, len(V)))
    backtrace = np.zeros((N, len(V)), dtype=int)

    # Initialize the first column of Viterbi matrix with the initial probabilities
    Viterbi[:, 0] = T_log[:, 0] + E_log[:, V[0]]

    # Perform the dynamic programming step for each time step
    for t in range(1, len(V)):
        temp = Viterbi[:, t - 1][:, np.newaxis] + T_log + E_log[:, V[t]][np.newaxis, :]
        Viterbi[:, t] = np.max(temp, axis=0) + V_log[t]
        backtrace[:, t] = np.argmax(temp, axis=0)

    # Backtrack to find the best path
    S = [np.argmax(Viterbi[:, -1])]
    for t in range(len(V) - 1, 0, -1):
        S.append(backtrace[S[-1], t])

    # Convert state indices to state names
    S = [StateNames[state_idx] for state_idx in reversed(S)]

    return S



# def viterbi(V, T, E, StateNames):
#     """
#   This function computes a sequence given observations, transition and emission prob. using the Viterbi algorithm.
#
#
#   Parameters
#   ----------
#   V : numpy array (Mx1)
#       Observations
#   T : numpy array (NxN)
#       Transition probabilities
#   E : numpy array (NxM)
#       Emission probabilities
#   StateNames : list
#                 ames for each state used in the HMM
#
#   Returns
#   -------
#   S :  list
#       Best sequence of hidden states
#
#   """
#     M = V.shape[0]
#     N = T.shape[0]
#
#     omega = np.zeros((M, N))
#     omega[0, :] = np.log(E[:, V[0]])
#
#     prev = np.zeros((M - 1, N))
#
#     for t in range(1, M):
#         for j in range(N):
#             # Same as Forward Probability
#             probability = omega[t - 1] + np.log(T[:, j]) + np.log(E[j, V[t]])
#
#             # This is our most probable state given previous state at time t (1)
#             prev[t - 1, j] = np.argmax(probability)
#
#             # This is the probability of the most probable state (2)
#             omega[t, j] = np.max(probability)
#
#         # Path Array
#         S_ = np.zeros(M)
#
#         # Find the most probable last hidden state
#         last_state = np.argmax(omega[M - 1, :])
#
#         S_[0] = last_state
#
#     backtrack_index = 1
#     for i in range(M - 2, -1, -1):
#         S_[backtrack_index] = prev[i, int(last_state)]
#         last_state = prev[i, int(last_state)]
#         backtrack_index += 1
#
#     # Flip the path array since we were backtracking
#     S_ = np.flip(S_, axis=0)
#
#     # Convert numeric values to actual hidden states
#     S = []
#     for s in S_:
#         S.append(StateNames[int(s)])
#
#     return S

def beam_search(V, T, E, StateNames, beam_width):
    M = V.shape[0]
    N = T.shape[0]

    M = V.shape[0]
    N = T.shape[0]

    # Create valid masks for transmission and emission matrices
    valid_transmission_mask = np.logical_and(T >= 1e-5, T != 1.0)
    valid_emission_mask = E > 1e-5

    candidates = [(i, np.log(E[i, V[0]])) for i in np.where(valid_emission_mask[:, V[0]])[0]]
    best_paths = candidates.copy()

    start = time.time()

    for t in range(1, M):
        new_candidates = {}

        for j in range(N):
            for idx, prev_score in candidates:
                if valid_transmission_mask[idx, j] and valid_emission_mask[j, V[t]]:
                    candidate_scores = np.log(T[idx, j]) + np.log(E[j, V[t]])
                    new_score = prev_score + candidate_scores

                    if j in new_candidates:
                        new_candidates[j] = max(new_candidates[j], new_score)
                    else:
                        new_candidates[j] = new_score

        candidates = [(k, v) for k, v in new_candidates.items()]
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        best_paths.extend(candidates)

    best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
    best_path = [idx for idx, _ in best_paths[:beam_width]]

    S = [StateNames[s] for s in best_path]

    end = time.time()
    print("Beam search took {} seconds".format(round(end - start, 3)))

    return S

def iterative_beam_search(V, T, E, StateNames, initial_beam_width, beam_width_reduction_factor,
                          num_iterations, min_score_change, threshold=1e-5):
    M = V.shape[0]
    N = T.shape[0]

    # Create valid masks for transmission and emission matrices
    valid_transmission_mask = T > 0.0
    valid_emission_mask = E > threshold

    best_paths_list = []

    for _ in range(num_iterations):
        # Initialize candidates with valid emission probabilities
        candidates = [(i, E[i, V[0]]) for i in range(N) if valid_emission_mask[i, V[0]]]
        best_paths = candidates.copy()
        beam_width = initial_beam_width

        for t in range(1, M):
            new_candidates = {}

            for j in range(N):
                for idx, prev_prob in candidates:
                    # Check validity of transmission and emission probabilities
                    if valid_transmission_mask[idx, j] and valid_emission_mask[j, V[t]]:
                        candidate_prob = T[idx, j] * E[j, V[t]]
                        new_prob = prev_prob * candidate_prob

                        if j in new_candidates:
                            new_candidates[j] = max(new_candidates[j], new_prob)
                        else:
                            new_candidates[j] = new_prob

            # Filter candidates based on beam width
            candidates = [(k, v) for k, v in new_candidates.items()]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            best_paths.extend(candidates)

            if beam_width > 1:
                beam_width = max(1, int(beam_width * beam_width_reduction_factor))

        # Sort and select best paths
        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        best_path = [idx for idx, _ in best_paths[:initial_beam_width]]

        best_paths_list.append(best_path)

        # Check termination criterion
        if len(best_paths_list) > 1:
            prev_alignment_score = sum([E[best_paths_list[-2][t], V[t]] for t in range(M)])
            current_alignment_score = sum([E[best_path[t], V[t]] for t in range(M)])
            score_change = current_alignment_score - prev_alignment_score
            print("Score at iteration {} is {}:".format(_, score_change))

            if score_change < min_score_change:
                print("Score at break:", score_change)
                print("Iteration count at break:", num_iterations)
                break

    # Choose the best alignment from the iterations
    final_best_path = None
    best_alignment_score = float('-inf')

    for path in best_paths_list:
        alignment_score = sum([E[path[t], V[t]] for t in range(M)])
        if alignment_score > best_alignment_score:
            final_best_path = path
            best_alignment_score = alignment_score

    S = [StateNames[s] for s in final_best_path]

    return S

def compute_maximal_cliques_from_transition_matrix(T, E, V, S):
    # Create a graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(len(S)))

    # Add edges based on transition matrix
    for i in range(len(S)):
        for j in range(len(S)):
            if T[i][j] > 0:
                G.add_edge(i, j)

    # Compute maximal cliques
    maximal_cliques = list(nx.find_cliques(G))

    # Map nodes to cliques
    node_cliques = {}
    for clique in maximal_cliques:
        for node in clique:
            if node not in node_cliques:
                node_cliques[node] = []
            node_cliques[node].append(clique)

    return node_cliques

def iterative_beam_search_with_mcq(V, T, E, StateNames, initial_beam_width, beam_width_reduction_factor,
                                   num_iterations, min_score_change, threshold=1e-5, node_cliques=None):
  # Compute maximal cliques
    node_cliques = compute_maximal_cliques_from_transition_matrix(T, E, V, StateNames)

    M = len(V)
    N = len(StateNames)

    # Create valid masks for transmission and emission matrices
    valid_transmission_mask = T > 0.0
    valid_emission_mask = E > threshold

    best_paths_list = []

    for _ in range(num_iterations):
        # Initialize candidates with valid emission probabilities
        candidates = [(i, E[i, V[0]]) for i in range(N) if valid_emission_mask[i, V[0]]]
        best_paths = candidates.copy()
        beam_width = initial_beam_width

        for t in range(1, M):
            new_candidates = {}

            for j in range(N):
                for idx, prev_prob in candidates:
                    # Check validity of transmission and emission probabilities
                    if valid_transmission_mask[idx, j] and valid_emission_mask[j, V[t]]:
                        candidate_prob = T[idx, j] * E[j, V[t]]
                        new_prob = prev_prob * candidate_prob

                        # Check if node belongs to a clique
                        if j in node_cliques:
                            # Prioritize or filter candidates based on clique information
                            clique_priority = len(node_cliques[j])
                            new_prob *= clique_priority

                        if j in new_candidates:
                            new_candidates[j] = max(new_candidates[j], new_prob)
                        else:
                            new_candidates[j] = new_prob

            # Filter candidates based on beam width
            candidates = [(k, v) for k, v in new_candidates.items()]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            best_paths.extend(candidates)

            if beam_width > 1:
                beam_width = max(1, int(beam_width * beam_width_reduction_factor))

        # Sort and select best paths
        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        best_path = [idx for idx, _ in best_paths[:initial_beam_width]]

        best_paths_list.append(best_path)

        # Check termination criterion
        if len(best_paths_list) > 1:
            prev_alignment_score = sum([E[best_paths_list[-2][t], V[t]] for t in range(M)])
            current_alignment_score = sum([E[best_path[t], V[t]] for t in range(M)])
            score_change = current_alignment_score - prev_alignment_score
            print("Score at iteration {} is {}:".format(_, score_change))

            if score_change < min_score_change:
                print("Score at break:", score_change)
                print("Iteration count at break:", num_iterations)
                break

    # Choose the best alignment from the iterations
    final_best_path = None
    best_alignment_score = float('-inf')

    for path in best_paths_list:
        alignment_score = sum([E[path[t], V[t]] for t in range(M)])
        if alignment_score > best_alignment_score:
            final_best_path = path
            best_alignment_score = alignment_score

    S = [StateNames[s] for s in final_best_path]

    return S

def compute_angle_principal(node_index, neighbors_indices, principal_directions):
    if len(neighbors_indices) < 2:
        return 0  # No angle can be defined
    angles = []
    node_direction = principal_directions[node_index]
    for i in range(len(neighbors_indices)):
        for j in range(i + 1, len(neighbors_indices)):
            vec1 = principal_directions[neighbors_indices[i]]
            vec2 = principal_directions[neighbors_indices[j]]
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angles.append(angle)
    return np.mean(angles) if angles else 0

