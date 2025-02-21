import networkx as nx
import numpy as np
import community  # Louvain community detection algorithm

def maximal_cliques(graph):
    return list(nx.find_cliques(graph))

def MaximalClique( correspondence_pairs, distance_pairs):
      # Convert the NumPy array to a list of tuples
    correspondence_pairs_list = [tuple(pair) for pair in correspondence_pairs]
    normalized_dist = (distance_pairs - np.min(distance_pairs))/(np.max(distance_pairs)-np.min(distance_pairs))

    # Create a graph
    G = nx.Graph()
    for i in range(correspondence_pairs.shape[0]):
        u, v = correspondence_pairs_list[i]
        distance = normalized_dist[i]
        G.add_edge(u, v, weight=distance)

    # Find maximal cliques using Bron-Kerbosch
    maximal_cliques = list(nx.find_cliques(G))

    print("Maximal cliques:")
    for clique in maximal_cliques:
        print(clique)

    return np.array(maximal_cliques)


def MaximalClique_new(correspondence_pairs, distance_pairs):

    correspondence_pairs_list = [tuple(pair) for pair in correspondence_pairs]
    normalized_dist = (distance_pairs - np.min(distance_pairs)) / (np.max(distance_pairs) - np.min(distance_pairs))
    G = nx.Graph()
    for i in range(correspondence_pairs.shape[0]):
        u, v = correspondence_pairs_list[i]
        distance = normalized_dist[i]  # Use distance as the edge weight
        G.add_edge(u, v, weight=distance)  # Use distance as the edge weight

    maximal_cliques = []

    def bron_kerbosch(R, P, X):
        if not P and not X:
            maximal_cliques.append(R)
            return

        pivot = max(P.union(X), key=lambda node: len(list(G.neighbors(node))))

        for node in list(P - set(G.neighbors(pivot))):
            new_R = R.union([node])
            new_P = P.intersection(set(G.neighbors(node)))
            new_X = X.intersection(set(G.neighbors(node)))
            bron_kerbosch(new_R, new_P, new_X)

            P.remove(node)
            X.add(node)

    bron_kerbosch(set(), set(G.nodes()), set())

    filtered_cliques = [clique for clique in maximal_cliques if len(clique) == len(set(clique))]

      # Convert filtered_cliques list of sets to a NumPy array
    result_array = np.array([list(clique) for clique in filtered_cliques])

    # print("Maximal cliques:")
    # np_res_array = np.zeros((len(filtered_cliques),2), dtype=int)
    #
    # for i in range(len(filtered_cliques)):
    #     print(filtered_cliques[i])
    #     np_res_array[i][0] = filtered_cliques[i][0]
    #     np_res_array[i][1] = filtered_cliques[i][1]

    return result_array

def maximal_clique_community(correspondence_pairs, distance_pairs, threshold):
    correspondence_pairs_list = [tuple(pair) for pair in correspondence_pairs]
    normalized_dist = (distance_pairs - np.min(distance_pairs)) / (np.max(distance_pairs) - np.min(distance_pairs))

    # Create a graph
    G = nx.Graph()
    for i, pair in enumerate(correspondence_pairs_list):
        u, v = pair
        distance = normalized_dist[i]  # Use distance as the edge weight
        if distance <= threshold:  # Check if distance is within threshold
            G.add_edge(u, v, weight=distance, correspondence=pair)  # Include correspondence pair as an attribute

    # Find maximal cliques using Bron-Kerbosch algorithm
    cliques = list(nx.find_cliques(G))

    # Filter out cliques with fewer nodes than min_clique_size
    cliques = [clique for clique in cliques if len(clique) >= 2]

    # Extract unique correspondence pairs from the maximal cliques
    result_set = set()  # Use a set to avoid duplicate correspondence pairs
    for clique in cliques:
        for u in clique:
            for v in clique:
                if u < v:
                    result_set.add(G[u][v]['correspondence'])

    result_array = np.array(list(result_set))

    # Check if the result_array is empty
    if result_array.size == 0:
        print("Result array is empty. No maximal cliques found.")
        return None

    # Check shape of the result array and reshape if necessary
    if result_array.ndim != 2 or result_array.shape[1] != 2:
        print("Error: Result array shape is not (n, 2)")
        return None

    return result_array


