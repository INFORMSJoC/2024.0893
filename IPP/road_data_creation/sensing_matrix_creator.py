import numpy as np
import networkx as nx


def depth_first_search(live_edge_graph, seed_node):
    """Run a depth first search on the graph to find the set of nodes
    that are infected.

    Args:
        live_edge_graph (_type_): _description_
        seed_node (_type_): _description_
    """
    visited = set()
    frontier = [seed_node]
    while frontier:
        node = frontier.pop()
        if node not in visited:
            visited.add(node)
            frontier.extend(list(np.where(live_edge_graph[node])[0]))
    return visited


def independent_cascade(adjacency, p, seed_node):
    """Runs the independent cascade to generate a set of nodes
    infected

    Args:
        adjacency (_type_): _description_
        p (_type_): _description_
        seed (_type_): _description_
    """
    random_matrix = np.random.random(adjacency.shape)

    live_edge_graph = random_matrix < p * adjacency
    infected = depth_first_search(live_edge_graph, seed_node)
    return infected


def create_sensing_matrix(adjacency, p, number_of_seed_nodes):
    # We deduct 1 from adjacency.shape because we leave the last node as the root
    # node
    seed_nodes = np.random.choice(adjacency.shape[0] - 1, number_of_seed_nodes)
    sensing_matrix = np.zeros((adjacency.shape[0], number_of_seed_nodes), dtype=bool)
    for i, seed_node in enumerate(seed_nodes):
        infected = independent_cascade(adjacency, p, seed_node)
        sensing_matrix[list(infected), i] = True
        assert sensing_matrix[:, i].sum() == len(infected)
    sensing_matrix = np.delete(sensing_matrix, -1, axis=0)
    assert sensing_matrix.shape[0] == adjacency.shape[0] - 1
    return sensing_matrix


if __name__ == "__main__":
    adjacency_distance = np.load("data/road_data/adjacency_matrix.npy")
    adjacency_matrix = adjacency_distance > 0
    # Reasonable range: 50-70?
    p = 68
    scns = 100
    sensing_matrix = create_sensing_matrix(adjacency_matrix, p / 100, scns)
    print(sensing_matrix.sum(axis=1))
    print(sensing_matrix.sum(axis=0))
    np.save(f"data/road_data/sensing_matrix_{p}_{scns}.npy", sensing_matrix)
