import logging
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import star_initializer
import uav_initializer
import road_initializer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

dir_path = os.path.dirname(os.path.dirname(__file__))
logs_path = os.path.join(dir_path, "logs")

logger = logging.getLogger(__name__)


def create_distance_matrix(adjacency: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Create the distance matrix where the i, j entry is the straight line distance
    between node i and node j.

    Args:
        adjacency (npt.NDArray): Distance between adjacent nodes of the graph.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Distance matrix between all pairs of nodes.
        Parent pointers
    """
    graph = csr_matrix(adjacency)
    dist_matrix, predecessors = floyd_warshall(
        csgraph=graph, directed=False, return_predecessors=True
    )
    logger.debug(dist_matrix)
    return dist_matrix, predecessors


def initialize_data(**kwargs: Any) -> tuple[int, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Generate the data for the problem.

    Args:
        problem (string): The category of the problem.
            Could be "uav", "star" or "road".

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: _description_
    """
    problem = kwargs["problem"]
    if problem == "uav":
        # Generate the adjacency matrix of the problem
        l = kwargs["l"]
        n = 2 * l**2 + 1
        c_top = kwargs.get("c_top", 1)
        c_bot = kwargs.get("c_bot", 4)
        height = kwargs.get("height", 10)
        occlusions = kwargs.get("occlusions", [])
        logger.info(
            f"Creating UAV problem with l={l}, c_top={c_top}, c_bot={c_bot}, height={height}, occlusions={occlusions}"
        )
        adjacency = uav_initializer.create_uav_adjaceny_matrix(l, c_top, c_bot, height)
        adjacency_file = os.path.join(dir_path, "data", "adjacency.txt")
        if __debug__:
            np.savetxt(adjacency_file, adjacency, fmt="%d")

        distance_matrix, _ = create_distance_matrix(adjacency)

        distance_matrix_file = os.path.join(dir_path, "data", "distance_matrix.txt")
        if __debug__:
            np.savetxt(distance_matrix_file, distance_matrix, fmt="%d")

        # Generate the sensing matrix of the problem
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(l, occlusions)
        sensing_matrix_file = os.path.join(dir_path, "data", "sensing_matrix.txt")
        if __debug__:
            np.savetxt(sensing_matrix_file, sensing_matrix, fmt="%d")
    elif problem == "star":
        # Generate the adjacency matrix of the problem
        bits = kwargs["bits"]
        d = kwargs["d"]
        n = 2**bits + bits + 2
        adjacency = star_initializer.create_star_adjacency_matrix(bits, d)
        adjacency_file = os.path.join(dir_path, "data", "adjacency.txt")
        if __debug__:
            np.savetxt(adjacency_file, adjacency, fmt="%d")

        distance_matrix, _ = create_distance_matrix(adjacency)
        assert distance_matrix.shape == (n, n), "Distance matrix is of the wrong size."
        distance_matrix_file = os.path.join(dir_path, "data", "distance_matrix.txt")
        if __debug__:
            np.savetxt(distance_matrix_file, distance_matrix, fmt="%d")

        # Generate the sensing matrix of the problem
        sensing_matrix = star_initializer.create_star_sensing_matrix(bits)
        sensing_matrix_file = os.path.join(dir_path, "data", "sensing_matrix.txt")
        assert sensing_matrix.shape == (
            n - 2,
            2**bits,
        ), f"Sensing matrix is of the wrong size, {sensing_matrix.shape} != {(n-2, 2**bits)}"
        if __debug__:
            np.savetxt(sensing_matrix_file, sensing_matrix, fmt="%d")
    elif problem == "road":
        file_name = kwargs["file_name"]
        adjacency = road_initializer.load_road_adjacency_matrix()
        distance_matrix = road_initializer.load_road_distance_matrix()
        sensing_matrix = road_initializer.load_road_sensing_matrix(file_name)
        n = adjacency.shape[0]
    else:
        raise ValueError(f"Problem {problem} not recognized.")

    return n, adjacency, distance_matrix, sensing_matrix


def initialize_variables(
    **kwargs: Any,
) -> tuple[list, list, set[int], set, dict[int, float]]:
    """Sets up variable for the problem at hand.

    Args:
        problem (str): String of the problem. Can be "uav", "star" or "road"

    Raises:
        ValueError: Warns that the problem is not recognized.

    Returns:
        tuple[list, list, set[int], set, dict[int, float]]: _description_
    """
    problem = kwargs["problem"]
    visited = []
    partial_realization = []
    if problem == "uav":
        l = kwargs["l"]
        N = l**2

        locations = set(np.arange(2 * N + 1))
        compatible_scenarios = set(range(N, 2 * N))
    elif problem == "star":
        bits = kwargs["bits"]
        locations = set(np.arange(2**bits + bits + 2))
        compatible_scenarios = set(range(2**bits))
    elif problem == "road":
        locations = set(
            np.arange(road_initializer.load_road_adjacency_matrix().shape[0])
        )
        compatible_scenarios = set(range(kwargs["num_scenarios"]))
    else:
        raise ValueError(f"Problem {problem} not recognized.")
    num_scenarios = len(compatible_scenarios)
    probability_distribution = {
        scenario: 1 / num_scenarios for scenario in compatible_scenarios
    }
    return (
        visited,
        partial_realization,
        locations,
        compatible_scenarios,
        probability_distribution,
    )


if __name__ == "__main__":
    n, adjacency, distance_matrix, sensing_matrix = initialize_data(
        "uav", l=8, c_top=1, c_bot=4, height=10, occlusions=[]
    )

    print(n)
    print(adjacency.shape)
    print(distance_matrix.shape)
    print(sensing_matrix.shape)
