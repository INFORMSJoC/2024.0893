import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def create_star_sensing_matrix(bits: int) -> npt.NDArray[np.bool_]:
    """Create the sensing matrix of the star graph.

    Args:
        bits (int): Number of bits in the binary representation of a number.

    Returns:
        npt.NDArray[np.bool_]: The sensing matrix of the star graph.
        where S[i, j] is True if the ith node can sense the jth scenario.
    """
    max_int = 2**bits
    b_node_matrix = np.full((bits, max_int), False)
    for i in range(max_int):
        binary_representation = bin(i)[2:].zfill(bits)

        row = list(reversed([bool(int(x)) for x in list(binary_representation)]))

        b_node_matrix[:, i] = row
    sensing_matrix = np.block(
        [[np.eye(max_int, dtype=bool)], [b_node_matrix]],
    )
    return sensing_matrix


def create_star_adjacency_matrix(bits: int, d) -> npt.NDArray[np.int_]:
    """Create the adjacency matrix of the star graph.

    Args:
        bits (_type_): Number of bits in the binary representation of a number.
        d (_type_): Distance between the center of the star graph.

    Returns:
        npt.NDArray[np.int_]: The adjacency matrix of the star graph.
    """
    s_index = np.arange(2**bits)
    b_index = np.arange(2**bits, 2**bits + bits)
    s_root_index = 2**bits + bits + 1
    b_root_index = 2**bits + bits
    adjacency_matrix = np.zeros((s_root_index + 1, s_root_index + 1), dtype=int)
    adjacency_matrix[s_root_index, s_index] = 1
    adjacency_matrix[s_index, s_root_index] = 1
    adjacency_matrix[b_root_index, b_index] = 1
    adjacency_matrix[b_index, b_root_index] = 1
    adjacency_matrix[s_root_index, b_root_index] = d
    adjacency_matrix[b_root_index, s_root_index] = d

    return adjacency_matrix


if __name__ == "__main__":
    print(create_star_sensing_matrix(3))
    print(create_star_adjacency_matrix(3, 10))
