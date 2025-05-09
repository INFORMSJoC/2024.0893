import logging
from typing import Iterable

import networkx as nx
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def create_uav_sensing_matrix(n: int, occlusions: Iterable[int] = []) -> npt.NDArray:
    """Return sensing matrix with n*n*2 plots.

    Args:
        n (int): Length of one side of the square grid.
    """

    # The adjacency matrix of the "Kings" graph is the strong product of two path grpah
    path_graph = nx.path_graph(n)
    G = nx.strong_product(path_graph, path_graph)
    G = nx.adjacency_matrix(G, dtype=bool)

    G = G.toarray(order=None, out=None)

    # Can sense plot from itself
    np.fill_diagonal(G, 1)
    identity_matrix = np.eye(n**2, dtype=bool)
    zero_matrix = np.zeros((n**2, n**2), dtype=bool)

    G = np.block([[zero_matrix, G], [zero_matrix, identity_matrix]])

    for occlusion in occlusions:
        if occlusion >= n**2:
            raise ValueError(f"Occlusion {occlusion} is not in the top grid")
        G[occlusion, :] = False

    return G


# https://stackoverflow.com/questions/16329403/how-can-you-make-an-adjacency-matrix-which-would-emulate-a-2d-grid
def create_2d_grid_adjacency(rows: int, cols: int) -> npt.NDArray[np.int64]:
    """
    Generate the adjacency matrix of a 2D grid. That is, a square is adjacent to the sides
    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    Returns
    -------
    M : np.array
        Adjacency matrix of the grid.
    """
    n = rows * cols
    M = np.zeros((n, n), dtype=np.int64)
    for r in np.arange(rows):
        for c in np.arange(cols):
            i = r * cols + c
            # Two inner diagonals
            if c > 0:
                M[i - 1, i] = M[i, i - 1] = 1
            # Two outer diagonals
            if r > 0:
                M[i - cols, i] = M[i, i - cols] = 1
    return M


def create_uav_adjaceny_matrix(
    n: int, c_top: float = 1, c_bot: float = 8, height: float = 10
) -> npt.NDArray[np.int_]:
    """Create a matrix of distances between adjacent points

    Args:
        n (int): Length of the square grid
        c_top (float): Cost of travelling between adjacent nodes in the top layer.
        c_bot (float): Cost of travelling between adjacent nodes in the bottom layer.
        height (float): Cost of ascending / descending vertically.

    Returns:
        npt.NDArray[np.int_]: Adjacency matrix of the grid
    """
    M = create_2d_grid_adjacency(n, n)
    identity = np.eye(n**2, dtype=np.int_)
    adjacency = np.block(
        [[M * c_top, identity * height], [identity * height, M * c_bot]]
    )

    # Add root node that is 1 distance away from the bottom node.
    N = 2 * n**2
    zero_row = np.zeros((1, N), dtype=np.int_)
    adjacency = np.block([[adjacency, zero_row.T], [zero_row, 0]])  # type: ignore
    adjacency[N, n**2] = 1
    adjacency[n**2, N] = 1

    logger.info(f"Generated adjacency matrix of size {adjacency.shape[0]}")
    logger.debug(adjacency)
    return adjacency


def generate_neighborhood(v: int, G: npt.NDArray) -> set[int]:
    """Indices of nodes that are immediately adjacent to v.
    !!! Probably not used in the main code but used in the tests so I am keeping it.

    Args:
        v (int): node
        G (npt.NDArray): Adjacency matrix of the graph.

    Returns:
        set: Set of nodes that are adjacent to v.
    """
    return set(np.where(G[v, :])[0])
