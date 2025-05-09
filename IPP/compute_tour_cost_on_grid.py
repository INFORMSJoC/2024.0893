import logging

import numpy.typing as npt

module_logger = logging.getLogger(__name__)


def compute_tour_cost_on_grid(
    grand_tour_without_source: list[int], distance_matrix: npt.NDArray, root_node: int
) -> int:
    """Given a list representing a tour without the source node,
    compute the cost of the tour.
    it is assumed that the tour returns to the root.

    Parameters
    ----------
    grand_tour_without_source : list[int]
        List of nodes in the tour without the source node.
    distance_matrix : npt.NDArray
        Distance matrix of the grid.
    root_node : int
        Index of the root node.

    Returns
    -------
    int
        Cost of the tour.
    """
    grand_tour = [root_node] + grand_tour_without_source
    module_logger.info(f"Grand tour: {grand_tour}")
    tour_cost = 0
    for i in range(len(grand_tour) - 1):
        tour_cost += distance_matrix[grand_tour[i], grand_tour[i + 1]]
    return tour_cost
