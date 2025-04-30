import logging
import time
from typing import TYPE_CHECKING

import fully_adaptive_iteration
import numpy.typing as npt
import sensing
from compute_tour_cost_on_grid import compute_tour_cost_on_grid

module_logger = logging.getLogger(__name__)


def fully_adaptive_identification(
    n: int,
    visited,
    partial_realization,
    locations,
    compatible_scenarios,
    probability_distribution,
    realized_scenario: int,
    distance_matrix: npt.NDArray,
    sensing_matrix: npt.NDArray,
) -> tuple[list[int], list, int, int, int, float, list[int]]:
    """Runs the fully adaptive identification algorithm.

    Parameters
    ----------
    n : int
        Size of the grid. (Length or breadth). The grid is assumed to be square.
    realized_scenario : int
        Underlying scenario that the algorithm wants to identify.
    distance_matrix : npt.NDArray
        Distance matrix that gives the pairwise graph distance of nodes on the grid.
    sensing_matrix : npt.NDArray
        Sensing matrix, such that M[u, v] == 1 if v can be sensed from u.

    Returns
    -------
    visited : list[int]
        List of nodes visited by the algorithm.
    partial_realization : list[Observation]
        List of observations made by the algorithm.
    item : int
        Scenario found by the algorithm.
        Used to verify the correctness of the algorithm.
    grand_tour_cost : int
        Cost of the grand tour.
    iteration : int
        Number of iterations taken by the algorithm.
    time_taken : float
        Time taken by the algorithm (in seconds).
    Planning_time_record : list[int]
        List of planning times for each iteration.
    """
    planning_time_record = []
    start_time = time.process_time_ns()
    iteration = 0
    while True:
        module_logger.info(f"Starting iteration {iteration}")
        if len(compatible_scenarios) <= 1:
            (item_,) = compatible_scenarios
            assert item_ == realized_scenario
            assert partial_realization == sensing.generate_partial_realization(
                realized_scenario, visited, sensing_matrix
            )
            assert len(visited) == len(set(visited)), "Visited nodes are unique"
            root_node = n - 1
            grand_tour_cost = compute_tour_cost_on_grid(
                visited, distance_matrix, root_node
            )
            end_time = time.process_time_ns()
            time_taken = end_time - start_time
            return (
                visited,
                partial_realization,
                item_,
                grand_tour_cost,
                iteration,
                time_taken,
                planning_time_record,
            )
        (
            visited_new,
            partial_realization_new,
            compatible_scenarios_new,
            planning_time,
        ) = fully_adaptive_iteration.fully_adaptive_iteration(
            realized_scenario,
            visited,
            partial_realization,
            compatible_scenarios,
            probability_distribution,
            locations,
            distance_matrix,
            sensing_matrix,
        )
        planning_time_record.append(planning_time)
        visited = visited_new
        partial_realization = partial_realization_new
        compatible_scenarios = compatible_scenarios_new

        iteration += 1
