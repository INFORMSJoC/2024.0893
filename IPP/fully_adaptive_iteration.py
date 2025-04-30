"""
Module runs an iteration of the fully adaptive algorithm.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import grouping
import numpy.typing as npt
import sensing
from orienteering import orienteer

if TYPE_CHECKING:
    from grouping import Group
    from sensing import Observation


module_logger = logging.getLogger(__name__)


def fully_adaptive_iteration(
    realized_scenario: int,
    visited: list[int],
    partial_realization: list[Observation],
    compatible_scenarios: set[int],
    probability_distribution: dict[int, float],
    locations: set[int],
    distance_matrix: npt.NDArray,
    sensing_matrix: npt.NDArray,
) -> tuple[list, list, set, int]:
    """Run an iteration of the fully adaptive algorithm.

    Parameters
    ----------
    realized_scenario : int
        Underlying scenario that the algorithm must find.
    visited : list[int]
        Set of nodes visited so far.
    partial_realization : list[Observation]
        Partial realization given the realized scearnio.
    compatible_scenarios : set[int]
        Set of compatible scenarios.
    probability_distribution : dict[int, float]
        Prior probability distribution over the scenarios.
    locations : set[int]
        Set of all locations.
    distance_matrix : npt.NDArray
        Distance matrix giving the graph distance between each pair of nodes.
    sensing_matrix : npt.NDArray
        Sensing matrix giving the observations that each node can make.

    Returns
    -------
    visited_new : list[int]
        List of visited nodes after the iteration.
    partial_realization_new : list[Observation]
        List of partial realization after the iteration.
    compatible_scenarios_new : set[int]
        Set of compatible scenarios after the iteration.
    planning_time : int
        Time taken during the planning phase.
    """
    # Construct the tour to find the next nodes to visit.
    assert len(visited) == len(partial_realization)
    module_logger.info(f"Visited: {visited}")
    module_logger.info(f"Compatible scenarios: {compatible_scenarios}")
    module_logger.info(f"Partial realization: {partial_realization}")

    # Copy iterates to avoid modifying the original iterates.
    visited_new = visited.copy()
    compatible_scenarios_new = compatible_scenarios.copy()
    partial_realization_new = partial_realization.copy()

    planning_start_time = time.process_time_ns()
    groups = grouping.create_groups(
        locations,
        compatible_scenarios,
        probability_distribution,
        sensing_matrix,
    )

    tour_nodes = orienteer(
        groups,
        locations,
        distance_matrix,
    )

    module_logger.info(f"Tour nodes: {tour_nodes}")
    planning_end_time = time.process_time_ns()
    planning_time = planning_end_time - planning_start_time
    tour_nodes_without_root = tour_nodes[1:-1]

    # Visiting phase of the algprithm
    for node in tour_nodes_without_root:
        next_node_to_add = [node]
        partial_realization_curr = sensing.generate_partial_realization(
            realized_scenario, next_node_to_add, sensing_matrix
        )
        # Find the scenarios that are eliminated by the new observations.
        eliminated_scenarios = sensing.find_eliminated_scenarios(
            partial_realization_curr, compatible_scenarios, sensing_matrix
        )
        compatible_scenarios_new = compatible_scenarios_new - eliminated_scenarios

        visited_new = visited_new + next_node_to_add
        partial_realization_new = partial_realization_new + (partial_realization_curr)
        if len(compatible_scenarios_new) == 1:
            break
    assert partial_realization_new == sensing.generate_partial_realization(
        realized_scenario, visited_new, sensing_matrix
    )
    return visited_new, partial_realization_new, compatible_scenarios_new, planning_time
