"""
Module handles an iteration of the k-round-adaptive algorithm.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, Optional

import grouping
import numpy.typing as npt
import scoring
import sensing
from orienteering import orienteer

if TYPE_CHECKING:
    from sensing import Observation

module_logger = logging.getLogger(__name__)


def create_large_partition(
    visited: Iterable[int],
    compatible_scenarios: set[int],
    sensing_matrix: npt.NDArray,
    delta: Optional[float] = None,
    m: Optional[int] = None,
) -> list[set[int]]:
    """Create partitions that are larger than delta*m

    Parameters
    ----------
    visited : Iterable[int]
        Set of nodes that were visited during the current iteration.
    compatible_scenarios : set[int]
        Set of scenarios that are compatible at the start of the current iteration.
        Nodes that are currently being planned are not included, as they have yet to be visited.
    sensing_matrix : npt.NDArray
        Matrix giving us the locations that can  be observed from each node.
    delta : Optional[float], optional
        Determines the stopping condition, by default None
    m : Optional[int], optional
        The number of compatible scenarios at the start of the current iteration, by default None

    Returns
    -------
    list[set[int]]
        List of large partitions.

    Raises
    ------
    ValueError
        Ensures that both delta and m are given. If both are none, allow for unfiltered partition.
    """
    # Must specify both delta and m, or leave them out for unfiltered partition.
    if delta is None and m is not None or delta is not None and m is None:
        raise ValueError("delta and m must be both None or both not None.")
    scenario_to_partial_realization = defaultdict(set)  # type: ignore

    for scenario in compatible_scenarios:
        partial_realization_of_scenario = tuple(
            sensing.generate_partial_realization(scenario, visited, sensing_matrix)
        )
        scenario_to_partial_realization[partial_realization_of_scenario].add(scenario)

    if delta is not None and m is not None:
        delta_times_m = delta * m
    else:
        delta_times_m = -1  # -1 to allow everything through

    partition = [part for part in scenario_to_partial_realization.values()]

    module_logger.debug(f"Partition: {partition}")
    large_partition = [part for part in partition if len(part) > delta_times_m]

    assert (
        set().union(*large_partition).issubset(compatible_scenarios)
    ), "Partitions do not cover all scenarios."
    assert set().intersection(*large_partition) == set(), "Partitions are not disjoint."

    return large_partition


def partial_covering_planning(
    visited: list[int],
    compatible_scenarios: set[int],
    probability_distribution: dict[int, float],
    locations: set[int],
    delta: float,
    distance_matrix: npt.NDArray,
    sensing_matrix: npt.NDArray,
) -> list[list[int]]:
    nodes_planned = set()
    grand_tour_planned = []

    module_logger.info(f"Planning tour for {len(compatible_scenarios)} scenarios.")
    m = len(compatible_scenarios)
    while nodes_planned.union(set(visited)) != locations:
        large_partitions = create_large_partition(
            nodes_planned, compatible_scenarios, sensing_matrix, delta, m
        )

        if len(large_partitions) == 0:
            module_logger.info("No large partitions found, prematurely ending tour.")
            break
        module_logger.info(f"Large partitions: {large_partitions}")
        groups = []
        for large_part in large_partitions:
            groups = groups + grouping.create_groups(
                locations,
                large_part,
                probability_distribution,
                sensing_matrix,
            )

        tour_nodes = orienteer(groups, locations, distance_matrix)
        module_logger.info(f"Tour nodes added: {tour_nodes}")
        tour_nodes_without_roots = tour_nodes[1:-1]
        grand_tour_planned.append(tour_nodes_without_roots)
        module_logger.info(f"Grand tour planned: {grand_tour_planned}")

        nodes_planned = nodes_planned.union(set(tour_nodes_without_roots))
        module_logger.info(f"Nodes planned: {nodes_planned}")
        module_logger.info(f"Planned tour: {grand_tour_planned}")
    return grand_tour_planned


def tour_visiting(
    tour: list[int],
    compatible_scenarios_during_iteration: set[int],
    realization_during_iteration: list[Observation],
    tour_during_iteration: list,
    realized_scenario: int,
    compatible_scenarios: set[int],
    delta: float,
    sensing_matrix: npt.NDArray,
    terminate_tour_early: bool,
) -> bool:
    reached_termination_condition = False
    for node in tour:
        module_logger.info(f"Visiting node {node}")

        eliminated_scenarios = sensing.find_eliminated_scenarios(
            realization_during_iteration, compatible_scenarios, sensing_matrix
        )
        module_logger.info(f"Eliminated scenarios: {eliminated_scenarios}")

        # Updating.
        tour_during_iteration.append(node)

        (node_realization,) = sensing.generate_partial_realization(
            realized_scenario, [node], sensing_matrix
        )

        realization_during_iteration.append(node_realization)

        compatible_scenarios_during_iteration.difference_update(
            sensing.find_eliminated_scenarios(
                realization_during_iteration,
                compatible_scenarios_during_iteration,
                sensing_matrix,
            )
        )

        number_of_compatible_scenarios = len(compatible_scenarios_during_iteration)
        module_logger.info(
            f"Realization after visiting node {node}: {realization_during_iteration}"
        )
        module_logger.info(
            f"Number of compatible scenarios after visiting node {node}: {number_of_compatible_scenarios}"
        )
        module_logger.info(
            f"Compatible scenario after visiting node {node}: {compatible_scenarios_during_iteration}"
        )
        function_value = scoring.compute_f_bar(
            realization_during_iteration, compatible_scenarios, sensing_matrix
        )
        module_logger.info(
            f"Function value before visiting node {node}: {function_value}"
        )
        if function_value == len(compatible_scenarios) - 1:
            reached_termination_condition = True
            module_logger.info("Terminated because we've identified the scenario.")
            break
        elif number_of_compatible_scenarios < delta * len(compatible_scenarios):
            module_logger.info(
                "Terminated because we've eliminated sufficient scenarios."
            )
            reached_termination_condition = True
            if terminate_tour_early:
                module_logger.info("Terminating tour early.")
                break

    return reached_termination_condition


def partial_covering_visiting(
    grand_tour_planned: list[list[int]],
    realized_scenario: int,
    compatible_scenarios: set[int],
    delta: float,
    sensing_matrix: npt.NDArray,
    terminate_tour_early: bool,
) -> tuple[list[int], list[Observation], set[int]]:
    tour_during_iteration = []
    realization_during_iteration = []
    compatible_scenarios_during_iteration = compatible_scenarios.copy()

    module_logger.info("Starting visitiing phase.")
    for tour in grand_tour_planned:
        reached_termination_condition = tour_visiting(
            tour,
            compatible_scenarios_during_iteration,
            realization_during_iteration,
            tour_during_iteration,
            realized_scenario,
            compatible_scenarios,
            delta,
            sensing_matrix,
            terminate_tour_early,
        )
        if reached_termination_condition:
            break

    assert len(compatible_scenarios_during_iteration) <= delta * len(
        compatible_scenarios
    ), "path did not give a small enough partition"
    return (
        tour_during_iteration,
        realization_during_iteration,
        compatible_scenarios_during_iteration,
    )


def partial_covering_algorithm(
    realized_scenario: int,
    visited: list[int],
    partial_realization: list[Observation],
    compatible_scenarios: set[int],
    probability_distribution: dict[int, float],
    locations: set[int],
    delta: float,
    distance_matrix: npt.NDArray,
    sensing_matrix: npt.NDArray,
    terminate_tour_early: bool,
    first_tour: list[list[int]] | None,
) -> tuple[list[int], list[Observation], set[int], int | None]:
    """Implements the partial covering algorithm found in the paper.

    Parameters
    ----------
    realized_scenario : int
        Underlying scenario that the algorithm is supposed to find.
    visited : list[int]
        Set of nodes that were visited prior to the current iteration.
    partial_realization : list[Observation]
        Partial realization prior to the current iteration.
    compatible_scenarios : set[int]
        Set of compatible scenarios prior to the current iteration.
    probability_distribution : dict[int, float]
        Prior probability distribution over the scenarios.
    locations : set[int]
        Set of all locations.
    delta : float
        Determines the stopping condition.
    distance_matrix : npt.NDArray
        Gives the pairwise graph distance between any two locations.
    sensing_matrix : npt.NDArray
        Gives the scenarios that can be observed from each location.

    Returns
    -------
    tuple[list[int], list[Observation], set[int]]
        visited_new : list[int]
            Updates the visited node to include the nodes visited in the current iteration.
        partial_realization_new : list[Observation]
            Updates the partial realization to include the observations made in the current iteration.
        compatible_scenarios_new : set[int]
            Updates the compatible scenarios to remove scenarios eliminated during the current iteration.
        planning_time : int
            CPU time spent only on the planning phase.
    """
    planning_start_time = time.process_time_ns()
    if first_tour is not None:
        module_logger.info("Using first tour.")
        grand_tour_planned = first_tour
    else:
        grand_tour_planned = partial_covering_planning(
            visited,
            compatible_scenarios,
            probability_distribution,
            locations,
            delta,
            distance_matrix,
            sensing_matrix,
        )

    planning_end_time = time.process_time_ns()
    planning_time = (
        planning_end_time - planning_start_time if first_tour is None else None
    )

    (
        tour_during_iteration,
        realization_during_iteration,
        compatible_scenarios_during_iteration,
    ) = partial_covering_visiting(
        grand_tour_planned,
        realized_scenario,
        compatible_scenarios,
        delta,
        sensing_matrix,
        terminate_tour_early,
    )
    return (
        tour_during_iteration,
        realization_during_iteration,
        compatible_scenarios_during_iteration,
        planning_time,
    )
