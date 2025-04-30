import logging
import time

import numpy.typing as npt
from compute_tour_cost_on_grid import compute_tour_cost_on_grid

module_logger = logging.getLogger(__name__)

import k_rounds_adaptive_iteration
import sensing
from sensing import Observation


def k_rounds_adaptive_identification(
    n: int,
    k: int,
    visited,
    partial_realization,
    locations,
    compatible_scenarios,
    probability_distribution,
    realized_scenario: int,
    distance_matrix: npt.NDArray,
    sensing_matrix: npt.NDArray,
    terminate_tour_early: bool = False,
    first_tour: list[list[int]] | None = None,
) -> tuple[list[int], list[Observation], int, int, int, float, list[int]]:
    """Runs the k-round identification algorithm.

    Parameters
    ----------
    n : int
        Size of the grid. (Length or breadth). The grid is assumed to be square.
    k : int
        Number of rounds of adaptivity allowed.
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
        Scenario found by the algorithm. Used to verify the correctness of the algorithm.
    grand_tour_cost : int
        Cost of the grand tour.
    rounds_taken : int
        Number of rounds taken by the algorithm.
    time_taken : float
        Time taken by the algorithm (in seconds).
    Planning_time_record : list[int]
        List of planning times for each iteration.

    Raises
    ------
    RuntimeError
        Algorithm did not identify the scenario. Caused by internal bugs
    """
    round = None
    planning_time_record = []
    start_time = time.process_time_ns()
    assert first_tour is not None
    for round in range(k, 0, -1):
        module_logger.info(f"Starting round {round}")
        delta = len(compatible_scenarios) ** (-1 / round)
        module_logger.info(
            f"Delta: {delta}, m = {len(compatible_scenarios)}, delta*m = {delta*len(compatible_scenarios)}"
        )
        assert round != k or first_tour is not None
        (
            tour_during_iteration,
            realization_during_iteration,
            compatible_scenarios_during_iteration,
            planning_time,
        ) = k_rounds_adaptive_iteration.partial_covering_algorithm(
            realized_scenario,
            visited,
            partial_realization,
            compatible_scenarios,
            probability_distribution,
            locations,
            delta,
            distance_matrix,
            sensing_matrix,
            terminate_tour_early,
            first_tour,
        )
        first_tour = None
        planning_time_record.append(planning_time)
        visited += tour_during_iteration
        module_logger.info(f"Visited after round {round}: {visited}")

        partial_realization += realization_during_iteration
        module_logger.info(
            f"Partial realization after round {round}: {partial_realization}"
        )

        compatible_scenarios = compatible_scenarios_during_iteration
        module_logger.info(
            f"Compatible scenarios after round {round}: {compatible_scenarios}"
        )
        if len(compatible_scenarios) <= 1:
            break

    grand_tour_cost = compute_tour_cost_on_grid(visited, distance_matrix, n - 1)

    assert len(compatible_scenarios) == 1
    item_ = compatible_scenarios.pop()
    assert item_ == realized_scenario
    assert partial_realization == sensing.generate_partial_realization(
        realized_scenario, visited, sensing_matrix
    )
    assert round is not None
    rounds_taken = k - round + 1
    end_time = time.process_time_ns()
    time_taken = end_time - start_time
    return (
        visited,
        partial_realization,
        item_,
        grand_tour_cost,
        rounds_taken,
        time_taken,
        planning_time_record,
    )
