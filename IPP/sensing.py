import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    location: int
    signal: bool


def generate_sensed_locations(v: int, sensing_matrix: npt.NDArray) -> set[int]:
    """Return the set of locations that can be sensed by the given vertex.

    Args:
        v (int): The vertex to check.
        sensing_matrix (npt.NDArray): The sensing matrix.

    Returns:
        set: The set of locations that can be sensed by the given vertex.
    """
    return set(np.where(sensing_matrix[v])[0])


def generate_sensing_locations(scn: int, sensing_matrix: npt.NDArray) -> set[int]:
    """Return the set of locations that returns true when the scenario is omega.

    Args:
        scn (int): Realized scenario
        sensing_matrix (npt.NDArray): _description_

    Returns:
        set: The set of locations that returns true when the scenario is omega.
    """
    return set(np.where(sensing_matrix[:, scn])[0])


def generate_partial_realization(
    realized_scenario: int,
    visited_locations: Iterable[int],
    sensing_matrix: npt.NDArray,
) -> list[Observation]:
    """Generate the partial realization of the scenario.

    Args:
        realized_scenario (int): The scenario that has been realized.
        visited_locations (Iterable[int]): The locations that have been visited.
        sensing_matrix (npt.NDArray): The sensing matrix.

    Returns:
        list: The partial realization of the scenario.
    """
    true_locations = generate_sensing_locations(realized_scenario, sensing_matrix)
    return [
        Observation(
            location=location,
            signal=(location in true_locations),
        )
        for location in visited_locations
    ]


def find_eliminated_scenarios(
    partial_observations: list,
    compatible_scenarios: set[int],
    sensing_matrix: npt.NDArray,
) -> set[int]:
    """Find the set of scenarios that are eliminated by the partial observation.

    Args:
        partial_observations (list): The partial observation.
        compatible_scenarios (set[int]): The set of compatible scenarios.
        sensing_matrix (npt.NDArray): The sensing matrix.

    Returns:
        set: The set of scenarios that are eliminated by the partial observation.
    """
    eliminated = set()
    for observation in partial_observations:
        if observation.signal:
            eliminated_curr = compatible_scenarios - generate_sensed_locations(
                observation.location, sensing_matrix
            )
        else:
            eliminated_curr = generate_sensed_locations(
                observation.location, sensing_matrix
            ).intersection(compatible_scenarios)

        eliminated = eliminated.union(eliminated_curr)
    return eliminated
