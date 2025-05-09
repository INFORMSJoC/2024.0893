"""
Module create groups that corresponds to the score and information gain of the orienteering problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scoring
import sensing
import uav_initializer as uav_initializer

module_logger = logging.getLogger(__name__)


@dataclass
class Group:
    scenario_set: set[int]
    group_weight: float

    def __iter__(self):
        return iter(self.scenario_set)


def generate_L_v(
    location: int, compatible_scenarios: set, sensing_matrix: npt.NDArray[np.int_]
) -> set[int]:
    """_summary_

    Parameters
    ----------
    location : int
        Index of sensing location
    compatible_scenarios : set
        Set of compatible scenarios
    sensing_matrix : npt.NDArray[np.int_]
        Sensing matrix of the problem

    Returns
    -------
    set[int]
        The complement of the largest part.
    """
    E_bar_yes = sensing.find_eliminated_scenarios(
        [sensing.Observation(location, True)], compatible_scenarios, sensing_matrix
    )
    E_bar_no = sensing.find_eliminated_scenarios(
        [sensing.Observation(location, False)], compatible_scenarios, sensing_matrix
    )

    if len(E_bar_yes) <= len(E_bar_no):
        return E_bar_yes
    else:
        return E_bar_no


def create_info_gain_groups(
    location_set: set[int],
    compatible_scenarios: set[int],
    probability_distribution: dict[int, float],
    sensing_matrix: npt.NDArray[np.int_],
) -> dict[int, Group]:
    """Create groups corresponding to information gain term of the score.

    Parameters
    ----------
    location_set : set[int]
        Set of all possible locations.
    compatible_scenarios : set[int]
        Set of compatible scenarios.
    probability_distribution : dict[int, float]
        Porbability distribution of the scenarios.
    sensing_matrix : npt.NDArray[np.int_]
        Sensing matrix, where M[u, v] = 1 if location u can sense location v.

    Returns
    -------
    dict[int, Group]
        Mapping of information gain index to the group.
    """
    groups = {
        scenario: Group(set(), probability_distribution[scenario])
        for scenario in compatible_scenarios
    }
    for location in location_set:
        if location >= len(sensing_matrix):
            continue
        L_v = generate_L_v(location, compatible_scenarios, sensing_matrix)

        for scenario in L_v:
            groups[scenario].scenario_set.add(location)

    return groups


def create_functional_gain_groups(
    location_set: set[int],
    compatible_scenarios: set[int],
    probability_distribution: dict[int, float],
    sensing_matrix: npt.NDArray[np.int_],
) -> dict[tuple[int, int], Group]:
    """
    Parameters
    ----------
    location_set : set
        Set of locations.
    compatible_scenarios : set
        Set of compatible scenarios.
    sensing_matrix : np.array
        Sensing matrix of the problem.

    Returns
    -------
    dict
        dict of groups.
    """
    groups = {}
    Q_bar = scoring.compute_Q_bar(compatible_scenarios)
    for omega in compatible_scenarios:
        for theta in compatible_scenarios:
            if omega > theta:
                omega_neighbors = sensing.generate_sensing_locations(
                    omega, sensing_matrix
                )
                theta_neighbors = sensing.generate_sensing_locations(
                    theta, sensing_matrix
                )
                items_ = omega_neighbors.symmetric_difference(theta_neighbors)
                items_ = items_ & location_set
                probability = 2 * probability_distribution[omega] / Q_bar
                groups[(omega, theta)] = Group(items_, probability)

    return groups


def create_groups(
    location_set: set[int],
    compatible_scenarios: set[int],
    probability_distribution: dict[int, float],
    sensing_matrix: npt.NDArray[np.int_],
) -> list[Group]:
    """
    Parameters
    ----------
    location_set : set
        Set of locations.
    compatible_scenarios : set
        Set of compatible scenarios.
    sensing_matrix : np.array
        Sensing matrix of the problem.

    Returns
    -------
    list
        List of groups.
    """

    info_gain_groups = create_info_gain_groups(
        location_set, compatible_scenarios, probability_distribution, sensing_matrix
    )
    functional_gain_groups = create_functional_gain_groups(
        location_set, compatible_scenarios, probability_distribution, sensing_matrix
    )
    groups = [*info_gain_groups.values(), *functional_gain_groups.values()]
    assert len(groups) == len(info_gain_groups) + len(functional_gain_groups)
    return groups


if __name__ == "__main__":
    print("start")
    n = 8
    sensing_matrix = uav_initializer.create_uav_sensing_matrix(n)
    obs1 = sensing.Observation(0, True)
    partial_observations = [obs1]
    compatible_scenario = set(range(n**2, 2 * n**2))
    probability_distribution = {
        i: 1 / len(compatible_scenario) for i in range(n**2, 2 * n**2)
    }
    assert len(compatible_scenario) == 64
    assert len(probability_distribution) == 64
    assert sum(probability_distribution.values()) == 1
    print(
        create_info_gain_groups(
            set(range(2 * n**2)),
            compatible_scenario,
            probability_distribution,
            sensing_matrix,
        )
    )
    relative_function_gain_groups = create_functional_gain_groups(
        set(range(2 * n**2)),
        compatible_scenario,
        probability_distribution,
        sensing_matrix,
    )
    print(relative_function_gain_groups)
    print(len(relative_function_gain_groups))
    print(sum([group.group_weight for group in relative_function_gain_groups.values()]))
    print(
        create_groups(
            set(range(2 * n**2)),
            compatible_scenario,
            probability_distribution,
            sensing_matrix,
        )
    )
