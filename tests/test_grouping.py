import os
import sys
from typing import Any, Hashable, Literal
from unittest import mock

import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")
import grouping
import initialization
import numpy as np
import scoring
import sensing
import uav_initializer
from sensing import Observation


def group_scoring(tour: list[int], groups: dict):
    tour_set = set(tour)
    score = 0
    for group_key in groups:
        if len(tour_set & groups[group_key].scenario_set) > 0:
            score += groups[group_key].group_weight
    return score


class TestGenerateLv:
    l = 8
    adjacency_matrix = uav_initializer.create_uav_adjaceny_matrix(l, 1, 4, 10)
    dist_matrix, predecessors = initialization.create_distance_matrix(adjacency_matrix)
    locations = set(range(2 * l**2 + 1))
    compatible_scenario_full = set(range(l**2, 2 * l**2))
    sensing_matrix = uav_initializer.create_uav_sensing_matrix(l)

    def test_L_v_corner(self):
        assert grouping.generate_L_v(
            0, self.compatible_scenario_full, self.sensing_matrix
        ) == {64, 65, 72, 73}

        assert grouping.generate_L_v(
            64, self.compatible_scenario_full, self.sensing_matrix
        ) == {64}

    def test_L_v_center(self):
        assert grouping.generate_L_v(
            11, self.compatible_scenario_full, self.sensing_matrix
        ) == {66, 67, 68, 74, 75, 76, 82, 83, 84}

    def test_false_set(self):
        compatible_scenario_small = {66, 67, 68, 74, 75, 76, 82, 83, 84, 127}
        assert grouping.generate_L_v(
            11, compatible_scenario_small, self.sensing_matrix
        ) == {127}


class TestCreateInfoGainGroups:
    l = 8
    c_top = 1
    c_bot = 4
    height = 8
    n, adjacency, distance_matrix, sensing_matrix = initialization.initialize_data(
        "uav", l=l, c_top=c_top, c_bot=c_bot, height=height
    )

    (
        visited,
        partial_realization,
        locations,
        compatible_scenarios,
        probability_distribution,
    ) = initialization.initialize_variables("uav", l=l)
    info_gain_groups = grouping.create_info_gain_groups(
        locations,
        compatible_scenarios,
        probability_distribution,
        sensing_matrix,
    )

    def test_info_gain_groups_first(self):
        assert self.info_gain_groups[64].scenario_set == {0, 1, 8, 9, 64}
        assert self.info_gain_groups[64].group_weight == 1 / len(
            self.compatible_scenarios
        )

    def test_info_gain_groups_definition(self):
        for scenario in self.info_gain_groups:
            group_omega = self.info_gain_groups[scenario]
            for location in self.locations:
                if location < 128:
                    L_v = grouping.generate_L_v(
                        location, self.compatible_scenarios, self.sensing_matrix
                    )
                    if location in group_omega:
                        assert scenario in L_v
                    else:
                        assert scenario not in L_v
            assert scenario in group_omega


class TestCreateFunctionalGainGroups:
    l = 8
    c_top = 1
    c_bot = 4
    height = 8
    n, adjacency, distance_matrix, sensing_matrix = initialization.initialize_data(
        "uav", l=l, c_top=c_top, c_bot=c_bot, height=height
    )

    (
        visited,
        partial_realization,
        locations,
        compatible_scenarios,
        probability_distribution,
    ) = initialization.initialize_variables("uav", l=l)
    functional_gain_groups = grouping.create_functional_gain_groups(
        locations,
        compatible_scenarios,
        probability_distribution,
        sensing_matrix,
    )

    def test_functional_gain_group_definition(self):
        for omega, theta in self.functional_gain_groups:
            for location in self.locations:
                observation_under_omega = sensing.generate_partial_realization(
                    omega, [location], self.sensing_matrix
                )
                observation_under_theta = sensing.generate_partial_realization(
                    theta, [location], self.sensing_matrix
                )
                if location in self.functional_gain_groups[omega, theta]:
                    assert (
                        observation_under_omega[0].signal
                        != observation_under_theta[0].signal
                    )
                else:
                    assert (
                        observation_under_omega[0].signal
                        == observation_under_theta[0].signal
                    )


class TestGroupEqualFunction:
    l = 8
    c_top = 1
    c_bot = 4
    height = 8
    n, adjacency, distance_matrix, sensing_matrix = initialization.initialize_data(
        "uav", l=l, c_top=c_top, c_bot=c_bot, height=height
    )

    (
        visited,
        partial_realization,
        locations,
        compatible_scenarios,
        probability_distribution,
    ) = initialization.initialize_variables("uav", l=l)
    functional_gain_groups = grouping.create_functional_gain_groups(
        locations,
        compatible_scenarios,
        probability_distribution,
        sensing_matrix,
    )
    info_gain_groups = grouping.create_info_gain_groups(
        locations,
        compatible_scenarios,
        probability_distribution,
        sensing_matrix,
    )

    @pytest.mark.parametrize("tour,score", [([], 0), (range(2 * l**2), 1)])
    def test_information_gain_edge(self, tour: range | list[Any], score: Literal[0, 1]):
        group_score = group_scoring(tour, self.info_gain_groups)
        formula_score = scoring.compute_information_gain(
            tour,
            self.compatible_scenarios,
            self.probability_distribution,
            self.sensing_matrix,
        )
        assert group_score == formula_score == score

    @pytest.mark.parametrize("tour,score", [([], 0), (range(2 * l**2), 1)])
    def test_functional_gain_edge(self, tour: range | list[Any], score: Literal[0, 1]):
        group_score = group_scoring(tour, self.functional_gain_groups)
        formula_score = scoring.compute_relative_function_gain(
            tour,
            self.compatible_scenarios,
            self.probability_distribution,
            self.sensing_matrix,
        )
        assert group_score == pytest.approx(score)
        assert formula_score == pytest.approx(score)

    @pytest.mark.parametrize("tour", [[0], [31], [27], [75], [0, 1], [0, 31, 27, 75]])
    def test_information_gain_equal(self, tour: list[int]):
        group_score = group_scoring(tour, self.info_gain_groups)
        formula_score = scoring.compute_information_gain(
            tour,
            self.compatible_scenarios,
            self.probability_distribution,
            self.sensing_matrix,
        )
        assert group_score == pytest.approx(formula_score)

    @pytest.mark.parametrize("tour", [[0], [31], [27], [75], [0, 1], [0, 31, 27, 75]])
    def test_function_gain_equal(self, tour: list[int]):
        group_score = group_scoring(tour, self.functional_gain_groups)
        formula_score = scoring.compute_relative_function_gain(
            tour,
            self.compatible_scenarios,
            self.probability_distribution,
            self.sensing_matrix,
        )
        assert group_score == pytest.approx(formula_score)
        assert group_score == pytest.approx(formula_score)
