import os
import sys
from unittest import mock

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")
import fully_adaptive_iteration
import initialization
import sensing
import uav_initializer
from sensing import Observation


class TestGeneratePartialRealization:
    board = uav_initializer.create_uav_sensing_matrix(8)

    def test_upper_corner_realization(self):
        for i in [0, 1, 8, 9]:
            assert sensing.generate_partial_realization(64, [i], self.board) == [
                Observation(i, True)
            ]
        for i in [2, 3, 4, 5, 63, 65, 127]:
            assert sensing.generate_partial_realization(64, [i], self.board) == [
                Observation(i, False)
            ]

    def test_scenario_91(self):
        sensable = {18, 19, 20, 26, 27, 28, 34, 35, 36, 91}
        correct_observation = [Observation(i, i in sensable) for i in range(128)]
        assert (
            sensing.generate_partial_realization(91, list(range(128)), self.board)
            == correct_observation
        )


class TestOrienteeringIteration:
    l = 8
    c_top = 1
    c_bot = 4
    height = 10
    # Pick a random scenario
    realized_scenario = 44
    n, adjacency, distance_matrix, sensing_matrix = initialization.initialize_data(
        "uav", l=l, c_top=c_top, c_bot=c_bot, height=height
    )

    @mock.patch("fully_adaptive_iteration.orienteer")
    def test_first_step(self, orienteering_mock):
        orienteering_mock.return_value = [128, 0, 1, 128]

        (
            visited,
            partial_realization,
            locations,
            compatible_scenarios,
            probability_distribution,
        ) = initialization.initialize_variables("uav", l=8)
        realized_scenario = 127
        (
            visited,
            partial_realization,
            compatible_scenarios,
            _,
        ) = fully_adaptive_iteration.fully_adaptive_iteration(
            realized_scenario,
            visited,
            partial_realization,
            compatible_scenarios,
            probability_distribution,
            locations,
            self.distance_matrix,
            self.sensing_matrix,
        )

        assert visited == [0, 1]
        assert partial_realization == [Observation(0, False), Observation(1, False)]
        assert compatible_scenarios == set(range(64, 128)) - {64, 65, 66, 72, 73, 74}

    @mock.patch("fully_adaptive_iteration.orienteer")
    def test_first_step_even(self, orienteering_mock):
        orienteering_mock.return_value = [128, 0, 1, 128]

        (
            visited,
            partial_realization,
            locations,
            compatible_scenarios,
            probability_distribution,
        ) = initialization.initialize_variables("uav", l=self.l)
        compatible_scenarios = set(range(64, 128, 2))
        realized_scenario = 126
        (
            visited,
            partial_realization,
            compatible_scenarios,
            _,
        ) = fully_adaptive_iteration.fully_adaptive_iteration(
            realized_scenario,
            visited,
            partial_realization,
            compatible_scenarios,
            probability_distribution,
            locations,
            self.distance_matrix,
            self.sensing_matrix,
        )

        assert visited == [0, 1]
        assert partial_realization == [Observation(0, False), Observation(1, False)]
        assert compatible_scenarios == set(range(64, 128, 2)) - {64, 66, 72, 74}

    @mock.patch("fully_adaptive_iteration.orienteer")
    def test_two_step_pass(self, orienteering_mock):
        orienteering_mock.return_value = [128, 0, 27, 128]

        (
            visited,
            partial_realization,
            locations,
            compatible_scenarios,
            probability_distribution,
        ) = initialization.initialize_variables("uav", l=self.l)
        compatible_scenarios = set(range(64, 128, 2))
        realized_scenario = 92
        (
            visited,
            partial_realization,
            compatible_scenarios,
            _,
        ) = fully_adaptive_iteration.fully_adaptive_iteration(
            realized_scenario,
            visited,
            partial_realization,
            compatible_scenarios,
            probability_distribution,
            locations,
            self.distance_matrix,
            self.sensing_matrix,
        )

        assert visited == [0, 27]
        assert partial_realization == [Observation(0, False), Observation(27, True)]
        assert compatible_scenarios == {82, 84, 90, 92, 98, 100}

        orienteering_mock.return_value = [128, 90, 92, 128]
        (
            visited,
            partial_realization,
            compatible_scenarios,
            _,
        ) = fully_adaptive_iteration.fully_adaptive_iteration(
            realized_scenario,
            visited,
            partial_realization,
            compatible_scenarios,
            probability_distribution,
            locations,
            self.distance_matrix,
            self.sensing_matrix,
        )

        assert visited == [0, 27, 90, 92]
        assert partial_realization == [
            Observation(0, False),
            Observation(27, True),
            Observation(90, False),
            Observation(92, True),
        ]
        assert compatible_scenarios == {92}
        assert compatible_scenarios == {92}
