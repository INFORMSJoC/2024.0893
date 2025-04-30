import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")

import sensing
import uav_initializer


class TestUAVSensing:
    def test_create_uav_sensing_matrix(self):
        for i in range(8, 11):
            assert uav_initializer.create_uav_sensing_matrix(i).shape == (
                2 * i**2,
                2 * i**2,
            )
            assert uav_initializer.create_uav_sensing_matrix(i).shape == (
                2 * i**2,
                2 * i**2,
            )

    def test_sensable_top_layer_corner(self):
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(8)
        assert sensing.generate_sensed_locations(0, sensing_matrix) == {
            64,
            65,
            72,
            73,
        }
        assert sensing.generate_sensed_locations(7, sensing_matrix) == {
            70,
            71,
            78,
            79,
        }
        assert sensing.generate_sensed_locations(56, sensing_matrix) == {
            112,
            113,
            120,
            121,
        }
        assert sensing.generate_sensed_locations(63, sensing_matrix) == {
            118,
            119,
            126,
            127,
        }

    def test_sensable_bottom_layer(self):
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(8)
        for i in range(64, 128):
            assert sensing.generate_sensed_locations(i, sensing_matrix) == {i}

    def test_sensing_location_corner(self):
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(8)
        assert sensing.generate_sensing_locations(64, sensing_matrix) == {
            0,
            1,
            8,
            9,
            64,
        }
        assert sensing.generate_sensing_locations(71, sensing_matrix) == {
            6,
            7,
            14,
            15,
            71,
        }
        assert sensing.generate_sensing_locations(120, sensing_matrix) == {
            48,
            49,
            56,
            57,
            120,
        }
        assert sensing.generate_sensing_locations(127, sensing_matrix) == {
            54,
            55,
            62,
            63,
            127,
        }

    def test_occlusions_single(self):
        sensing_matrix_one_out = uav_initializer.create_uav_sensing_matrix(
            8, occlusions=[0]
        )
        sensing_matrix_full = uav_initializer.create_uav_sensing_matrix(8)
        assert sensing.generate_sensed_locations(0, sensing_matrix_one_out) == set()
        assert sensing.generate_sensing_locations(64, sensing_matrix_one_out) == {
            1,
            8,
            9,
            64,
        }
        assert sensing.generate_sensed_locations(64, sensing_matrix_one_out) == {64}
        assert sensing.generate_sensed_locations(1, sensing_matrix_one_out) == {
            64,
            65,
            66,
            72,
            73,
            74,
        }
        for i in range(0, 128):
            if i != 0:
                assert sensing.generate_sensed_locations(
                    i, sensing_matrix_one_out
                ) == sensing.generate_sensed_locations(i, sensing_matrix_full)
        for i in range(64, 128):
            assert sensing.generate_sensing_locations(
                i, sensing_matrix_one_out
            ) == sensing.generate_sensing_locations(i, sensing_matrix_full) - {0}

        sensing_matrix_one_out = uav_initializer.create_uav_sensing_matrix(
            8, occlusions=[9]
        )
        assert sensing.generate_sensed_locations(9, sensing_matrix_one_out) == set()
        assert sensing.generate_sensing_locations(73, sensing_matrix_one_out) == {
            0,
            1,
            2,
            8,
            10,
            16,
            17,
            18,
            73,
        }
        assert sensing.generate_sensed_locations(64, sensing_matrix_one_out) == {64}

    def test_occlusions_multiple(self):
        occlusions = [19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39]
        sensing_matrix_chunk_removed = uav_initializer.create_uav_sensing_matrix(
            8, occlusions
        )
        sensing_matrix_full = uav_initializer.create_uav_sensing_matrix(8)
        for location in occlusions:
            assert (
                sensing.generate_sensed_locations(
                    location, sensing_matrix_chunk_removed
                )
                == set()
            )

        scenario_below_occlusions = [
            83,
            84,
            85,
            86,
            87,
            91,
            92,
            93,
            94,
            95,
            99,
            100,
            101,
            102,
            103,
        ]
        fully_blocked = [
            92,
            93,
            94,
            95,
        ]
        for scenario in fully_blocked:
            assert sensing.generate_sensing_locations(
                scenario, sensing_matrix_chunk_removed
            ) == {scenario}

        for location in range(128):
            if location not in occlusions:
                assert sensing.generate_sensed_locations(
                    location, sensing_matrix_chunk_removed
                ) == sensing.generate_sensed_locations(location, sensing_matrix_full)

        for scenario in range(64, 128):
            assert sensing.generate_sensing_locations(
                scenario, sensing_matrix_chunk_removed
            ) == sensing.generate_sensing_locations(
                scenario, sensing_matrix_full
            ) - set(
                occlusions
            )


class TestPartialRealization:
    def test_generate_partial_realization_scenario_0(self):
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(8)
        actual_true_location = {0, 1, 8, 9, 64}
        assert sensing.generate_partial_realization(
            64, range(2 * 8**2), sensing_matrix
        ) == [
            sensing.Observation(location=i, signal=i in actual_true_location)
            for i in range(2 * 8**2)
        ]

    def test_generate_partial_realization_scenario_1(self):
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(9)
        assert sensing.generate_partial_realization(82, [0], sensing_matrix) == [
            sensing.Observation(location=0, signal=True)
        ]
        assert sensing.generate_partial_realization(82, [1], sensing_matrix) == [
            sensing.Observation(location=1, signal=True)
        ]
        assert sensing.generate_partial_realization(82, [82], sensing_matrix) == [
            sensing.Observation(location=82, signal=True)
        ]
        assert sensing.generate_partial_realization(82, [83], sensing_matrix) == [
            sensing.Observation(location=83, signal=False)
        ]


class TestFindEliminatedScenarios:
    def test_corner_eliminated(self):
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(8)
        observation1 = sensing.Observation(0, True)
        compatible_scenarios = set(range(64, 128))
        assert sensing.find_eliminated_scenarios(
            [observation1], compatible_scenarios, sensing_matrix
        ) == (compatible_scenarios - {64, 65, 72, 73})
        observation2 = sensing.Observation(0, False)
        compatible_scenarios = set(range(64, 128))
        assert sensing.find_eliminated_scenarios(
            [observation2], compatible_scenarios, sensing_matrix
        ) == {64, 65, 72, 73}

    def test_identified_diagonal(self):
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(8)
        observation1 = sensing.Observation(27, True)
        observation2 = sensing.Observation(45, True)
        compatible_scenarios = set(range(64, 128))
        assert sensing.find_eliminated_scenarios(
            [observation1, observation2], compatible_scenarios, sensing_matrix
        ) == (compatible_scenarios - {100})

    def test_identified_lower(self):
        sensing_matrix = uav_initializer.create_uav_sensing_matrix(8)
        observation1 = sensing.Observation(100, True)
        compatible_scenarios = set(range(64, 128))
        assert sensing.find_eliminated_scenarios(
            [observation1], compatible_scenarios, sensing_matrix
        ) == (compatible_scenarios - {100})
