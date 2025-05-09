import os
import sys
from unittest import mock

import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")
import FRT_tree
import numpy as np
import scoring
import sensing
import uav_initializer
from sensing import Observation


class TestScoring:
    n = 8
    c_top = 1
    c_bot = 4
    height = 10
    sensing_matrix = uav_initializer.create_uav_sensing_matrix(n)
    compatible_scenarios_full = set(range(n**2, 2 * n**2))

    @pytest.mark.parametrize(
        "partial_observations,expected_f_bar",
        [
            ([], 0),
            ([sensing.Observation(0, True)], 60),
            ([sensing.Observation(0, True), sensing.Observation(1, True)], 64 - 4),
            ([sensing.Observation(0, False), sensing.Observation(1, False)], 6),
        ],
    )
    def test_full_realization(self, partial_observations, expected_f_bar):
        assert (
            scoring.compute_f_bar(
                partial_observations,
                self.compatible_scenarios_full,
                self.sensing_matrix,
            )
            == expected_f_bar
        )
