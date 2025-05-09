import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../IPP")
import problem_runner


class TestStarIdentificationTermination:
    def test_termination_4_fully_adaptive(self):
        problem_runner.run_experiment("star", 1, bits=5, d=10)

    def test_termination_4_non_adaptive(self):
        problem_runner.run_experiment("star", 2, bits=5, d=10)

    def test_termination_4_2_adaptive(self):
        problem_runner.run_experiment("star", 3, bits=5, d=10)
