import unittest

import numpy as np

import solver


class TestExhaustiveProb(unittest.TestCase):

    def test_invalid(self):
        board = np.zeros((1, 2))
        self.assertIsNone(solver.permutate_board(board, {1: 2}))

    def test_basic(self):
        board = np.zeros((1, 3))
        self.assertEqual(solver.permutate_board(board, {1: 2}).all(), np.array([2,0,2]).all())

    def test_example(self):
        board = np.zeros((3, 3))
        self.assertEqual(solver.permutate_board(board, {3: 1, 1: 2}).all(), np.array([2,0,2]).all())


if __name__ == '__main__':
    unittest.main()