import unittest

import numpy as np

import solver


class TestExhaustiveProb(unittest.TestCase):

    def test_invalid(self):
        board = np.zeros((1, 2))
        pmap = solver.permutate_board(board, {1: 2})
        self.assertIsNone(pmap)

    def test_basic(self):
        board = np.zeros((1, 3))
        pmap = solver.permutate_board(board, {1: 2})
        self.assertTrue(np.array_equal(pmap, np.array([[2, 0, 2]])))

    def test_example(self):
        board = np.zeros((3, 3))
        pmap = solver.permutate_board(board, {3: 1, 1: 2})
        self.assertTrue(np.array_equal(pmap, np.array([[6, 1, 6], [1, 0, 1], [6, 1, 6]])))


if __name__ == '__main__':
    unittest.main()
