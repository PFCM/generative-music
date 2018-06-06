"""
Tests for break_midi.py
"""
import unittest

import drums.break_midi as bm


class BinarySearchTests(unittest.TestCase):
    """check the binary search used to find split points works"""

    def test_val_exists(self):
        """binary search finds a value that exists already"""
        data = [1, 2, 3, 4, 5, 6, 7, 8]  # easy power of two
        for i, v in enumerate(data):
            self.assertEqual(bm.binary_search(v, data), i)
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for i, v in enumerate(data):
            self.assertEqual(bm.binary_search(v, data), i)

    def test_val_not_exists(self):
        """Ensure it works when the value isn't present"""
        data = [1, 2, 3, 4, 5, 6, 8, 9, 10]
        self.assertEqual(bm.binary_search(7, data), 6)
