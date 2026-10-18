# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import unittest

from vllm._mwf_candidate_index import CandidateKind, MWFCandidateIndex


class TestMWFCandidateIndexBindings(unittest.TestCase):
    """Exercise the installed native binding through its Python public API."""

    def test_keyword_defaults_and_python_return_types(self):
        index = MWFCandidateIndex(alpha=1.0, beta=2.0)
        self.assertIsNone(index.add(block_id=7, kind=CandidateKind.UNHASHED))
        self.assertIs(type(index.size()), int)
        self.assertEqual(index.size(), 1)
        self.assertIs(index.update(block_id=7, kind=CandidateKind.TRACKED), True)
        self.assertIs(index.update(block_id=99, kind=CandidateKind.MISSING), False)
        ids = index.pop_n(num_blocks=1, elapsed=0.0)
        self.assertIs(type(ids), list)
        self.assertEqual(ids, [7])
        self.assertIs(type(ids[0]), int)
        self.assertIs(index.erase(block_id=7), False)
        index.add(8, CandidateKind.MISSING)
        self.assertIs(index.erase(block_id=8), True)
        index.add(9, CandidateKind.TRACKED)
        self.assertIsNone(index.clear())
        self.assertEqual(index.size(), 0)
        self.assertEqual(index.pop_n(num_blocks=0, elapsed=0.0), [])

    def test_elapsed_changes_the_winning_score_branch(self):
        for elapsed, expected in ((0.0, [1, 0]), (4.0, [0, 1])):
            with self.subTest(elapsed=elapsed):
                index = MWFCandidateIndex(1.0, 2.0)
                index.add(0, CandidateKind.TRACKED, key_a=0.0, key_b=10.0)
                index.add(1, CandidateKind.TRACKED, key_a=10.0, key_b=-1.0)
                self.assertEqual(index.pop_n(2, elapsed), expected)

    def test_enum_kinds_free_order_and_score_ties(self):
        index = MWFCandidateIndex(1.0, 2.0)
        index.add(9, CandidateKind.TRACKED, 0.0, 2.0)
        index.add(3, CandidateKind.TRACKED, 2.0, 0.0)
        index.add(1, CandidateKind.TRACKED, 0.0, 0.0)
        index.add(8, CandidateKind.MISSING)
        index.add(7, CandidateKind.UNHASHED)
        index.add(6, CandidateKind.UNHASHED)
        index.add(5, CandidateKind.MISSING)
        self.assertEqual(index.pop_n(7, 0.0), [7, 6, 8, 5, 1, 3, 9])

    def test_update_and_erase_remove_both_score_keys(self):
        index = MWFCandidateIndex(1.0, 2.0)
        index.add(0, CandidateKind.TRACKED, -10.0, -20.0)
        index.add(1, CandidateKind.TRACKED, 0.0, 0.0)
        self.assertIs(
            index.update(0, CandidateKind.TRACKED, key_a=10.0, key_b=20.0), True
        )
        self.assertEqual(index.pop_n(1, 0.0), [1])
        self.assertIs(index.erase(0), True)
        index.add(2, CandidateKind.TRACKED, 1.0, 1.0)
        self.assertEqual(index.pop_n(1, 0.0), [2])
        self.assertEqual(index.size(), 0)

    def test_cpp_invalid_argument_maps_to_value_error(self):
        with self.assertRaisesRegex(ValueError, "alpha and beta"):
            MWFCandidateIndex(alpha=0.0, beta=1.0)
        index = MWFCandidateIndex(1.0, 2.0)
        index.add(0, CandidateKind.TRACKED)
        with self.assertRaisesRegex(ValueError, "already a free candidate"):
            index.add(0, CandidateKind.TRACKED)
        with self.assertRaises(ValueError):
            index.add(-1, CandidateKind.UNHASHED)
        with self.assertRaises(ValueError):
            index.add(1, CandidateKind.TRACKED, key_a=math.inf)
        with self.assertRaises(ValueError):
            index.update(0, CandidateKind.TRACKED, key_b=math.nan)
        with self.assertRaises(ValueError):
            index.pop_n(num_blocks=2, elapsed=0.0)
        with self.assertRaises(ValueError):
            index.pop_n(num_blocks=1, elapsed=-1.0)
        self.assertEqual(index.size(), 1)
        self.assertEqual(index.pop_n(1, 0.0), [0])
