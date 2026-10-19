# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm._mwf_candidate_index import CandidateKind, MWFCandidateIndex
from vllm.v1.core.mwf_eviction import MWFEvictionManager


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


class TestMWFNativeRegistration(unittest.TestCase):
    """Exercise real native registration with minimal block-attribute fixtures."""

    def setUp(self):
        self.clock = Mock(return_value=1000.0)
        self.params = dict(
            minexp_a=-3.0, minexp_b=4.0, minexp_alpha=17.0, minexp_beta=29.0
        )
        self.manager = MWFEvictionManager(clock=self.clock, **self.params)

    def block(self, block_id, block_hash=1, ref_cnt=0, is_null=False):
        return SimpleNamespace(
            block_id=block_id, block_hash=block_hash, ref_cnt=ref_cnt, is_null=is_null
        )

    def invalid_blocks(self):
        return [self.block(2, ref_cnt=count) for count in (1, -1)] + [
            self.block(2, is_null=True),
            *(self.block(block_id) for block_id in (-1, True, 1.0, "1")),
        ]

    def test_three_kinds_and_tracked_scores_determine_native_order(self):
        manager = self.manager
        manager.on_block_cached(30, 20000)
        self.clock.return_value = 1010.0
        manager.on_block_cached(20, 64)
        manager.on_block_cached(10, 256)
        tracked = sorted(
            (30, 20), key=lambda bid: manager.get_score(bid, now=1020).weight
        )
        blocks = [
            self.block(30),
            self.block(8, 0),
            self.block(10, None),
            self.block(20),
            self.block(11, None),
        ]
        snapshots = [vars(block).copy() for block in blocks]
        self.clock.reset_mock()
        self.clock.side_effect = AssertionError("Free registration must not read clock")
        for block in blocks:
            manager.on_block_freed(block)
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertEqual(manager._index.size(), 5)
        self.assertEqual(manager._index.pop_n(5, 20.0), [10, 11, 8, *tracked])
        self.assertEqual([vars(block) for block in blocks], snapshots)
        self.clock.assert_not_called()

    def test_duplicate_registration_preserves_original_candidate_and_record(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        block = self.block(1, 0)
        manager.on_block_freed(block)
        keys, score = manager.get_score_keys(1), manager.get_score(1, now=1020)
        with self.assertRaises(ValueError):
            manager.on_block_freed(block)
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertEqual(manager._index.size(), 1)
        self.assertEqual(manager.get_score_keys(1), keys)
        self.assertEqual(manager.get_score(1, now=1020), score)
        self.assertEqual(manager._index.pop_n(1, 20.0), [1])

    def test_invalid_free_or_keys_are_rejected_before_native_loading(self):
        for block in self.invalid_blocks():
            with self.subTest(block=vars(block)):
                with patch("builtins.__import__", side_effect=AssertionError):
                    with self.assertRaises(ValueError):
                        self.manager.on_block_freed(block)
                self.assertIsNone(self.manager._index)
                self.assertIsNone(self.manager._kind)
        manager = MWFEvictionManager(
            clock=self.clock,
            **(self.params | {"minexp_a": 0.0, "minexp_alpha": 5e-324}),
        )
        self.clock.return_value = 1001.0
        manager.on_block_cached(1, 64)
        with patch("builtins.__import__", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "score keys must be finite"):
                manager.on_block_freed(self.block(1))
        self.assertIsNone(manager._index)
        self.assertIsNone(manager._kind)

    def test_invalid_free_preserves_existing_native_membership(self):
        self.manager.on_block_freed(self.block(1))
        self.assertIsInstance(self.manager._index, MWFCandidateIndex)
        for block in self.invalid_blocks():
            with self.subTest(block=vars(block)):
                snapshot = vars(block).copy()
                with self.assertRaises(ValueError):
                    self.manager.on_block_freed(block)
                self.assertEqual(self.manager._index.size(), 1)
                self.assertEqual(vars(block), snapshot)
        self.assertEqual(self.manager._index.pop_n(1, 20.0), [1])

    def test_failed_native_import_preserves_record_and_allows_retry(self):
        manager, original_import = self.manager, __import__
        manager.on_block_cached(1, 64)
        block = self.block(1)
        keys, score = manager.get_score_keys(1), manager.get_score(1, now=1020)

        def import_without_native(name, *args, **kwargs):
            if name == "vllm._mwf_candidate_index":
                raise ModuleNotFoundError("native MWF extension unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_native):
            with self.assertRaisesRegex(ModuleNotFoundError, "extension unavailable"):
                manager.on_block_freed(block)
        self.assertEqual((manager._kind, manager._index), (None, None))
        self.assertEqual(manager.get_score_keys(1), keys)
        self.assertEqual(manager.get_score(1, now=1020), score)
        manager.on_block_freed(block)
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertEqual(manager._index.size(), 1)
