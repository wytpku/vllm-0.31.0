# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm._mwf_candidate_index import CandidateKind, MWFCandidateIndex
from vllm.v1.core.mwf_eviction import MWFEvictionManager, compute_mwf_score


class _LinkedFreeQueueFixture:
    """Supply linked-queue attributes for manager/native component tests."""

    def __init__(self, blocks):
        self.fake_free_list_head = SimpleNamespace(pool=None)
        self.fake_free_list_tail = SimpleNamespace(pool=None)
        self.num_free_blocks = len(blocks)
        self.iterated = 0
        self.yield_limit = None
        nodes = [self.fake_free_list_head, *blocks, self.fake_free_list_tail]
        for position, node in enumerate(nodes):
            node.prev_free_block = nodes[position - 1] if position else None
            node.next_free_block = (
                nodes[position + 1] if position + 1 < len(nodes) else None
            )

    def iter_blocks_after(self, block):
        current = (self.fake_free_list_head if block is None else block).next_free_block
        while current is not self.fake_free_list_tail:
            if self.yield_limit is not None and self.iterated >= self.yield_limit:
                raise AssertionError("Recovery exceeded its bounded queue traversal")
            self.iterated += 1
            yield current
            current = current.next_free_block


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

    def selection_pool(self, hashes, order=None):
        """Build pool/queue attribute fixtures for the real manager and native index."""
        pool = SimpleNamespace(blocks=[self.block(i, h) for i, h in enumerate(hashes)])
        for block in pool.blocks:
            block.pool = pool
            block.prev_free_block = block.next_free_block = None
        order = range(len(pool.blocks)) if order is None else order
        queue = _LinkedFreeQueueFixture([pool.blocks[i] for i in order])
        pool.free_block_queue = queue
        pool.get_num_free_blocks = lambda: queue.num_free_blocks
        return pool

    def pool_state(self, pool):
        blocks = [
            (
                b.block_hash, b.ref_cnt, id(b.prev_free_block),
                id(b.next_free_block), id(b.pool),
            )
            for b in pool.blocks
        ]
        queue = pool.free_block_queue
        return (
            blocks,
            id(queue.fake_free_list_head.next_free_block),
            id(queue.fake_free_list_tail.prev_free_block),
        )

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

    def test_access_refresh_changes_native_priority_without_adding_candidates(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_freed(self.block(1))
        self.clock.return_value = 1010.0
        manager.on_block_cached(2, 64)
        manager.on_block_freed(self.block(2))
        self.clock.return_value = 1020.0
        manager.on_block_accessed(1)
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertEqual(manager._index.size(), 2)
        self.assertLess(
            manager.get_score(2, now=1020).weight,
            manager.get_score(1, now=1020).weight,
        )
        self.assertEqual(manager._index.pop_n(2, 20.0), [2, 1])

    def test_cached_content_refreshes_prefix_and_promotes_missing_candidate(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_cached(2, 128)
        for block_id in (1, 2, 3):
            manager.on_block_freed(self.block(block_id))
        manager.on_block_cached(1, 20000)
        manager.on_block_cached(3, 40000)
        self.assertEqual(manager.get_score(1, now=1000).prefix_tokens, 20000)
        self.assertEqual(manager.get_score(3, now=1000).prefix_tokens, 40000)
        expected = sorted(
            (1, 2, 3), key=lambda bid: manager.get_score(bid, now=1000).weight
        )
        self.assertEqual(expected, [2, 1, 3])
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertEqual(manager._index.size(), 3)
        self.assertEqual(manager._index.pop_n(3, 0.0), expected)

    def test_record_updates_do_not_insert_blocks_without_free_candidates(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_freed(self.block(1))
        held = self.block(2, ref_cnt=1)
        manager.on_block_cached(held.block_id, 128)
        self.clock.return_value = 1010.0
        manager.on_block_accessed(held.block_id)
        manager.on_block_accessed(3)
        self.assertEqual(manager.get_score(2, now=1015).age_s, 5.0)
        self.assertEqual(manager.get_score(2, now=1015).prefix_tokens, 128)
        with self.assertRaises(KeyError):
            manager.get_score(3, now=1015)
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertEqual(manager._index.size(), 1)
        self.assertEqual(manager._index.pop_n(1, 15.0), [1])

    def test_failed_key_refresh_preserves_record_and_real_native_candidate(self):
        for event in ("cached", "accessed"):
            with self.subTest(event=event):
                clock = Mock(return_value=1000.0)
                manager = MWFEvictionManager(
                    clock=clock,
                    **(self.params | {"minexp_a": 0.0, "minexp_alpha": 5e-324}),
                )
                manager.on_block_cached(1, 64)
                manager.on_block_freed(self.block(1))
                keys = manager.get_score_keys(1)
                score = manager.get_score(1, now=1000.0)
                self.assertTrue(all(math.isfinite(key) for key in keys))
                clock.return_value = 1001.0
                with self.assertRaisesRegex(ValueError, "score keys must be finite"):
                    if event == "cached":
                        manager.on_block_cached(1, 128)
                    else:
                        manager.on_block_accessed(1)
                self.assertEqual(manager.get_score_keys(1), keys)
                self.assertEqual(manager.get_score(1, now=1000.0), score)
                self.assertIsInstance(manager._index, MWFCandidateIndex)
                self.assertEqual(manager._index.size(), 1)
                self.assertEqual(manager._index.pop_n(1, 0.0), [1])

    def test_reused_tracked_and_missing_keep_original_free_order(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_cached(4, 128)
        for block in (self.block(1), self.block(2), self.block(3, None), self.block(4)):
            manager.on_block_freed(block)
        self.clock.reset_mock()
        self.clock.side_effect = AssertionError("Reuse must not read the clock")
        for block_id in (2, 1, 2, 1):
            manager.on_block_reused(block_id)
        for block_id in (1, 2):
            with self.subTest(block_id=block_id):
                with self.assertRaises(KeyError):
                    manager.get_score(block_id, now=1000.0)
                with self.assertRaises(KeyError):
                    manager.get_score_keys(block_id)
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertEqual(manager._index.size(), 4)
        self.assertEqual(manager._index.pop_n(4, 0.0), [1, 2, 3, 4])
        self.clock.assert_not_called()

    def test_reused_non_candidate_clears_record_without_loading_or_adding(self):
        for native_loaded in (False, True):
            with self.subTest(native_loaded=native_loaded):
                clock = Mock(return_value=1000.0)
                manager = MWFEvictionManager(clock=clock, **self.params)
                manager.on_block_cached(2, 64)
                if native_loaded:
                    manager.on_block_freed(self.block(1))
                clock.reset_mock()
                clock.side_effect = AssertionError("Reuse must not read the clock")
                with patch("builtins.__import__", side_effect=AssertionError):
                    manager.on_block_reused(2)
                    manager.on_block_reused(2)
                    manager.on_block_reused(3)
                with self.assertRaises(KeyError):
                    manager.get_score(2, now=1000.0)
                with self.assertRaises(KeyError):
                    manager.get_score_keys(2)
                if native_loaded:
                    self.assertEqual(manager._index.size(), 1)
                    self.assertEqual(manager._index.pop_n(1, 0.0), [1])
                else:
                    self.assertIsNone(manager._index)
                clock.assert_not_called()

    def test_reused_invalid_ids_preserve_record_and_native_candidate(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_freed(self.block(1))
        keys, score = manager.get_score_keys(1), manager.get_score(1, now=1000.0)
        for block_id in (-1, True, 1.0, "1", None):
            with self.subTest(block_id=block_id):
                with self.assertRaises(ValueError):
                    manager.on_block_reused(block_id)
                self.assertEqual(manager.get_score_keys(1), keys)
                self.assertEqual(manager.get_score(1, now=1000.0), score)
                self.assertEqual(manager._index.size(), 1)
        self.assertEqual(manager._index.pop_n(1, 0.0), [1])

    def test_allocated_removes_each_candidate_kind_but_keeps_scoring_record(self):
        for removed in (1, 2, 3):
            with self.subTest(removed=removed):
                manager = MWFEvictionManager(clock=self.clock, **self.params)
                manager.on_block_cached(1, 64)
                for block in (self.block(1), self.block(2), self.block(3, None)):
                    manager.on_block_freed(block)
                keys = manager.get_score_keys(1)
                score = manager.get_score(1, now=1000.0)
                manager.on_block_allocated(removed)
                manager.on_block_allocated(removed)
                manager.on_block_allocated(99)
                self.assertEqual(manager.get_score_keys(1), keys)
                self.assertEqual(manager.get_score(1, now=1000.0), score)
                self.assertEqual(manager._index.size(), 2)
                self.assertEqual(
                    manager._index.pop_n(2, 0.0),
                    [block_id for block_id in (3, 2, 1) if block_id != removed],
                )

    def test_allocated_access_then_free_registers_latest_priority(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_freed(self.block(1))
        self.clock.return_value = 1010.0
        manager.on_block_cached(2, 64)
        manager.on_block_freed(self.block(2))
        manager.on_block_allocated(1)
        self.clock.return_value = 1020.0
        manager.on_block_accessed(1)
        self.assertEqual(manager.get_score(1, now=1020.0).age_s, 0.0)
        self.assertEqual(manager._index.size(), 1)
        manager.on_block_freed(self.block(1))
        self.assertEqual(manager._index.size(), 2)
        self.assertEqual(manager._index.pop_n(2, 20.0), [2, 1])

    def test_allocated_lazy_and_invalid_notifications_preserve_other_state(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        keys, score = manager.get_score_keys(1), manager.get_score(1, now=1000.0)
        self.clock.reset_mock()
        self.clock.side_effect = AssertionError("Allocation must not read the clock")
        with patch("builtins.__import__", side_effect=AssertionError):
            manager.on_block_allocated(1)
            manager.on_block_allocated(1)
            manager.on_block_allocated(99)
        self.assertIsNone(manager._index)
        self.assertEqual(manager.get_score_keys(1), keys)
        self.assertEqual(manager.get_score(1, now=1000.0), score)
        manager.on_block_freed(self.block(1))
        for block_id in (-1, True, 1.0, "1", None):
            with self.subTest(block_id=block_id):
                with self.assertRaises(ValueError):
                    manager.on_block_allocated(block_id)
        self.assertEqual(manager.get_score_keys(1), keys)
        self.assertEqual(manager.get_score(1, now=1000.0), score)
        self.assertEqual(manager._index.size(), 1)
        self.assertEqual(manager._index.pop_n(1, 0.0), [1])
        self.clock.assert_not_called()

    def test_reset_clears_all_kinds_and_reuses_native_index_for_new_content(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_cached(4, 128)
        for block in (self.block(1), self.block(2), self.block(3, None)):
            manager.on_block_freed(block)
        index = manager._index
        self.assertIsInstance(index, MWFCandidateIndex)
        manager.reset()
        manager.reset()
        self.assertIs(manager._index, index)
        self.assertEqual(index.size(), 0)
        for block_id in (1, 4):
            with self.subTest(block_id=block_id):
                with self.assertRaises(KeyError):
                    manager.get_score(block_id, now=1000.0)
                with self.assertRaises(KeyError):
                    manager.get_score_keys(block_id)
        self.clock.return_value = 1010.0
        manager.on_block_cached(1, 128)
        manager.on_block_freed(self.block(1))
        self.clock.return_value = 1020.0
        manager.on_block_cached(5, 64)
        manager.on_block_freed(self.block(5))
        self.assertEqual(
            manager.get_score(1, now=1030.0),
            compute_mwf_score(
                prefix_tokens=128, last_accessed=1010.0, now=1030.0, **self.params
            ),
        )
        self.assertEqual(index.size(), 2)
        self.assertEqual(index.pop_n(2, 30.0), [1, 5])

    def test_resync_replaces_candidates_in_queue_order_and_preserves_records(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_cached(2, 64)
        manager.on_block_freed(self.block(1))
        old_index = manager._index
        self.clock.return_value = 1010.0
        for block_id in (5, 8, 99):
            manager.on_block_cached(block_id, 128)
        records = {
            bid: (manager.get_score(bid, now=1020), manager.get_score_keys(bid))
            for bid in (1, 8, 99)
        }
        blocks = [
            self.block(5),
            self.block(7, 0),
            self.block(8, None),
            self.block(2),
            self.block(9, None),
            self.block(6),
        ]
        self.clock.reset_mock()
        self.clock.side_effect = AssertionError("Resync must not read the clock")
        manager.resync(iter(blocks))
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertIsNot(manager._index, old_index)
        self.assertEqual(manager._index.size(), 6)
        self.assertEqual(manager._index.pop_n(6, 20.0), [8, 9, 7, 6, 2, 5])
        for bid, record in records.items():
            self.assertEqual(
                (manager.get_score(bid, now=1020), manager.get_score_keys(bid)), record
            )
        self.clock.assert_not_called()

    def test_resync_empty_input_commits_empty_index_without_clearing_records(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_freed(self.block(1))
        score, keys = manager.get_score(1, now=1000), manager.get_score_keys(1)
        manager.resync(())
        self.assertIsInstance(manager._index, MWFCandidateIndex)
        self.assertEqual(manager._index.size(), 0)
        self.assertEqual(manager.get_score(1, now=1000), score)
        self.assertEqual(manager.get_score_keys(1), keys)

    def test_resync_partial_build_failures_preserve_old_index_and_record(self):
        manager = self.manager
        manager.on_block_cached(1, 64)
        manager.on_block_freed(self.block(1))
        old_index = manager._index
        score, keys = manager.get_score(1, now=1000), manager.get_score_keys(1)
        temporary = self.block(9, None)

        def interrupted():
            yield temporary
            raise RuntimeError("free queue iteration failed")

        cases = [
            ([temporary, temporary], ValueError),
            ([temporary, self.block(True)], ValueError),
            ([temporary, self.block(8, is_null=True)], ValueError),
            ([temporary, self.block(8, ref_cnt=1)], ValueError),
            ([temporary, self.block(8, ref_cnt=-1)], ValueError),
            (interrupted(), RuntimeError),
        ]
        for position, (blocks, error) in enumerate(cases):
            with self.subTest(case=position):
                with self.assertRaises(error):
                    manager.resync(blocks)
                self.assertIs(manager._index, old_index)
                self.assertEqual(old_index.size(), 1)
                self.assertEqual(manager.get_score(1, now=1000), score)
                self.assertEqual(manager.get_score_keys(1), keys)
        self.assertEqual(old_index.pop_n(1, 0.0), [1])

    def test_resync_bad_score_keys_preserve_old_candidates_and_metadata(self):
        clock = Mock(return_value=1000.0)
        manager = MWFEvictionManager(
            clock=clock,
            **(self.params | {"minexp_a": 0.0, "minexp_alpha": 5e-324}),
        )
        manager.on_block_cached(1, 64)
        clock.return_value = 1001.0
        manager.on_block_cached(2, 128)
        manager.on_block_freed(self.block(1))
        old_index = manager._index
        score, keys = manager.get_score(1, now=1000), manager.get_score_keys(1)
        invalid_record_score = manager.get_score(2, now=1001)
        with self.assertRaisesRegex(ValueError, "score keys must be finite"):
            manager.resync([self.block(9, None), self.block(2)])
        self.assertIs(manager._index, old_index)
        self.assertEqual(old_index.size(), 1)
        self.assertEqual(manager.get_score(1, now=1000), score)
        self.assertEqual(manager.get_score_keys(1), keys)
        self.assertEqual(manager.get_score(2, now=1001), invalid_record_score)
        self.assertEqual(old_index.pop_n(1, 0.0), [1])

    def test_select_returns_canonical_order_without_mutating_pool_or_metadata(self):
        manager = self.manager
        pool = self.selection_pool((1, 1, None, 0), order=(1, 3, 0, 2))
        manager.on_block_cached(0, 20000)
        self.clock.return_value = 1010.0
        manager.on_block_cached(1, 64)
        for block in pool.free_block_queue.iter_blocks_after(None):
            manager.on_block_freed(block)
        before = self.pool_state(pool)
        records = [
            (manager.get_score(i, now=1020), manager.get_score_keys(i)) for i in (0, 1)
        ]
        pool.free_block_queue.iterated = 0
        self.clock.return_value = 1020.0
        selected = manager.select_blocks(3, pool=pool)
        self.assertEqual([block.block_id for block in selected], [2, 3, 0])
        for block in selected:
            self.assertIs(block, pool.blocks[block.block_id])
        self.assertEqual(manager._index.size(), 1)
        self.assertEqual(pool.free_block_queue.iterated, 0)
        self.assertEqual(self.pool_state(pool), before)
        self.assertEqual(pool.get_num_free_blocks(), 4)
        self.assertEqual(
            [
                (manager.get_score(i, now=1020), manager.get_score_keys(i))
                for i in (0, 1)
            ],
            records,
        )

    def test_select_zero_and_invalid_counts_do_not_read_clock_or_pop(self):
        manager = self.manager
        pool = self.selection_pool((None,))
        manager.on_block_freed(pool.blocks[0])
        self.clock.reset_mock()
        self.clock.side_effect = AssertionError("Invalid count must not read the clock")
        self.assertEqual(manager.select_blocks(0, pool=pool), [])
        for count in (-1, True, 1.0, None, 2):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    manager.select_blocks(count, pool=pool)
                self.assertEqual(manager._index.size(), 1)
        self.assertEqual(manager._index.pop_n(1, 0.0), [0])
        self.clock.assert_not_called()

    def test_select_rejects_nonfinite_or_negative_elapsed_before_pop(self):
        pool = self.selection_pool((None,))
        cases = [(1000.0, value) for value in (math.nan, math.inf, -math.inf, 999.0)]
        for origin, now in (*cases, (-1e308, 1e308)):
            with self.subTest(origin=origin, now=now):
                clock = Mock(return_value=origin)
                manager = MWFEvictionManager(clock=clock, **self.params)
                manager.on_block_freed(pool.blocks[0])
                clock.return_value = now
                with self.assertRaisesRegex(ValueError, "elapsed time"):
                    manager.select_blocks(1, pool=pool)
                self.assertEqual(manager._index.size(), 1)
                clock.return_value = origin
                self.assertIs(manager.select_blocks(1, pool=pool)[0], pool.blocks[0])

    def test_select_missing_index_or_count_mismatch_recovers_then_raises(self):
        for partial_index in (False, True):
            with self.subTest(partial_index=partial_index):
                manager = MWFEvictionManager(clock=self.clock, **self.params)
                pool = self.selection_pool((None, None))
                if partial_index:
                    manager.on_block_freed(pool.blocks[1])
                with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                    manager.select_blocks(1, pool=pool)
                self.assertIsInstance(manager._index, MWFCandidateIndex)
                self.assertEqual(manager._index.size(), 2)
                selected = manager.select_blocks(2, pool=pool)
                self.assertEqual(
                    [id(block) for block in selected],
                    [id(block) for block in pool.blocks],
                )

    def test_select_stale_native_id_recovers_from_healthy_queue_after_pop(self):
        manager = self.manager
        pool = self.selection_pool((None, None))
        manager.on_block_freed(self.block(99, None))
        manager.on_block_freed(pool.blocks[1])
        with self.assertRaisesRegex(RuntimeError, "outside this pool"):
            manager.select_blocks(1, pool=pool)
        self.assertEqual(manager._index.size(), 2)
        self.assertEqual(pool.free_block_queue.iterated, 2)
        selected = manager.select_blocks(2, pool=pool)
        self.assertEqual(
            [id(block) for block in selected],
            [id(block) for block in pool.blocks],
        )

    def test_select_recovery_rejects_cycles_or_bad_lengths_with_bounded_iteration(self):
        for damage in ("cycle", "too_short", "too_long"):
            with self.subTest(damage=damage):
                manager = MWFEvictionManager(clock=self.clock, **self.params)
                pool = self.selection_pool((None, None))
                queue = pool.free_block_queue
                manager.resync(queue.iter_blocks_after(None))
                manager._index.clear()
                queue.iterated = 0
                if damage == "cycle":
                    pool.blocks[1].next_free_block = pool.blocks[0]
                else:
                    queue.num_free_blocks = 3 if damage == "too_short" else 1
                queue.yield_limit = queue.num_free_blocks + 1
                with self.assertRaisesRegex(RuntimeError, "invalid free queue"):
                    manager.select_blocks(1, pool=pool)
                self.assertLessEqual(queue.iterated, queue.num_free_blocks + 1)
                self.assertEqual(manager._index.size(), 0)
