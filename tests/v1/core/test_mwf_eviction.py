# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import unittest
from decimal import Decimal, localcontext
from unittest.mock import Mock

from vllm.v1.core.mwf_eviction import (
    MWFEvictionManager,
    compute_mwf_score,
    compute_mwf_score_keys,
)


class TestMWFScore(unittest.TestCase):
    def inputs(self, **overrides):
        inputs = dict(
            prefix_tokens=64,
            last_accessed=1000.0,
            now=1000.0,
            minexp_a=0.0,
            minexp_b=-19.15349,
            minexp_alpha=1915.332,
            minexp_beta=47.63942,
        )
        inputs.update(overrides)
        return inputs

    def score(self, **overrides):
        return compute_mwf_score(**self.inputs(**overrides))

    def keys(self, *, time_origin=900.0, **overrides):
        inputs = self.inputs(**overrides)
        inputs.pop("now")
        return compute_mwf_score_keys(time_origin=time_origin, **inputs)

    def test_original_minexp_reference_values(self):
        # Decimal reference values cover both branches of the original formula.
        samples = (
            (0.0, 1.0, 17.00306489059910894),
            (10.0, 0.9947925789169414, 16.91452277201118963),
            (100.0, 0.18322311194777874, 3.11535446190558678),
            (10000.0, 1.027469031242884e-91, 1.747012261130376e-90),
        )
        for age, frequency, weight in samples:
            with self.subTest(age=age):
                result = self.score(now=1000.0 + age)
                self.assertTrue(
                    math.isclose(result.frequency, frequency, rel_tol=1e-12)
                )
                self.assertTrue(math.isclose(result.weight, weight, rel_tol=1e-12))
                self.assertEqual(result.age_s, age)
                self.assertEqual(result.prefix_tokens, 64)

    def test_age_and_cumulative_prefix_change_eviction_priority(self):
        recent = self.score()
        old = self.score(now=1100.0)
        longer_prefix = self.score(prefix_tokens=128)
        self.assertLess(old.weight, recent.weight)
        self.assertGreater(longer_prefix.weight, recent.weight)
        self.assertEqual(longer_prefix.frequency, recent.frequency)
        self.assertEqual(longer_prefix.prefix_tokens, 128)

    def test_negative_offsets_can_produce_frequency_above_one(self):
        result = self.score(
            minexp_a=-2.0, minexp_b=-3.0, minexp_alpha=2.0, minexp_beta=3.0
        )
        self.assertAlmostEqual(result.frequency, math.e)

    def test_future_access_time_clamps_age_to_zero(self):
        self.assertEqual(self.score(last_accessed=1001.0), self.score())

    def test_prefix_tokens_must_be_positive(self):
        for value in (0, -16):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "prefix_tokens must be positive"
                ):
                    self.score(prefix_tokens=value)

    def test_parameters_and_timestamps_must_be_finite(self):
        for name in (
            "last_accessed",
            "now",
            "minexp_a",
            "minexp_b",
            "minexp_alpha",
            "minexp_beta",
        ):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        self.score(**{name: value})

    def test_time_scales_must_be_positive(self):
        for name in ("minexp_alpha", "minexp_beta"):
            for value in (0.0, -1.0):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "must be positive"):
                        self.score(**{name: value})

    def test_unselected_exponential_overflow_does_not_break_finite_score(self):
        result = self.score(
            prefix_tokens=16,
            last_accessed=0.0,
            now=0.0,
            minexp_a=-1000.0,
            minexp_b=0.0,
            minexp_alpha=1.0,
            minexp_beta=1.0,
        )
        self.assertEqual(result.frequency, 1.0)
        self.assertTrue(math.isfinite(result.weight))

    def test_native_keys_reconstruct_original_log_weight(self):
        for prefix, last, now, origin in (
            (16, 1000.0, 1000.0, 900.0),
            (64, 1010.0, 1110.0, 900.0),
            (128, 1000.0, 1010.0, 1000.0),
            (256, 1020.0, 1040.0, -100.0),
        ):
            with self.subTest(prefix=prefix, last=last, now=now, origin=origin):
                inputs = self.inputs(prefix_tokens=prefix, last_accessed=last, now=now)
                score = self.score(**inputs)
                key_a, key_b = self.keys(time_origin=origin, **inputs)
                log_weight = min(
                    key_a - (now - origin) / inputs["minexp_alpha"],
                    key_b - (now - origin) / inputs["minexp_beta"],
                )
                self.assertGreater(score.weight, 0.0)
                self.assertAlmostEqual(log_weight, math.log(score.weight), places=12)

    def test_keys_preserve_priority_after_float_weight_underflows(self):
        inputs = dict(
            now=100000.0,
            minexp_a=0.0,
            minexp_b=0.0,
            minexp_alpha=1.0,
            minexp_beta=1.0,
        )
        with localcontext() as context:
            context.prec = 50
            cost = Decimal("17.00306489059910893984")
            older = cost * Decimal(-100000).exp()
            newer = cost * Decimal(-99999).exp()
            self.assertLess(older, newer)
            self.assertGreater(older, 0)
            expected_log_gap = float((newer / older).ln())
        self.assertEqual(self.score(last_accessed=0.0, **inputs).weight, 0.0)
        self.assertEqual(self.score(last_accessed=1.0, **inputs).weight, 0.0)
        older_key = min(self.keys(time_origin=0.0, last_accessed=0.0, **inputs))
        newer_key = min(self.keys(time_origin=0.0, last_accessed=1.0, **inputs))
        self.assertLess(older_key, newer_key)
        self.assertAlmostEqual(newer_key - older_key, expected_log_gap)

    def test_keys_are_invariant_to_common_timestamp_translation(self):
        baseline = self.keys(last_accessed=1000.0, time_origin=900.0)
        for shift in (-2000.0, float(2**30)):
            with self.subTest(shift=shift):
                self.assertEqual(
                    self.keys(last_accessed=1000.0 + shift, time_origin=900.0 + shift),
                    baseline,
                )

    def test_keys_reject_invalid_inputs_and_nonfinite_results(self):
        cases = [{"prefix_tokens": value} for value in (0, -16)]
        for name in (
            "last_accessed",
            "time_origin",
            "minexp_a",
            "minexp_b",
            "minexp_alpha",
            "minexp_beta",
        ):
            cases.extend({name: value} for value in (math.nan, math.inf, -math.inf))
        for name in ("minexp_alpha", "minexp_beta"):
            cases.extend({name: value} for value in (0.0, -1.0))
        cases.extend(
            (
                {"last_accessed": 1e308, "time_origin": -1e308},
                {"last_accessed": 1e308, "time_origin": 0.0, "minexp_a": -1e308},
                {"last_accessed": 1.0, "time_origin": 0.0, "minexp_alpha": 5e-324},
            )
        )
        for inputs in cases:
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError):
                    self.keys(**inputs)


class TestMWFMetadata(unittest.TestCase):
    def params(self, **overrides):
        params = dict(
            minexp_a=-3.0,
            minexp_b=4.0,
            minexp_alpha=17.0,
            minexp_beta=29.0,
        )
        params.update(overrides)
        return params

    def test_registered_score_matches_formula_and_reads_do_not_refresh_access(self):
        clock = Mock(return_value=10.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(0, 64)
        clock.return_value = 20.0
        self.assertEqual(
            manager.get_score(0),
            compute_mwf_score(
                prefix_tokens=64, last_accessed=10.0, now=20.0, **self.params()
            ),
        )
        self.assertEqual(
            manager.get_score(0, now=30.0),
            compute_mwf_score(
                prefix_tokens=64, last_accessed=10.0, now=30.0, **self.params()
            ),
        )

    def test_content_replacement_resets_only_its_own_record(self):
        clock = Mock(return_value=10.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(1, 16)
        manager.on_block_cached(2, 32)
        unchanged = manager.get_score(2, now=20.0)
        manager.on_block_reused(1)
        manager.on_block_reused(1)
        with self.assertRaises(KeyError):
            manager.get_score(1, now=20.0)
        self.assertEqual(manager.get_score(2, now=20.0), unchanged)
        clock.return_value = 30.0
        manager.on_block_cached(1, 128)
        self.assertEqual(manager.get_score(1, now=35.0).age_s, 5.0)
        self.assertEqual(manager.get_score(1, now=35.0).prefix_tokens, 128)
        clock.return_value = 40.0
        manager.on_block_cached(1, 64)
        self.assertEqual(manager.get_score(1, now=41.0).age_s, 1.0)
        self.assertEqual(manager.get_score(1, now=41.0).prefix_tokens, 64)
        self.assertEqual(manager.get_score(2, now=20.0), unchanged)

    def test_explicit_zero_now_does_not_read_clock(self):
        clock = Mock(return_value=0.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(0, 16)
        clock.reset_mock()
        clock.side_effect = AssertionError("Explicit now must bypass the clock")
        self.assertEqual(manager.get_score(0, now=0.0).age_s, 0.0)
        clock.assert_not_called()

    def test_failed_registration_and_invalid_ids_preserve_existing_record(self):
        clock = Mock(return_value=10.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(1, 64)
        original = manager.get_score(1, now=20.0)
        for block_id in (-1, True, 1.0, "1", None):
            for method, args, kwargs in (
                (manager.on_block_cached, (block_id, 128), {}),
                (manager.get_score, (block_id,), {"now": 20.0}),
                (manager.get_score_keys, (block_id,), {}),
                (manager.on_block_reused, (block_id,), {}),
            ):
                with self.subTest(block_id=block_id, method=method.__name__):
                    with self.assertRaises(ValueError):
                        method(*args, **kwargs)
                    self.assertEqual(manager.get_score(1, now=20.0), original)
        for prefix in (0, -16, True, 16.0, None):
            with self.subTest(prefix=prefix):
                with self.assertRaises(ValueError):
                    manager.on_block_cached(1, prefix)
                self.assertEqual(manager.get_score(1, now=20.0), original)
        for timestamp in (math.nan, math.inf, -math.inf):
            with self.subTest(timestamp=timestamp):
                clock.return_value = timestamp
                with self.assertRaises(ValueError):
                    manager.on_block_cached(1, 128)
                self.assertEqual(manager.get_score(1, now=20.0), original)
        clock.side_effect = RuntimeError("clock unavailable")
        with self.assertRaisesRegex(RuntimeError, "clock unavailable"):
            manager.on_block_cached(1, 128)
        self.assertEqual(manager.get_score(1, now=20.0), original)

    def test_constructor_rejects_invalid_parameters_and_noncallable_clock(self):
        cases = []
        for name in self.params():
            cases.extend({name: value} for value in (math.nan, math.inf, -math.inf))
        for name in ("minexp_alpha", "minexp_beta"):
            cases.extend({name: value} for value in (0.0, -1.0))
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    MWFEvictionManager(
                        clock=Mock(return_value=0.0), **self.params(**params)
                    )
        for clock in (None, 0, "clock"):
            with self.subTest(clock=clock):
                with self.assertRaises(TypeError):
                    MWFEvictionManager(clock=clock, **self.params())

    def test_failed_queries_preserve_record_for_later_valid_reads(self):
        clock = Mock(return_value=10.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(1, 64)
        original = manager.get_score(1, now=20.0)
        for timestamp in (9.0, math.nan, math.inf, -math.inf):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    manager.get_score(1, now=timestamp)
                clock.return_value = timestamp
                with self.assertRaises(ValueError):
                    manager.get_score(1)
                self.assertEqual(manager.get_score(1, now=20.0), original)
        clock.side_effect = RuntimeError("clock unavailable")
        with self.assertRaisesRegex(RuntimeError, "clock unavailable"):
            manager.get_score(1)
        self.assertEqual(manager.get_score(1, now=20.0), original)

    def test_access_refreshes_one_record_once_and_preserves_prefix(self):
        clock = Mock(return_value=1000.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(1, 64)
        manager.on_block_cached(2, 32)
        unchanged = manager.get_score(2, now=1015.0)
        for _ in range(2):
            clock.return_value = 1010.0
            clock.reset_mock()
            manager.on_block_accessed(1)
            clock.assert_called_once_with()
            score = manager.get_score(1, now=1015.0)
            self.assertEqual(score.age_s, 5.0)
            self.assertEqual(score.prefix_tokens, 64)
            self.assertEqual(manager.get_score(2, now=1015.0), unchanged)

    def test_access_to_missing_or_invalid_ids_never_creates_or_changes_records(self):
        clock = Mock(return_value=1000.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(1, 64)
        original = manager.get_score(1, now=1015.0)
        clock.reset_mock()
        clock.side_effect = AssertionError("No valid record was accessed")
        self.assertIsNone(manager.on_block_accessed(2))
        with self.assertRaises(KeyError):
            manager.get_score(2, now=1015.0)
        with self.assertRaises(KeyError):
            manager.get_score_keys(2)
        for block_id in (-1, True, 1.0, "1", None):
            with self.subTest(block_id=block_id):
                with self.assertRaises(ValueError):
                    manager.on_block_accessed(block_id)
                self.assertEqual(manager.get_score(1, now=1015.0), original)
        clock.assert_not_called()

    def test_failed_access_clock_preserves_last_valid_access(self):
        clock = Mock(return_value=1000.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(1, 64)
        clock.return_value = 1010.0
        manager.on_block_accessed(1)
        original = manager.get_score(1, now=1015.0)
        for timestamp in (math.nan, math.inf, -math.inf, 1009.0):
            with self.subTest(timestamp=timestamp):
                clock.return_value = timestamp
                with self.assertRaises(ValueError):
                    manager.on_block_accessed(1)
                self.assertEqual(manager.get_score(1, now=1015.0), original)
        clock.side_effect = RuntimeError("clock unavailable")
        with self.assertRaisesRegex(RuntimeError, "clock unavailable"):
            manager.on_block_accessed(1)
        self.assertEqual(manager.get_score(1, now=1015.0), original)

    def test_keys_share_one_constructor_origin_and_reconstruct_score(self):
        clock = Mock(side_effect=(1000.0, 1010.0, 1020.0))
        manager = MWFEvictionManager(clock=clock, **self.params())
        clock.assert_called_once_with()
        manager.on_block_cached(1, 64)
        manager.on_block_cached(2, 128)
        clock.reset_mock()
        clock.side_effect = AssertionError("Reading keys must not read the clock")
        for block_id, prefix, last in ((1, 64, 1010.0), (2, 128, 1020.0)):
            with self.subTest(block_id=block_id):
                keys = manager.get_score_keys(block_id)
                self.assertEqual(
                    keys,
                    compute_mwf_score_keys(
                        prefix_tokens=prefix,
                        last_accessed=last,
                        time_origin=1000.0,
                        **self.params(),
                    ),
                )
                log_weight = min(
                    keys[0] - 30.0 / self.params()["minexp_alpha"],
                    keys[1] - 30.0 / self.params()["minexp_beta"],
                )
                self.assertAlmostEqual(
                    log_weight, math.log(manager.get_score(block_id, now=1030.0).weight)
                )
        clock.assert_not_called()

    def test_keys_follow_access_and_content_invalidation_without_reading_clock(self):
        clock = Mock(return_value=1000.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(1, 64)
        manager.on_block_cached(2, 32)
        before = manager.get_score_keys(1)
        unchanged = manager.get_score_keys(2)
        clock.return_value = 1010.0
        manager.on_block_accessed(1)
        clock.reset_mock()
        clock.side_effect = AssertionError("Reading keys must not read the clock")
        after = manager.get_score_keys(1)
        for old, new, scale in zip(before, after, (17.0, 29.0)):
            self.assertAlmostEqual(new - old, 10.0 / scale)
        self.assertEqual(manager.get_score_keys(1), after)
        manager.on_block_reused(1)
        with self.assertRaises(KeyError):
            manager.get_score_keys(1)
        self.assertEqual(manager.get_score_keys(2), unchanged)
        clock.assert_not_called()

    def test_invalid_origin_or_cache_time_cannot_publish_replacement_keys(self):
        for origin in (math.nan, math.inf, -math.inf):
            with self.subTest(origin=origin):
                with self.assertRaises(ValueError):
                    MWFEvictionManager(clock=Mock(return_value=origin), **self.params())
        with self.assertRaisesRegex(RuntimeError, "clock unavailable"):
            MWFEvictionManager(
                clock=Mock(side_effect=RuntimeError("clock unavailable")),
                **self.params(),
            )
        clock = Mock(return_value=1000.0)
        manager = MWFEvictionManager(clock=clock, **self.params())
        manager.on_block_cached(1, 64)
        original = manager.get_score(1, now=1010.0)
        keys = manager.get_score_keys(1)
        clock.return_value = 999.0
        with self.assertRaises(ValueError):
            manager.on_block_cached(1, 128)
        self.assertEqual(manager.get_score(1, now=1010.0), original)
        self.assertEqual(manager.get_score_keys(1), keys)
