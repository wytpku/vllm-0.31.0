# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import unittest

from vllm.v1.core.mwf_eviction import compute_mwf_score


class TestMWFScore(unittest.TestCase):
    def score(self, **overrides):
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
        return compute_mwf_score(**inputs)

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
