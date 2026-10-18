import math
from dataclasses import dataclass

MWF_DELTA_SLOPE = 0.00043533405552524906
MWF_DELTA_INTERCEPT = 16.975203511045493


@dataclass(frozen=True)
class MWFScore:
    weight: float
    frequency: float
    prefix_tokens: int
    age_s: float


def compute_mwf_score(
    *,
    prefix_tokens: int,
    last_accessed: float,
    now: float,
    minexp_a: float,
    minexp_b: float,
    minexp_alpha: float,
    minexp_beta: float,
) -> MWFScore:
    """Compute the original MinExp MWF score for one cached block."""
    if prefix_tokens <= 0:
        raise ValueError("MWF prefix_tokens must be positive")

    values = (
        last_accessed,
        now,
        minexp_a,
        minexp_b,
        minexp_alpha,
        minexp_beta,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("MWF parameters and timestamps must be finite")
    if minexp_alpha <= 0 or minexp_beta <= 0:
        raise ValueError("MWF MinExp alpha and beta must be positive")

    age_s = max(0.0, now - last_accessed)
    log_frequency = min(
        -(minexp_a + age_s) / minexp_alpha,
        -(minexp_b + age_s) / minexp_beta,
    )
    frequency = math.exp(log_frequency)
    recompute_cost = (
        MWF_DELTA_SLOPE * prefix_tokens + MWF_DELTA_INTERCEPT
    )
    return MWFScore(
        weight=frequency * recompute_cost,
        frequency=frequency,
        prefix_tokens=prefix_tokens,
        age_s=age_s,
    )