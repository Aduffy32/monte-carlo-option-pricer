import numpy as np
from pricer import bs_call, mc_call

PARAMS = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def test_bs_known_value():
    # Standard textbook value for these inputs is about 10.4506
    assert abs(bs_call(**PARAMS) - 10.4506) < 1e-3


def test_mc_within_four_standard_errors():
    rng = np.random.default_rng(1)
    price, se = mc_call(**PARAMS, n_paths=200_000, rng=rng)
    assert abs(price - bs_call(**PARAMS)) < 4 * se


def test_antithetic_reduces_standard_error():
    n = 200_000
    _, se_plain = mc_call(**PARAMS, n_paths=n, rng=np.random.default_rng(2))
    _, se_anti = mc_call(**PARAMS, n_paths=n, rng=np.random.default_rng(2), antithetic=True)
    assert se_anti < se_plain


def test_deep_out_of_the_money_is_near_zero():
    p = dict(PARAMS, K=1000.0)
    assert bs_call(**p) < 1e-6
