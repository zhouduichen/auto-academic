import math

from experiments.m1_simple_baseline_tuning import beta_grid, clip_grid


def test_clip_grid_is_exact_frozen_grid() -> None:
    grid = clip_grid(2.0)
    assert len(grid) == 12
    assert [item.max_grad_norm for item in grid] == [
        2.0 * 2.0**exponent
        for exponent in (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    ]
    assert all(math.isfinite(item.max_grad_norm or math.nan) for item in grid)


def test_beta_grid_is_exact_and_contains_adamw_anchor() -> None:
    grid = beta_grid()
    assert len(grid) == 12
    assert {(item.beta1, item.beta2) for item in grid} == {
        (beta1, beta2) for beta1 in (0.5, 0.7, 0.9, 0.95) for beta2 in (0.95, 0.99, 0.999)
    }
    anchor = [item for item in grid if (item.beta1, item.beta2) == (0.9, 0.999)]
    assert len(anchor) == 1
    assert anchor[0].nondefault_count == 0
