from copy import deepcopy

import pytest
import torch

from arw.m0_core import (
    BranchState,
    build_causal_branches,
    capture_optimizer_state,
    capture_trainable_state,
    optimizer_moment_distance,
    optimizer_state_distance,
    recovery_half_life,
    restore_branch,
    take_candidate_step,
    trapezoid_auc,
)


def _model() -> torch.nn.Module:
    torch.manual_seed(7)
    return torch.nn.Linear(2, 2, bias=False)


def _candidate_pair(
    optimizer_factory: object,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, BranchState]]:
    model = _model()
    optimizer = optimizer_factory(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    clean_y = torch.tensor([0, 1])
    corrupt_y = torch.tensor([1, 0])
    pre_parameters = capture_trainable_state(model)
    pre_optimizer = capture_optimizer_state(optimizer)

    clean = take_candidate_step(
        model,
        optimizer,
        criterion,
        x,
        clean_y,
        pre_parameters,
        pre_optimizer,
    )
    corrupt = take_candidate_step(
        model,
        optimizer,
        criterion,
        x,
        corrupt_y,
        pre_parameters,
        pre_optimizer,
        target_gradient_norm=clean.gradient_norm,
    )
    return model, optimizer, build_causal_branches(clean, corrupt)


def test_adamw_cross_combines_parameter_and_optimizer_state() -> None:
    model, optimizer, branches = _candidate_pair(
        lambda params: torch.optim.AdamW(params, lr=0.05, weight_decay=0.0)
    )

    for name in branches["control"].parameters:
        torch.testing.assert_close(
            branches["parameter_only"].parameters[name],
            branches["full"].parameters[name],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            branches["state_only"].parameters[name],
            branches["control"].parameters[name],
            rtol=0,
            atol=0,
        )
    assert (
        optimizer_state_distance(branches["state_only"].optimizer, branches["control"].optimizer)
        > 0
    )
    assert (
        optimizer_moment_distance(
            branches["state_only"].optimizer,
            branches["control"].optimizer,
            "exp_avg",
        )
        > 0
    )
    assert (
        optimizer_moment_distance(
            branches["state_only"].optimizer,
            branches["control"].optimizer,
            "exp_avg_sq",
        )
        > 0
    )

    restore_branch(model, optimizer, branches["state_only"])
    restored_parameters = capture_trainable_state(model)
    for name, value in branches["control"].parameters.items():
        torch.testing.assert_close(restored_parameters[name], value, rtol=0, atol=0)


def test_sgd_without_momentum_state_only_is_exact_control() -> None:
    model, optimizer, branches = _candidate_pair(
        lambda params: torch.optim.SGD(params, lr=0.05, momentum=0.0)
    )

    assert (
        optimizer_state_distance(branches["state_only"].optimizer, branches["control"].optimizer)
        == 0.0
    )
    restore_branch(model, optimizer, branches["control"])
    control_parameters = deepcopy(capture_trainable_state(model))
    restore_branch(model, optimizer, branches["state_only"])
    state_only_parameters = capture_trainable_state(model)
    for name in control_parameters:
        torch.testing.assert_close(
            state_only_parameters[name], control_parameters[name], rtol=0, atol=0
        )


def test_corrupt_gradient_is_norm_matched() -> None:
    _, _, branches = _candidate_pair(
        lambda params: torch.optim.AdamW(params, lr=0.05, weight_decay=0.0)
    )
    assert branches["control"].gradient_norm == pytest.approx(
        branches["full"].gradient_norm, rel=1e-6
    )


def test_trapezoid_auc_uses_irregular_horizons() -> None:
    assert trapezoid_auc([0, 1, 4], [0.0, 1.0, 1.0]) == pytest.approx(3.5)


def test_recovery_half_life_returns_first_observed_recovery() -> None:
    assert recovery_half_life([0, 1, 4, 8], [2.0, 1.5, 0.9, 0.1]) == 4
    assert recovery_half_life([0, 1], [0.0, 0.0]) == 0
    assert recovery_half_life([0, 1, 4], [2.0, 1.5, 1.1]) is None
