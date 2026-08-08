from __future__ import annotations

import copy
import inspect

import pytest
import torch

from experiments.m1_optimizers import CAdamW, ProtectAdamW


def _parameters() -> list[dict[str, object]]:
    return [
        {
            "params": [
                torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32)),
                torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32)),
            ],
            "weight_decay": 0.1,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
        {
            "params": [torch.nn.Parameter(torch.tensor([-0.25, 3.0], dtype=torch.float32))],
            "weight_decay": 0.0,
            "betas": (0.7, 0.95),
            "eps": 1e-6,
        },
    ]


def _clone_groups(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **{key: value for key, value in group.items() if key != "params"},
            "params": [
                torch.nn.Parameter(parameter.detach().clone()) for parameter in group["params"]
            ],
        }
        for group in groups
    ]


def _flat(groups: list[dict[str, object]]) -> list[torch.nn.Parameter]:
    return [parameter for group in groups for parameter in group["params"]]


def _assert_adam_state_equal(
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
    left_parameters: list[torch.nn.Parameter],
    right_parameters: list[torch.nn.Parameter],
) -> None:
    for left_parameter, right_parameter in zip(left_parameters, right_parameters, strict=True):
        assert torch.equal(left_parameter, right_parameter)
        left_state = left.state.get(left_parameter, {})
        right_state = right.state.get(right_parameter, {})
        assert set(left_state) == set(right_state)
        for key in left_state:
            assert torch.equal(left_state[key], right_state[key])


def test_protection_disabled_is_bit_exact_adamw() -> None:
    base = _parameters()
    reference_groups = _clone_groups(base)
    candidate_groups = _clone_groups(base)
    reference_parameters = _flat(reference_groups)
    candidate_parameters = _flat(candidate_groups)
    reference = torch.optim.AdamW(
        reference_groups, lr=3e-4, foreach=False, fused=False, amsgrad=False
    )
    candidate = ProtectAdamW(
        candidate_groups,
        lr=3e-4,
        protection_target="m",
        protection_enabled=False,
    )
    fixtures = (
        ([0.2, -0.1], [0.0], [-0.3, 0.4]),
        ([0.0, 0.0], None, [0.2, -0.7]),
        ([-0.4, 0.6], [0.8], [0.0, 0.0]),
        ([0.3, 0.1], [-0.2], [0.4, 0.5]),
        ([-0.9, -0.2], [0.1], [-0.6, 0.2]),
    )
    for step in fixtures:
        for left, right, gradient in zip(
            reference_parameters, candidate_parameters, step, strict=True
        ):
            left.grad = None if gradient is None else torch.tensor(gradient, dtype=torch.float32)
            right.grad = None if gradient is None else torch.tensor(gradient, dtype=torch.float32)
        reference.step()
        candidate.step()
        _assert_adam_state_equal(
            reference, candidate, reference_parameters, candidate_parameters
        )


def _forced_optimizer(target: str) -> tuple[ProtectAdamW, torch.nn.Parameter]:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = ProtectAdamW(
        [parameter],
        lr=1e-2,
        weight_decay=0.1,
        protection_target=target,
        alpha_fast=0.9,
        alpha_slow=0.99,
        tau=0.1,
    )
    parameter.grad = torch.tensor([0.5, -0.25])
    optimizer.step()
    optimizer.set_detector_state_for_test(
        fast_ema=0.8,
        slow_ema=0.98,
        admit=True,
        bad_streak=1,
        good_streak=0,
    )
    return optimizer, parameter


def test_rejected_step_holds_only_registered_carriers() -> None:
    m_optimizer, m_parameter = _forced_optimizer("m")
    mv_optimizer, mv_parameter = _forced_optimizer("mv")
    m_before = copy.deepcopy(m_optimizer.state[m_parameter])
    mv_before = copy.deepcopy(mv_optimizer.state[mv_parameter])
    gradient = torch.tensor([-1.0, 0.5])
    m_parameter.grad = gradient.clone()
    mv_parameter.grad = gradient.clone()
    m_optimizer.step()
    mv_optimizer.step()
    assert m_optimizer.last_diagnostics["admit"] is False
    assert mv_optimizer.last_diagnostics["admit"] is False
    assert torch.equal(m_parameter, mv_parameter)
    assert torch.equal(m_optimizer.state[m_parameter]["exp_avg"], m_before["exp_avg"])
    assert torch.equal(mv_optimizer.state[mv_parameter]["exp_avg"], mv_before["exp_avg"])
    assert not torch.equal(m_optimizer.state[m_parameter]["exp_avg_sq"], m_before["exp_avg_sq"])
    assert torch.equal(mv_optimizer.state[mv_parameter]["exp_avg_sq"], mv_before["exp_avg_sq"])
    assert m_optimizer.state[m_parameter]["step"].item() == 2
    assert mv_optimizer.state[mv_parameter]["step"].item() == 2


def test_default_reachability_witness_switches_off_and_on() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ProtectAdamW([parameter], protection_target="m")
    states = [optimizer.observe_score_for_test(score) for score in [-1.0] * 10 + [1.0] * 36]
    assert states[0]["admit"] is True
    assert states[1]["admit"] is False
    assert states[44]["admit"] is False
    assert states[45]["admit"] is True


def test_device_detector_matches_host_reference() -> None:
    scores = [-1.0] * 10 + [1.0] * 36
    host = ProtectAdamW(
        [torch.nn.Parameter(torch.tensor([1.0]))], protection_target="m"
    )
    device = ProtectAdamW(
        [torch.nn.Parameter(torch.tensor([1.0]))], protection_target="m"
    )
    expected = [host.observe_score_for_test(score) for score in scores]
    actual = [device.observe_score_device_for_test(score) for score in scores]
    assert actual == expected


def test_step_has_no_direct_scalar_materialization() -> None:
    assert ".item()" not in inspect.getsource(ProtectAdamW.step)


def test_epoch_diagnostics_materialize_and_reset() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ProtectAdamW([parameter], protection_target="m")
    for gradient in (0.2, -0.1, 0.3, -0.2, 0.4):
        parameter.grad = torch.tensor([gradient])
        optimizer.step()
    summary = optimizer.diagnostics_summary(reset=True)
    assert summary["successful_steps"] == 5
    assert summary["admitted_steps"] + summary["rejected_steps"] == 5
    assert optimizer.diagnostics_summary()["successful_steps"] == 0


def test_device_commit_supports_multiple_parameter_shapes() -> None:
    first = torch.nn.Parameter(torch.ones(3, 2))
    second = torch.nn.Parameter(torch.ones(4))
    optimizer = ProtectAdamW([first, second], protection_target="mv")
    first.grad = torch.full_like(first, 0.2)
    second.grad = torch.full_like(second, -0.1)
    optimizer.step()
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_parameter_update_is_independent_of_rejection() -> None:
    rejected, rejected_parameter = _forced_optimizer("m")
    admitted, admitted_parameter = _forced_optimizer("m")
    admitted.set_detector_state_for_test(
        fast_ema=1.0,
        slow_ema=1.0,
        admit=True,
        bad_streak=0,
        good_streak=0,
    )
    gradient = torch.tensor([-1.0, 0.5])
    rejected_parameter.grad = gradient.clone()
    admitted_parameter.grad = gradient.clone()
    rejected.step()
    admitted.step()
    assert rejected.last_diagnostics["admit"] is False
    assert admitted.last_diagnostics["admit"] is True
    assert torch.equal(rejected_parameter, admitted_parameter)


def test_resume_is_bit_exact_and_rejects_mismatched_target() -> None:
    optimizer, parameter = _forced_optimizer("mv")
    checkpoint = copy.deepcopy(optimizer.state_dict())
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = ProtectAdamW([resumed_parameter], lr=1e-2, weight_decay=0.1, protection_target="mv")
    resumed.load_state_dict(checkpoint)
    for gradient in (torch.tensor([-1.0, 0.5]), torch.tensor([0.2, -0.4])):
        parameter.grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()
        optimizer.step()
        resumed.step()
        assert torch.equal(parameter, resumed_parameter)
        assert optimizer.protect_state_dict() == resumed.protect_state_dict()
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                optimizer.state[parameter][key], resumed.state[resumed_parameter][key]
            )

    wrong_parameter = torch.nn.Parameter(parameter.detach().clone())
    wrong = ProtectAdamW([wrong_parameter], lr=1e-2, weight_decay=0.1, protection_target="m")
    before = copy.deepcopy(wrong.state_dict())
    with pytest.raises(ValueError, match="protection_target"):
        wrong.load_state_dict(checkpoint)
    assert wrong.state_dict() == before


@pytest.mark.parametrize("bad_gradient", [float("nan"), float("inf")])
def test_nonfinite_gradient_fails_before_mutation(bad_gradient: float) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ProtectAdamW([parameter], protection_target="m")
    before_parameter = parameter.detach().clone()
    before_state = copy.deepcopy(optimizer.state_dict())
    parameter.grad = torch.tensor([bad_gradient])
    with pytest.raises(FloatingPointError, match="non-finite"):
        optimizer.step()
    assert torch.equal(parameter, before_parameter)
    assert optimizer.state_dict() == before_state


def test_cadam_writes_standard_moments_but_masks_adaptive_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    optimizer = CAdamW([parameter], lr=0.1, weight_decay=0.0, betas=(0.9, 0.999))
    parameter.grad = torch.tensor([1.0, 1.0])
    optimizer.step()
    before = parameter.detach().clone()
    before_m = optimizer.state[parameter]["exp_avg"].clone()
    parameter.grad = torch.tensor([-0.01, 1.0])
    optimizer.step()
    assert parameter[0].item() == before[0].item()
    assert parameter[1].item() < before[1].item()
    assert not torch.equal(optimizer.state[parameter]["exp_avg"], before_m)
