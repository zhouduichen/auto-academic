from __future__ import annotations

import copy
import inspect

import pytest
import torch

from experiments.m1_optimizers import ProtectM11AdamW


def _assert_adam_state_equal(
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
    left_parameters: list[torch.nn.Parameter],
    right_parameters: list[torch.nn.Parameter],
) -> None:
    for left_parameter, right_parameter in zip(
        left_parameters, right_parameters, strict=True
    ):
        assert torch.equal(left_parameter, right_parameter)
        left_state = left.state.get(left_parameter, {})
        right_state = right.state.get(right_parameter, {})
        assert set(left_state) == set(right_state)
        for key in left_state:
            assert torch.equal(left_state[key], right_state[key])


def _forced_m11_rejection() -> tuple[ProtectM11AdamW, torch.nn.Parameter]:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = ProtectM11AdamW(
        [parameter], lr=1e-2, weight_decay=0.1, warmup_steps=0
    )
    parameter.grad = torch.tensor([0.5, -0.25])
    optimizer.step()
    optimizer.set_detector_state_for_test(
        mu=0.0, scale=0.1, previous_rejected=False, steps=1
    )
    return optimizer, parameter


def _m11_after_fixture_steps() -> tuple[ProtectM11AdamW, torch.nn.Parameter]:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ProtectM11AdamW(
        [parameter], lr=1e-2, weight_decay=0.1, warmup_steps=1
    )
    for gradient in (torch.tensor([0.2]), torch.tensor([-0.4])):
        parameter.grad = gradient.clone()
        optimizer.step()
    return optimizer, parameter


def test_m11_warmup_matches_adamw_and_rejection_is_single_step() -> None:
    reference_p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    candidate_p = torch.nn.Parameter(reference_p.detach().clone())
    reference = torch.optim.AdamW(
        [reference_p], lr=1e-2, foreach=False, fused=False
    )
    candidate = ProtectM11AdamW([candidate_p], lr=1e-2, warmup_steps=2)
    for gradient in (torch.tensor([0.2, -0.1]), torch.tensor([-0.1, 0.3])):
        reference_p.grad = gradient.clone()
        candidate_p.grad = gradient.clone()
        reference.step()
        candidate.step()
    _assert_adam_state_equal(reference, candidate, [reference_p], [candidate_p])
    candidate.set_detector_state_for_test(
        mu=0.0, scale=0.1, previous_rejected=False, steps=2
    )
    states = [
        candidate.observe_score_for_test(-1.0),
        candidate.observe_score_for_test(-1.0),
    ]
    assert states[0]["reject"] is True
    assert states[1]["reject"] is False


def test_m11_rejection_holds_only_first_moment() -> None:
    optimizer, parameter = _forced_m11_rejection()
    before = copy.deepcopy(optimizer.state[parameter])
    parameter.grad = torch.tensor([-1.0, 0.5])
    optimizer.step()
    after = optimizer.state[parameter]
    assert optimizer.last_diagnostics["admit"] is False
    assert torch.equal(after["exp_avg"], before["exp_avg"])
    assert not torch.equal(after["exp_avg_sq"], before["exp_avg_sq"])
    assert after["step"].item() == before["step"].item() + 1


def test_m11_disabled_protection_is_bit_exact_adamw() -> None:
    reference_p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    candidate_p = torch.nn.Parameter(reference_p.detach().clone())
    reference = torch.optim.AdamW(
        [reference_p], lr=1e-2, weight_decay=0.1, foreach=False, fused=False
    )
    candidate = ProtectM11AdamW(
        [candidate_p],
        lr=1e-2,
        weight_decay=0.1,
        warmup_steps=0,
        protection_enabled=False,
    )
    for gradient in (torch.tensor([0.2, -0.1]), torch.tensor([-0.4, 0.2])):
        reference_p.grad = gradient.clone()
        candidate_p.grad = gradient.clone()
        reference.step()
        candidate.step()
    _assert_adam_state_equal(reference, candidate, [reference_p], [candidate_p])


def test_m11_resume_is_bit_exact() -> None:
    optimizer, parameter = _m11_after_fixture_steps()
    payload = copy.deepcopy(optimizer.state_dict())
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = ProtectM11AdamW(
        [resumed_parameter], lr=1e-2, weight_decay=0.1, warmup_steps=1
    )
    resumed.load_state_dict(payload)
    for gradient in (torch.tensor([-0.4]), torch.tensor([0.3])):
        parameter.grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()
        optimizer.step()
        resumed.step()
    assert torch.equal(parameter, resumed_parameter)
    assert (
        optimizer.state_dict()["protect_state"]
        == resumed.state_dict()["protect_state"]
    )


def test_m11_checkpoint_is_exact_and_failed_load_is_atomic() -> None:
    optimizer, _parameter = _m11_after_fixture_steps()
    expected_keys = {
        "schema_version",
        "mechanism_id",
        "protection_enabled",
        "alpha",
        "threshold",
        "warmup_steps",
        "mu",
        "scale",
        "previous_rejected",
        "successful_steps",
    }
    assert set(optimizer.protect_state_dict()) == expected_keys
    before = copy.deepcopy(optimizer.state_dict())
    invalid = copy.deepcopy(before)
    invalid["protect_state"]["scale"] = 0.0
    with pytest.raises(ValueError, match="scale"):
        optimizer.load_state_dict(invalid)
    assert optimizer.state_dict() == before


@pytest.mark.parametrize("bad_gradient", [float("nan"), float("inf")])
def test_m11_nonfinite_gradient_fails_before_mutation(bad_gradient: float) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ProtectM11AdamW([parameter])
    before_parameter = parameter.detach().clone()
    before_state = copy.deepcopy(optimizer.state_dict())
    parameter.grad = torch.tensor([bad_gradient])
    with pytest.raises(FloatingPointError, match="non-finite"):
        optimizer.step()
    assert torch.equal(parameter, before_parameter)
    assert optimizer.state_dict() == before_state


def test_m11_diagnostics_materialize_and_reset() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ProtectM11AdamW([parameter], warmup_steps=1)
    for gradient in (0.2, -0.1, 0.3):
        parameter.grad = torch.tensor([gradient])
        optimizer.step()
    summary = optimizer.diagnostics_summary(reset=True)
    assert summary["successful_steps"] == 3
    assert summary["admitted_steps"] + summary["rejected_steps"] == 3
    assert summary["detector_successful_steps"] == 3
    assert optimizer.diagnostics_summary()["successful_steps"] == 0


def test_m11_cuda_step_has_no_direct_item() -> None:
    assert ".item()" not in inspect.getsource(ProtectM11AdamW._next_detector_device)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"alpha": 1.0}, "configuration"),
        ({"threshold": 0.0}, "configuration"),
        ({"warmup_steps": -1}, "configuration"),
    ],
)
def test_m11_rejects_invalid_configuration(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ProtectM11AdamW([torch.nn.Parameter(torch.tensor([1.0]))], **options)
