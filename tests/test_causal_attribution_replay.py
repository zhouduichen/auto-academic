from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments import causal_attribution_replay as replay


def _state(seed: int, step: int = 2) -> replay.CheckpointState:
    torch.manual_seed(seed)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
    for _ in range(step):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
    return replay.CheckpointState(
        parameters=deepcopy(model.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()),
        next_epoch=1,
    )


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_factorial_branches_cross_only_requested_carriers() -> None:
    clean = _state(1)
    noisy = _state(2)
    branches = replay.build_factorial_branches(clean, noisy)
    assert tuple(branches) == ("CC", "CN", "NC", "NN")
    assert all(
        torch.equal(branches["CN"].parameters[key], clean.parameters[key])
        for key in clean.parameters
    )
    assert all(
        torch.equal(branches["NC"].parameters[key], noisy.parameters[key])
        for key in noisy.parameters
    )
    _assert_nested_equal(branches["CN"].optimizer, noisy.optimizer)
    _assert_nested_equal(branches["NC"].optimizer, clean.optimizer)


def test_checkpoint_pair_rejects_step_mismatch_atomically() -> None:
    clean = _state(1, step=2)
    noisy = _state(2, step=3)
    with pytest.raises(ValueError, match="step counters"):
        replay.validate_checkpoint_pair(clean, noisy)


def test_checkpoint_pair_rejects_nonfinite_state() -> None:
    clean = _state(1)
    noisy = _state(2)
    first_state = next(iter(noisy.optimizer["state"].values()))
    first_state["exp_avg"].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        replay.validate_checkpoint_pair(clean, noisy)


def test_factorial_effect_formula() -> None:
    effects = replay.factorial_effects({"CC": 1.0, "CN": 2.0, "NC": 4.0, "NN": 8.0})
    assert effects == {"parameter": 4.5, "state": 2.5, "interaction": 3.0}


def test_load_checkpoint_state_keeps_only_trainable_parameters(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW([model.weight], lr=0.01, foreach=False, fused=False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "next_epoch": 1},
        path,
    )
    loaded = replay.load_checkpoint_state(model, path, torch.device("cpu"))
    assert set(loaded.parameters) == {"weight"}
    assert loaded.next_epoch == 1


def test_common_replay_core_is_bitwise_deterministic() -> None:
    clean = _state(1)
    noisy = _state(2)
    branches = replay.build_factorial_branches(clean, noisy)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
    cached = [
        (torch.tensor([[1.0, -1.0]]), torch.tensor([0])),
        (torch.tensor([[0.5, 2.0]]), torch.tensor([1])),
    ]
    criterion = torch.nn.CrossEntropyLoss()

    def measure(branch: str, horizon: int) -> dict[str, object]:
        total = sum(float(parameter.detach().sum()) for parameter in model.parameters())
        return {
            "branch": branch,
            "horizon": horizon,
            "parameter_sum": total,
            "test_loaded": False,
        }

    rng = replay.capture_common_rng()
    first = replay.run_cached_branches(
        model, optimizer, branches, cached, criterion, rng, (0, 1, 2), measure
    )
    second = replay.run_cached_branches(
        model, optimizer, branches, cached, criterion, rng, (0, 1, 2), measure
    )
    assert first == second
    assert {row["branch"] for row in first} == {"CC", "CN", "NC", "NN"}
    assert all(row["test_loaded"] is False for row in first)
