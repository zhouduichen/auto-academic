from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from math import sqrt
from typing import Any

import torch
from torch import Tensor, nn

TensorState = dict[str, Tensor]
OptimizerState = dict[str, Any]


@dataclass(frozen=True)
class CandidateState:
    parameters: TensorState
    optimizer: OptimizerState
    loss: float
    gradient_norm: float


@dataclass(frozen=True)
class BranchState:
    parameters: TensorState
    optimizer: OptimizerState
    pulse_loss: float
    gradient_norm: float


def capture_trainable_state(model: nn.Module) -> TensorState:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model: nn.Module, state: TensorState) -> None:
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(trainable) != set(state):
        raise ValueError("trainable parameter names do not match the snapshot")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state[name])


def capture_optimizer_state(optimizer: torch.optim.Optimizer) -> OptimizerState:
    return deepcopy(optimizer.state_dict())


def restore_optimizer_state(optimizer: torch.optim.Optimizer, state: OptimizerState) -> None:
    optimizer.load_state_dict(deepcopy(state))


def restore_branch(model: nn.Module, optimizer: torch.optim.Optimizer, branch: BranchState) -> None:
    restore_trainable_state(model, branch.parameters)
    restore_optimizer_state(optimizer, branch.optimizer)


def _model_logits(output: Tensor | object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("model output must be a Tensor or expose Tensor logits")
    return logits


def _gradient_norm(model: nn.Module) -> Tensor:
    squared = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
    return squared.sqrt()


def take_candidate_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    pre_parameters: TensorState,
    pre_optimizer: OptimizerState,
    *,
    target_gradient_norm: float | None = None,
    forward_fn: Callable[[Tensor], Tensor | object] | None = None,
) -> CandidateState:
    restore_trainable_state(model, pre_parameters)
    restore_optimizer_state(optimizer, pre_optimizer)
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs) if forward_fn is None else forward_fn(inputs)
    loss = criterion(_model_logits(output), labels)
    loss.backward()
    gradient_norm = _gradient_norm(model)
    if target_gradient_norm is not None:
        if gradient_norm.item() == 0:
            raise ValueError("cannot norm-match a zero corrupt gradient")
        scale = target_gradient_norm / gradient_norm.item()
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter.grad.mul_(scale)
        gradient_norm = _gradient_norm(model)
    optimizer.step()
    return CandidateState(
        parameters=capture_trainable_state(model),
        optimizer=capture_optimizer_state(optimizer),
        loss=float(loss.detach()),
        gradient_norm=float(gradient_norm.detach()),
    )


def build_causal_branches(clean: CandidateState, corrupt: CandidateState) -> dict[str, BranchState]:
    return {
        "control": BranchState(clean.parameters, clean.optimizer, clean.loss, clean.gradient_norm),
        "parameter_only": BranchState(
            corrupt.parameters, clean.optimizer, corrupt.loss, corrupt.gradient_norm
        ),
        "state_only": BranchState(
            clean.parameters, corrupt.optimizer, clean.loss, corrupt.gradient_norm
        ),
        "full": BranchState(
            corrupt.parameters, corrupt.optimizer, corrupt.loss, corrupt.gradient_norm
        ),
    }


def compose_adamw_state(
    clean: OptimizerState,
    corrupt: OptimizerState,
    corrupt_moments: frozenset[str],
) -> OptimizerState:
    allowed = frozenset({"exp_avg", "exp_avg_sq"})
    if not corrupt_moments or not corrupt_moments <= allowed:
        raise ValueError("corrupt_moments must be a non-empty subset of AdamW moments")
    if clean.get("param_groups") != corrupt.get("param_groups"):
        raise ValueError("optimizer parameter groups do not match")
    clean_state = clean.get("state")
    corrupt_state = corrupt.get("state")
    if not isinstance(clean_state, dict) or not isinstance(corrupt_state, dict):
        raise ValueError("optimizer state is malformed")
    if set(clean_state) != set(corrupt_state):
        raise ValueError("optimizer state parameter IDs do not match")

    combined = deepcopy(clean)
    combined_state = combined["state"]
    for parameter_id in clean_state:
        clean_parameter = clean_state[parameter_id]
        corrupt_parameter = corrupt_state[parameter_id]
        combined_parameter = combined_state[parameter_id]
        if not all(
            isinstance(item, dict)
            for item in (
                clean_parameter,
                corrupt_parameter,
                combined_parameter,
            )
        ):
            raise ValueError("optimizer parameter state is malformed")
        if set(clean_parameter) != set(corrupt_parameter):
            raise ValueError("optimizer parameter-state keys do not match")
        for moment in allowed:
            left = clean_parameter.get(moment)
            right = corrupt_parameter.get(moment)
            if not isinstance(left, Tensor) or not isinstance(right, Tensor):
                raise ValueError(f"AdamW state is missing {moment}")
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"AdamW {moment} tensors do not match")
        clean_step = clean_parameter.get("step")
        corrupt_step = corrupt_parameter.get("step")
        if isinstance(clean_step, Tensor) and isinstance(corrupt_step, Tensor):
            if not torch.equal(clean_step, corrupt_step):
                raise ValueError("AdamW step counters do not match")
        elif clean_step != corrupt_step:
            raise ValueError("AdamW step counters do not match")
        for moment in corrupt_moments:
            combined_parameter[moment] = corrupt_parameter[moment].detach().clone()
    return combined


def build_state_carrier_branches(
    clean: CandidateState, corrupt: CandidateState
) -> dict[str, BranchState]:
    return {
        "control": BranchState(clean.parameters, clean.optimizer, clean.loss, clean.gradient_norm),
        "m_only": BranchState(
            clean.parameters,
            compose_adamw_state(clean.optimizer, corrupt.optimizer, frozenset({"exp_avg"})),
            clean.loss,
            corrupt.gradient_norm,
        ),
        "v_only": BranchState(
            clean.parameters,
            compose_adamw_state(clean.optimizer, corrupt.optimizer, frozenset({"exp_avg_sq"})),
            clean.loss,
            corrupt.gradient_norm,
        ),
        "state_both": BranchState(
            clean.parameters, corrupt.optimizer, clean.loss, corrupt.gradient_norm
        ),
        "parameter_only": BranchState(
            corrupt.parameters, clean.optimizer, corrupt.loss, corrupt.gradient_norm
        ),
    }


def _tensor_leaves(value: object) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value.detach().float()]
    if isinstance(value, dict):
        leaves: list[Tensor] = []
        for key in sorted(value, key=str):
            leaves.extend(_tensor_leaves(value[key]))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_tensor_leaves(item))
        return leaves
    return []


def optimizer_state_distance(
    left: OptimizerState, right: OptimizerState, *, epsilon: float = 1e-12
) -> float:
    left_tensors = _tensor_leaves(left.get("state", {}))
    right_tensors = _tensor_leaves(right.get("state", {}))
    if len(left_tensors) != len(right_tensors):
        raise ValueError("optimizer states have different tensor structures")
    if not left_tensors:
        return 0.0
    squared_difference = 0.0
    squared_reference = 0.0
    for left_tensor, right_tensor in zip(left_tensors, right_tensors, strict=True):
        if left_tensor.shape != right_tensor.shape:
            raise ValueError("optimizer states have different tensor shapes")
        squared_difference += float((left_tensor - right_tensor).square().sum())
        squared_reference += float(right_tensor.square().sum())
    return sqrt(squared_difference) / (sqrt(squared_reference) + epsilon)


def optimizer_moment_distance(
    left: OptimizerState,
    right: OptimizerState,
    moment: str,
    *,
    epsilon: float = 1e-12,
) -> float:
    if moment not in {"momentum_buffer", "exp_avg", "exp_avg_sq"}:
        raise ValueError(f"unsupported optimizer moment: {moment}")

    def moment_tensors(state: OptimizerState) -> list[Tensor]:
        tensors: list[Tensor] = []
        raw_state = state.get("state", {})
        if not isinstance(raw_state, dict):
            raise ValueError("optimizer state is malformed")
        for parameter_id in sorted(raw_state, key=str):
            parameter_state = raw_state[parameter_id]
            if isinstance(parameter_state, dict) and isinstance(
                parameter_state.get(moment), Tensor
            ):
                tensors.append(parameter_state[moment].detach().float())
        return tensors

    left_tensors = moment_tensors(left)
    right_tensors = moment_tensors(right)
    if len(left_tensors) != len(right_tensors):
        raise ValueError("optimizer moments have different tensor structures")
    if not left_tensors:
        return 0.0
    squared_difference = 0.0
    squared_reference = 0.0
    for left_tensor, right_tensor in zip(left_tensors, right_tensors, strict=True):
        if left_tensor.shape != right_tensor.shape:
            raise ValueError("optimizer moments have different tensor shapes")
        squared_difference += float((left_tensor - right_tensor).square().sum())
        squared_reference += float(right_tensor.square().sum())
    return sqrt(squared_difference) / (sqrt(squared_reference) + epsilon)


def parameter_state_distance(
    left: TensorState, right: TensorState, *, epsilon: float = 1e-12
) -> float:
    if set(left) != set(right):
        raise ValueError("parameter states have different names")
    squared_difference = 0.0
    squared_reference = 0.0
    for name in sorted(left):
        squared_difference += float((left[name].float() - right[name].float()).square().sum())
        squared_reference += float(right[name].float().square().sum())
    return sqrt(squared_difference) / (sqrt(squared_reference) + epsilon)


def trapezoid_auc(horizons: list[int], values: list[float]) -> float:
    if len(horizons) != len(values) or len(horizons) < 2:
        raise ValueError("AUC needs matching horizon/value sequences of length at least two")
    if any(right <= left for left, right in pairwise(horizons)):
        raise ValueError("horizons must be strictly increasing")
    return sum(
        (right_h - left_h) * (left_y + right_y) / 2
        for (left_h, right_h), (left_y, right_y) in zip(
            pairwise(horizons), pairwise(values), strict=True
        )
    )


def recovery_half_life(horizons: list[int], excess: list[float]) -> int | None:
    if len(horizons) != len(excess) or not horizons:
        raise ValueError("half-life needs matching non-empty sequences")
    peak_index = max(range(len(excess)), key=lambda index: abs(excess[index]))
    peak = abs(excess[peak_index])
    if peak == 0:
        return 0
    threshold = peak / 2
    for horizon, value in zip(horizons[peak_index + 1 :], excess[peak_index + 1 :], strict=True):
        if abs(value) <= threshold:
            return horizon
    return None
