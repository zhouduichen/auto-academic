"""Frozen optimizers for the M1 Protect-M candidate Pilot."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

type ParamsT = Iterable[Tensor] | Iterable[dict[str, Any]]
ProtectionTarget = Literal["m", "mv"]


def _validate_common_options(
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    amsgrad: bool,
    maximize: bool,
    foreach: bool | None,
    fused: bool | None,
    capturable: bool,
    differentiable: bool,
) -> None:
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("lr must be finite and nonnegative")
    if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
        raise ValueError("betas must be finite values in [0, 1)")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if amsgrad or maximize or bool(foreach) or bool(fused) or capturable or differentiable:
        raise ValueError("unsupported AdamW option")


def _validate_group(group: dict[str, Any]) -> None:
    _validate_common_options(
        lr=float(group["lr"]),
        betas=tuple(group["betas"]),
        eps=float(group["eps"]),
        weight_decay=float(group["weight_decay"]),
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
    )


def _validate_parameter_and_gradient(parameter: Tensor, gradient: Tensor) -> None:
    if parameter.dtype != torch.float32 or gradient.dtype != torch.float32:
        raise TypeError("parameters and gradients must be FP32")
    if gradient.is_sparse:
        raise RuntimeError("sparse gradients are unsupported")
    if gradient.shape != parameter.shape:
        raise ValueError("gradient shape differs from parameter shape")
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("non-finite gradient")


def _read_state(state: dict[str, Any], parameter: Tensor) -> tuple[Tensor, Tensor, int]:
    if not state:
        return torch.zeros_like(parameter), torch.zeros_like(parameter), 0
    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("invalid AdamW state keys")
    step = state["step"]
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    if not isinstance(step, Tensor) or step.numel() != 1 or step.dtype != torch.float32:
        raise TypeError("step must be a scalar FP32 tensor")
    step_value = float(step.item())
    if not math.isfinite(step_value) or step_value < 0 or not step_value.is_integer():
        raise ValueError("step must be a finite nonnegative integer")
    for name, value in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
        if not isinstance(value, Tensor) or value.shape != parameter.shape:
            raise ValueError(f"invalid {name} shape")
        if value.dtype != torch.float32 or value.device != parameter.device:
            raise TypeError(f"invalid {name} dtype or device")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite {name}")
    return exp_avg, exp_avg_sq, int(step_value)


class ProtectAdamW(Optimizer):
    """AdamW whose gate controls only persistent moment writes."""

    mechanism_id = "protect_persistent_adamw_v1"
    schema_version = 1

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        protection_target: ProtectionTarget,
        protection_enabled: bool = True,
        alpha_fast: float = 0.9,
        alpha_slow: float = 0.99,
        tau: float = 0.1,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool | None = False,
        fused: bool | None = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        _validate_common_options(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            fused=fused,
            capturable=capturable,
            differentiable=differentiable,
        )
        if protection_target not in {"m", "mv"}:
            raise ValueError("protection_target must be 'm' or 'mv'")
        if not all(math.isfinite(value) for value in (alpha_fast, alpha_slow, tau)):
            raise ValueError("detector values must be finite")
        if not 0 <= alpha_fast < alpha_slow < 1 or not 0 < tau < 2:
            raise ValueError("invalid detector constraints")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            _validate_group(group)
        self.protection_target: ProtectionTarget = protection_target
        self.protection_enabled = bool(protection_enabled)
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.tau = float(tau)
        self.fast_ema = 1.0
        self.slow_ema = 1.0
        self.admit = True
        self.bad_streak = 0
        self.good_streak = 0
        self.last_diagnostics: dict[str, object] = {}

    def _next_detector(self, score: float) -> dict[str, object]:
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("detector score must be finite and in [-1, 1]")
        fast = float(
            (
                torch.tensor(self.alpha_fast, dtype=torch.float32)
                * torch.tensor(self.fast_ema, dtype=torch.float32)
                + torch.tensor(1 - self.alpha_fast, dtype=torch.float32)
                * torch.tensor(score, dtype=torch.float32)
            ).item()
        )
        slow = float(
            (
                torch.tensor(self.alpha_slow, dtype=torch.float32)
                * torch.tensor(self.slow_ema, dtype=torch.float32)
                + torch.tensor(1 - self.alpha_slow, dtype=torch.float32)
                * torch.tensor(score, dtype=torch.float32)
            ).item()
        )
        difference = float(
            (
                torch.tensor(fast, dtype=torch.float32)
                - torch.tensor(slow, dtype=torch.float32)
            ).item()
        )
        admit = self.admit
        bad_streak = self.bad_streak
        good_streak = self.good_streak
        transition = "none"
        if admit:
            good_streak = 0
            if difference <= -self.tau:
                if bad_streak == 1:
                    admit = False
                    bad_streak = 0
                    transition = "off"
                else:
                    bad_streak = 1
            else:
                bad_streak = 0
        else:
            bad_streak = 0
            if difference >= self.tau:
                if good_streak == 1:
                    admit = True
                    good_streak = 0
                    transition = "on"
                else:
                    good_streak = 1
            else:
                good_streak = 0
        return {
            "fast_ema": fast,
            "slow_ema": slow,
            "d": difference,
            "admit": admit,
            "bad_streak": bad_streak,
            "good_streak": good_streak,
            "transition": transition,
        }

    def _commit_detector(self, state: dict[str, object]) -> None:
        self.fast_ema = float(state["fast_ema"])
        self.slow_ema = float(state["slow_ema"])
        self.admit = bool(state["admit"])
        self.bad_streak = int(state["bad_streak"])
        self.good_streak = int(state["good_streak"])

    def observe_score_for_test(self, score: float) -> dict[str, object]:
        state = self._next_detector(score)
        self._commit_detector(state)
        return copy.deepcopy(state)

    def set_detector_state_for_test(
        self,
        *,
        fast_ema: float,
        slow_ema: float,
        admit: bool,
        bad_streak: int,
        good_streak: int,
    ) -> None:
        self._validate_detector_state(
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            admit=admit,
            bad_streak=bad_streak,
            good_streak=good_streak,
        )
        self.fast_ema = float(fast_ema)
        self.slow_ema = float(slow_ema)
        self.admit = bool(admit)
        self.bad_streak = int(bad_streak)
        self.good_streak = int(good_streak)

    @staticmethod
    def _validate_detector_state(
        *,
        fast_ema: float,
        slow_ema: float,
        admit: bool,
        bad_streak: int,
        good_streak: int,
    ) -> None:
        if not all(math.isfinite(value) and -1 <= value <= 1 for value in (fast_ema, slow_ema)):
            raise ValueError("invalid detector EMA")
        if not isinstance(admit, bool):
            raise TypeError("admit must be boolean")
        if bad_streak not in {0, 1} or good_streak not in {0, 1}:
            raise ValueError("invalid detector streak")

    def protect_state_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mechanism_id": self.mechanism_id,
            "protection_target": self.protection_target,
            "protection_enabled": self.protection_enabled,
            "alpha_fast": self.alpha_fast,
            "alpha_slow": self.alpha_slow,
            "tau": self.tau,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "admit": self.admit,
            "bad_streak": self.bad_streak,
            "good_streak": self.good_streak,
        }

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["protect_state"] = self.protect_state_dict()
        return payload

    def _validate_loaded_protect_state(self, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError("missing protect_state")
        expected_keys = set(self.protect_state_dict())
        if set(value) != expected_keys:
            raise ValueError("invalid protect_state keys")
        for key, expected in (
            ("schema_version", self.schema_version),
            ("mechanism_id", self.mechanism_id),
            ("protection_target", self.protection_target),
            ("protection_enabled", self.protection_enabled),
            ("alpha_fast", self.alpha_fast),
            ("alpha_slow", self.alpha_slow),
            ("tau", self.tau),
        ):
            if value[key] != expected:
                raise ValueError(f"checkpoint {key} mismatch")
        self._validate_detector_state(
            fast_ema=float(value["fast_ema"]),
            slow_ema=float(value["slow_ema"]),
            admit=value["admit"],
            bad_streak=int(value["bad_streak"]),
            good_streak=int(value["good_streak"]),
        )
        return value

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        incoming = copy.deepcopy(state_dict)
        protect = self._validate_loaded_protect_state(incoming.pop("protect_state", None))
        before_base = super().state_dict()
        before_protect = self.protect_state_dict()
        try:
            super().load_state_dict(incoming)
            for group in self.param_groups:
                _validate_group(group)
                for parameter in group["params"]:
                    if parameter in self.state:
                        _read_state(self.state[parameter], parameter)
            self.set_detector_state_for_test(
                fast_ema=float(protect["fast_ema"]),
                slow_ema=float(protect["slow_ema"]),
                admit=bool(protect["admit"]),
                bad_streak=int(protect["bad_streak"]),
                good_streak=int(protect["good_streak"]),
            )
        except Exception:
            super().load_state_dict(before_base)
            self.set_detector_state_for_test(
                fast_ema=float(before_protect["fast_ema"]),
                slow_ema=float(before_protect["slow_ema"]),
                admit=bool(before_protect["admit"]),
                bad_streak=int(before_protect["bad_streak"]),
                good_streak=int(before_protect["good_streak"]),
            )
            raise

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        active: list[tuple[dict[str, Any], nn.Parameter, Tensor, Tensor, Tensor, int]] = []
        device: torch.device | None = None
        for group in self.param_groups:
            _validate_group(group)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                _validate_parameter_and_gradient(parameter, gradient)
                if device is None:
                    device = parameter.device
                elif parameter.device != device:
                    raise ValueError("all active parameters must use one device")
                exp_avg, exp_avg_sq, step = _read_state(self.state[parameter], parameter)
                active.append((group, parameter, gradient, exp_avg, exp_avg_sq, step))
        if not active:
            self.last_diagnostics = {}
            return loss
        if device is None:
            raise RuntimeError("active parameter device was not resolved")
        dot = torch.zeros((), dtype=torch.float32, device=device)
        gg = torch.zeros((), dtype=torch.float32, device=device)
        rr = torch.zeros((), dtype=torch.float32, device=device)
        for group, _parameter, gradient, exp_avg, _exp_avg_sq, step in active:
            beta1 = float(group["betas"][0])
            reference = torch.zeros_like(exp_avg) if step == 0 else exp_avg / (1 - beta1**step)
            dot = dot + torch.sum(gradient * reference, dtype=torch.float32)
            gg = gg + torch.sum(gradient * gradient, dtype=torch.float32)
            rr = rr + torch.sum(reference * reference, dtype=torch.float32)
        rr_value = float(rr.item())
        gg_value = float(gg.item())
        if rr_value == 0:
            score = 1.0
        elif gg_value == 0:
            score = 0.0
        else:
            score = float(torch.clamp(dot / torch.sqrt(gg * rr), -1, 1).item())
        detector = self._next_detector(score)
        staged: list[tuple[nn.Parameter, Tensor, Tensor, Tensor, int]] = []
        for group, parameter, gradient, exp_avg, exp_avg_sq, step in active:
            beta1, beta2 = (float(value) for value in group["betas"])
            next_step = step + 1
            exp_avg_candidate = exp_avg.clone(memory_format=torch.preserve_format)
            exp_avg_sq_candidate = exp_avg_sq.clone(memory_format=torch.preserve_format)
            exp_avg_candidate.lerp_(gradient, 1 - beta1)
            exp_avg_sq_candidate.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            parameter_candidate = parameter.clone(memory_format=torch.preserve_format)
            parameter_candidate.mul_(1 - float(group["lr"]) * float(group["weight_decay"]))
            bias_correction1 = 1 - beta1**next_step
            bias_correction2 = 1 - beta2**next_step
            step_size = float(group["lr"]) / bias_correction1
            denominator = exp_avg_sq_candidate.sqrt() / math.sqrt(bias_correction2)
            denominator.add_(float(group["eps"]))
            parameter_candidate.addcdiv_(
                exp_avg_candidate, denominator, value=-step_size
            )
            staged.append(
                (
                    parameter,
                    parameter_candidate,
                    exp_avg_candidate,
                    exp_avg_sq_candidate,
                    next_step,
                )
            )
        admitted = bool(detector["admit"]) or not self.protection_enabled
        for parameter, parameter_candidate, exp_avg_candidate, exp_avg_sq_candidate, step in staged:
            state = self.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            parameter.copy_(parameter_candidate)
            state["step"].fill_(float(step))
            if admitted:
                state["exp_avg"].copy_(exp_avg_candidate)
            if admitted or self.protection_target == "m":
                state["exp_avg_sq"].copy_(exp_avg_sq_candidate)
        self._commit_detector(detector)
        self.last_diagnostics = {
            "q": score,
            "F": self.fast_ema,
            "S": self.slow_ema,
            "d": float(detector["d"]),
            "admit": bool(detector["admit"]),
            "transition": str(detector["transition"]),
            "active_parameter_count": len(active),
            "gradient_norm": math.sqrt(gg_value),
            "overflow_skipped": False,
            "protection_target": self.protection_target,
        }
        return loss


class CAdamW(Optimizer):
    """AdamW-compatible reproduction of CAdam Algorithm 1 (arXiv:2411.19647v2)."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool | None = False,
        fused: bool | None = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        _validate_common_options(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            fused=fused,
            capturable=capturable,
            differentiable=differentiable,
        )
        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )
        for group in self.param_groups:
            _validate_group(group)
        self.last_diagnostics: dict[str, object] = {}

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        staged: list[tuple[nn.Parameter, Tensor, Tensor, Tensor, int, float]] = []
        aligned = total = 0
        for group in self.param_groups:
            _validate_group(group)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                _validate_parameter_and_gradient(parameter, gradient)
                exp_avg, exp_avg_sq, step = _read_state(self.state[parameter], parameter)
                beta1, beta2 = (float(value) for value in group["betas"])
                next_step = step + 1
                exp_avg_candidate = exp_avg.clone(memory_format=torch.preserve_format)
                exp_avg_sq_candidate = exp_avg_sq.clone(memory_format=torch.preserve_format)
                exp_avg_candidate.lerp_(gradient, 1 - beta1)
                exp_avg_sq_candidate.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                mask = exp_avg_candidate.mul(gradient).gt(0)
                masked_moment = exp_avg_candidate * mask
                parameter_candidate = parameter.clone(memory_format=torch.preserve_format)
                parameter_candidate.mul_(
                    1 - float(group["lr"]) * float(group["weight_decay"])
                )
                denominator = exp_avg_sq_candidate.sqrt() / math.sqrt(1 - beta2**next_step)
                denominator.add_(float(group["eps"]))
                parameter_candidate.addcdiv_(
                    masked_moment,
                    denominator,
                    value=-(float(group["lr"]) / (1 - beta1**next_step)),
                )
                staged.append(
                    (
                        parameter,
                        parameter_candidate,
                        exp_avg_candidate,
                        exp_avg_sq_candidate,
                        next_step,
                        float(mask.float().mean().item()),
                    )
                )
                aligned += int(mask.count_nonzero().item())
                total += mask.numel()
        for (
            parameter,
            parameter_candidate,
            exp_avg_candidate,
            exp_avg_sq_candidate,
            step,
            _,
        ) in staged:
            state = self.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            parameter.copy_(parameter_candidate)
            state["step"].fill_(float(step))
            state["exp_avg"].copy_(exp_avg_candidate)
            state["exp_avg_sq"].copy_(exp_avg_sq_candidate)
        self.last_diagnostics = {
            "alignment_ratio": aligned / total if total else 0.0,
            "active_parameter_count": len(staged),
        }
        return loss
