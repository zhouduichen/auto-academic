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
_DIAGNOSTIC_KEYS = (
    "successful_steps",
    "admitted_steps",
    "rejected_steps",
    "transitions_off",
    "transitions_on",
)


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
        self._device_detector: dict[str, Tensor] | None = None
        self._step_cache: dict[nn.Parameter, int] = {}
        self._diagnostic_counters: Tensor | None = None
        self._last_device_diagnostics: dict[str, Tensor] = {}

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
        self._device_detector = None

    def _ensure_device_detector(self, device: torch.device) -> dict[str, Tensor]:
        state = self._device_detector
        if state is not None:
            if state["fast_ema"].device != device:
                raise ValueError("detector device changed")
            return state
        state = {
            "fast_ema": torch.tensor(self.fast_ema, dtype=torch.float32, device=device),
            "slow_ema": torch.tensor(self.slow_ema, dtype=torch.float32, device=device),
            "admit": torch.tensor(self.admit, dtype=torch.bool, device=device),
            "bad_streak": torch.tensor(bool(self.bad_streak), dtype=torch.bool, device=device),
            "good_streak": torch.tensor(
                bool(self.good_streak), dtype=torch.bool, device=device
            ),
        }
        self._device_detector = state
        return state

    def _next_detector_device(self, score: Tensor) -> dict[str, Tensor]:
        if score.dtype != torch.float32 or score.numel() != 1:
            raise TypeError("detector score must be a scalar FP32 tensor")
        state = self._ensure_device_detector(score.device)
        fast = state["fast_ema"] * self.alpha_fast + score * (1 - self.alpha_fast)
        slow = state["slow_ema"] * self.alpha_slow + score * (1 - self.alpha_slow)
        difference = fast - slow
        admit = state["admit"]
        bad = state["bad_streak"]
        good = state["good_streak"]
        below = difference <= -self.tau
        above = difference >= self.tau
        turn_off = admit & below & bad
        turn_on = (~admit) & above & good
        return {
            "fast_ema": fast,
            "slow_ema": slow,
            "d": difference,
            "admit": torch.where(admit, ~turn_off, turn_on),
            "bad_streak": admit & below & (~bad),
            "good_streak": (~admit) & above & (~good),
            "transition_off": turn_off,
            "transition_on": turn_on,
        }

    def _commit_device_detector(self, state: dict[str, Tensor]) -> None:
        self._device_detector = {
            key: state[key]
            for key in ("fast_ema", "slow_ema", "admit", "bad_streak", "good_streak")
        }

    def _sync_detector_to_host(self) -> None:
        state = self._device_detector
        if state is None:
            return
        self.fast_ema = float(state["fast_ema"].item())
        self.slow_ema = float(state["slow_ema"].item())
        self.admit = bool(state["admit"].item())
        self.bad_streak = int(state["bad_streak"].item())
        self.good_streak = int(state["good_streak"].item())

    @staticmethod
    def _materialize_device_detector(state: dict[str, Tensor]) -> dict[str, object]:
        transition = "none"
        if bool(state["transition_off"].item()):
            transition = "off"
        elif bool(state["transition_on"].item()):
            transition = "on"
        return {
            "fast_ema": float(state["fast_ema"].item()),
            "slow_ema": float(state["slow_ema"].item()),
            "d": float(state["d"].item()),
            "admit": bool(state["admit"].item()),
            "bad_streak": int(state["bad_streak"].item()),
            "good_streak": int(state["good_streak"].item()),
            "transition": transition,
        }

    def observe_score_for_test(self, score: float) -> dict[str, object]:
        self._sync_detector_to_host()
        state = self._next_detector(score)
        self._commit_detector(state)
        return copy.deepcopy(state)

    def observe_score_device_for_test(self, score: float) -> dict[str, object]:
        value = torch.tensor(score, dtype=torch.float32)
        state = self._next_detector_device(value)
        self._commit_device_detector(state)
        self._sync_detector_to_host()
        return self._materialize_device_detector(state)

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
        self._device_detector = None

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
        self._sync_detector_to_host()
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
            self._step_cache.clear()
            for group in self.param_groups:
                _validate_group(group)
                for parameter in group["params"]:
                    if parameter in self.state:
                        _exp_avg, _exp_avg_sq, step = _read_state(
                            self.state[parameter], parameter
                        )
                        self._step_cache[parameter] = step
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

    def _record_device_diagnostics(
        self,
        detector: dict[str, Tensor],
        score: Tensor,
        gradient_squared_norm: Tensor,
        active_parameter_count: int,
    ) -> None:
        device = score.device
        counters = self._diagnostic_counters
        if counters is None:
            counters = torch.zeros(5, dtype=torch.int64, device=device)
            self._diagnostic_counters = counters
        elif counters.device != device:
            raise ValueError("diagnostic device changed")
        admitted = detector["admit"].to(dtype=torch.int64)
        counters.add_(
            torch.stack(
                (
                    torch.ones_like(admitted),
                    admitted,
                    1 - admitted,
                    detector["transition_off"].to(torch.int64),
                    detector["transition_on"].to(torch.int64),
                )
            )
        )
        self._last_device_diagnostics = {
            "q": score,
            "d": detector["d"],
            "gradient_norm": torch.sqrt(gradient_squared_norm),
            "active_parameter_count": torch.tensor(
                active_parameter_count, dtype=torch.int64, device=device
            ),
        }

    def _materialize_cpu_step_diagnostics(
        self,
        detector: dict[str, Tensor],
        score: Tensor,
        gradient_squared_norm: Tensor,
        active_parameter_count: int,
    ) -> dict[str, object]:
        state = self._materialize_device_detector(detector)
        return {
            "q": float(score.item()),
            "F": state["fast_ema"],
            "S": state["slow_ema"],
            "d": state["d"],
            "admit": state["admit"],
            "transition": state["transition"],
            "active_parameter_count": active_parameter_count,
            "gradient_norm": math.sqrt(float(gradient_squared_norm.item())),
            "overflow_skipped": False,
            "protection_target": self.protection_target,
        }

    def diagnostics_summary(self, *, reset: bool = False) -> dict[str, object]:
        self._sync_detector_to_host()
        counters = self._diagnostic_counters
        result: dict[str, object] = {
            "successful_steps": 0,
            "admitted_steps": 0,
            "rejected_steps": 0,
            "transitions_off": 0,
            "transitions_on": 0,
            "F": self.fast_ema,
            "S": self.slow_ema,
            "admit": self.admit,
            "protection_target": self.protection_target,
        }
        if counters is not None:
            for key, value in zip(_DIAGNOSTIC_KEYS, counters, strict=True):
                result[key] = int(value.item())
        for key, value in self._last_device_diagnostics.items():
            result[key] = (
                int(value.item()) if key == "active_parameter_count" else float(value.item())
            )
        if reset and counters is not None:
            counters.zero_()
        return result

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
                state = self.state.get(parameter, {})
                if not state:
                    exp_avg = torch.zeros_like(parameter)
                    exp_avg_sq = torch.zeros_like(parameter)
                    step = 0
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step = self._step_cache.get(parameter)
                    if step is None:
                        _exp_avg, _exp_avg_sq, step = _read_state(state, parameter)
                        self._step_cache[parameter] = step
                active.append((group, parameter, gradient, exp_avg, exp_avg_sq, step))
        if not active:
            self.last_diagnostics = {}
            return loss
        if device is None:
            raise RuntimeError("active parameter device was not resolved")
        gradients = [gradient for _group, _parameter, gradient, *_ in active]
        finite = torch.isfinite(
            torch.stack(torch._foreach_norm(gradients, ord=float("inf")))
        ).all()
        if not bool(finite):
            raise FloatingPointError("non-finite gradient")
        dot = torch.zeros((), dtype=torch.float32, device=device)
        gg = torch.zeros((), dtype=torch.float32, device=device)
        rr = torch.zeros((), dtype=torch.float32, device=device)
        for group, _parameter, gradient, exp_avg, _exp_avg_sq, step in active:
            beta1 = float(group["betas"][0])
            reference = torch.zeros_like(exp_avg) if step == 0 else exp_avg / (1 - beta1**step)
            dot = dot + torch.sum(gradient * reference, dtype=torch.float32)
            gg = gg + torch.sum(gradient * gradient, dtype=torch.float32)
            rr = rr + torch.sum(reference * reference, dtype=torch.float32)
        one = torch.ones((), dtype=torch.float32, device=device)
        zero = torch.zeros((), dtype=torch.float32, device=device)
        denominator = torch.sqrt(gg * rr)
        safe_denominator = torch.where(denominator == 0, one, denominator)
        cosine = torch.clamp(dot / safe_denominator, -1, 1)
        score = torch.where(rr == 0, one, torch.where(gg == 0, zero, cosine))
        detector = self._next_detector_device(score)
        staged: list[
            tuple[nn.Parameter, Tensor, Tensor, Tensor, Tensor, float, float, float, int]
        ] = []
        for group, parameter, gradient, exp_avg, exp_avg_sq, step in active:
            beta1, beta2 = (float(value) for value in group["betas"])
            next_step = step + 1
            exp_avg_candidate = exp_avg.clone(memory_format=torch.preserve_format)
            exp_avg_sq_candidate = exp_avg_sq.clone(memory_format=torch.preserve_format)
            exp_avg_candidate.lerp_(gradient, 1 - beta1)
            exp_avg_sq_candidate.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            bias_correction1 = 1 - beta1**next_step
            bias_correction2 = 1 - beta2**next_step
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            step_size = lr / bias_correction1
            denominator = exp_avg_sq_candidate.sqrt() / math.sqrt(bias_correction2)
            denominator.add_(float(group["eps"]))
            staged.append(
                (
                    parameter,
                    exp_avg,
                    exp_avg_sq,
                    exp_avg_candidate,
                    exp_avg_sq_candidate,
                    denominator,
                    lr,
                    step_size,
                    weight_decay,
                    next_step,
                )
            )
        admitted = detector["admit"]
        for (
            parameter,
            exp_avg,
            exp_avg_sq,
            exp_avg_candidate,
            exp_avg_sq_candidate,
            denominator,
            lr,
            step_size,
            weight_decay,
            step,
        ) in staged:
            state = self.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            parameter.mul_(1 - lr * weight_decay)
            parameter.addcdiv_(exp_avg_candidate, denominator, value=-step_size)
            state["step"].fill_(float(step))
            self._step_cache[parameter] = step
            if not self.protection_enabled:
                state["exp_avg"].copy_(exp_avg_candidate)
                state["exp_avg_sq"].copy_(exp_avg_sq_candidate)
                continue
            state["exp_avg"].copy_(torch.where(admitted, exp_avg_candidate, exp_avg))
            if self.protection_target == "m":
                state["exp_avg_sq"].copy_(exp_avg_sq_candidate)
            else:
                state["exp_avg_sq"].copy_(
                    torch.where(admitted, exp_avg_sq_candidate, exp_avg_sq)
                )
        self._commit_device_detector(detector)
        self._record_device_diagnostics(detector, score, gg, len(active))
        if device.type == "cpu":
            self._sync_detector_to_host()
            self.last_diagnostics = self._materialize_cpu_step_diagnostics(
                detector, score, gg, len(active)
            )
        else:
            self.last_diagnostics = {}
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
