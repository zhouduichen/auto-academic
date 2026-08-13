"""Paired clean/noisy exposure primitives for the causal transport audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR100

from experiments import causal_attribution_replay as replay
from experiments import m1_calibration_clean as clean
from experiments import m1_calibration_noisy as noisy
from experiments import repair_handoff_models as repair_models
from experiments.m0_optimizer_state import m0_run as m0
from experiments.m1_noisy_data import _sha, validate_public
from experiments.m1_pilot import source_commit as current_source_commit
from src.arw import m0_core

FROZEN_DOSES = (1, 8, 64, 1250)
SNAPSHOT_SCHEMA = "causal-transport-exposure/1"
SNAPSHOT_MANIFEST = "EXPOSURE_SNAPSHOTS.json"


@dataclass(frozen=True)
class TransportRequest:
    method: Literal["adamw", "cadam"]
    seed: int
    dose: int
    warmup_steps: int
    continuation: Literal["clean", "noisy"]
    config: clean.Config
    data_dir: Path
    image_store: Path
    tuning_store: Path
    noisy_bundle: Path
    output: Path
    source_commit: str
    contract_sha256: str
    device: str = "cuda"
    probe_size: int = 256
    expected_input_binding: dict[str, object] | None = None

    @classmethod
    def testing(cls, root: Path, *, dose: int) -> TransportRequest:
        config = clean.Config(
            seed=401,
            augmentation_seed=10_401,
            epochs=1,
            batch_size=2,
            workers=0,
            device="cpu",
            method="adamw",
        )
        return cls(
            method="adamw",
            seed=401,
            dose=dose,
            warmup_steps=2,
            continuation="clean",
            config=config,
            data_dir=root / "data",
            image_store=root / "image",
            tuning_store=root / "tuning",
            noisy_bundle=root / "training",
            output=root / "output",
            source_commit="0" * 40,
            contract_sha256="a" * 64,
            device="cpu",
            probe_size=2,
            expected_input_binding=None,
        )


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_request(request: TransportRequest, *, allow_test_values: bool = False) -> None:
    if not _plain_int(request.dose) or request.dose not in FROZEN_DOSES:
        raise ValueError(f"dose must be one of {FROZEN_DOSES}")
    if request.method not in {"adamw", "cadam"}:
        raise ValueError("method must be adamw or cadam")
    if request.continuation not in {"clean", "noisy"}:
        raise ValueError("continuation must be clean or noisy")
    if not _plain_int(request.seed) or request.seed <= 0:
        raise ValueError("seed must be a positive integer")
    if request.config.seed != request.seed or request.config.method != request.method:
        raise ValueError("request and config disagree")
    if request.config.device != request.device:
        raise ValueError("request and config device disagree")
    if request.config.augmentation_seed != 10_000 + request.seed:
        raise ValueError("augmentation_seed must equal 10000 + seed")
    if not _plain_int(request.probe_size) or request.probe_size <= 0:
        raise ValueError("probe_size must be a positive integer")
    if not allow_test_values and request.warmup_steps != 500:
        raise ValueError("warmup_steps must equal the frozen value 500")
    if not allow_test_values and request.source_commit != current_source_commit():
        raise RuntimeError("request source commit is not current")
    if len(request.contract_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in request.contract_sha256
    ):
        raise ValueError("contract_sha256 must be lowercase SHA-256")
    if request.expected_input_binding is not None:
        required = {
            "noise_bundle_sha256",
            "image_store_sha256",
            "tuning_store_sha256",
            "source_commit",
            "test_loaded",
        }
        if (
            set(request.expected_input_binding) != required
            or request.expected_input_binding.get("test_loaded") is not False
        ):
            raise ValueError("expected input binding is malformed")


class PairedTransportDataset(Dataset[tuple[Tensor, int, int]]):
    """Return one augmentation with both clean and noisy labels."""

    def __init__(
        self,
        image_store: Path,
        training: Path,
        *,
        clean_targets: np.ndarray,
        augmentation_seed: int,
    ) -> None:
        ids = np.load(image_store / "sample_ids.npy", allow_pickle=False)
        public_ids = np.load(training / "sample_ids.npy", allow_pickle=False)
        if not np.array_equal(ids, public_ids):
            raise RuntimeError("public training/image IDs differ")
        self.images = np.load(image_store / "images.npy", mmap_mode="r", allow_pickle=False)
        self.noisy_labels = np.load(
            training / "noisy_labels.npy", mmap_mode="r", allow_pickle=False
        )
        self.order = np.load(image_store / "train_order.npy", allow_pickle=False)
        originals = np.asarray([int(value.decode("ascii")[-5:]) for value in ids])
        self.row_by_original = {int(value): row for row, value in enumerate(originals)}
        self.clean_targets = np.asarray(clean_targets)
        if len(self.clean_targets) <= int(originals.max(initial=-1)):
            raise RuntimeError("clean targets do not cover all public IDs")
        if len(self.images) != len(ids) or len(self.noisy_labels) != len(ids):
            raise RuntimeError("public store lengths do not match IDs")
        self.augmentation_seed = augmentation_seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, position: int) -> tuple[Tensor, int, int]:
        original = int(self.order[position])
        try:
            row = self.row_by_original[original]
        except KeyError as error:
            raise RuntimeError("train order references an unknown public ID") from error
        rng = np.random.default_rng(
            np.random.SeedSequence([self.augmentation_seed, self.epoch, original])
        )
        image = Image.fromarray(np.asarray(self.images[row]))
        tensor = m0._transform(
            image,
            int(rng.integers(0, 9)),
            int(rng.integers(0, 9)),
            bool(rng.integers(0, 2)),
        )
        return tensor, int(self.clean_targets[original]), int(self.noisy_labels[row])


def run_paired_exposure(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    common: m0_core.BranchState,
    common_rng: dict[str, object],
    batches: list[tuple[Tensor, Tensor, Tensor]],
    *,
    noisy: bool,
    device: torch.device,
) -> replay.CheckpointState:
    if not batches:
        raise ValueError("exposure batches must not be empty")
    m0_core.restore_branch(model, optimizer, common)
    m0.restore_rng_state(common_rng)
    for images, clean_labels, noisy_labels in batches:
        labels = noisy_labels if noisy else clean_labels
        replay._training_step(
            model,
            optimizer,
            criterion,
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
    return replay.CheckpointState(
        deepcopy(m0_core.capture_trainable_state(model)),
        deepcopy(m0_core.capture_optimizer_state(optimizer)),
        0,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _snapshot_payload(
    state: replay.CheckpointState, contract_sha256: str
) -> dict[str, object]:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "contract_sha256": contract_sha256,
        "parameters": state.parameters,
        "optimizer": state.optimizer,
        "next_epoch": state.next_epoch,
    }


def write_snapshot_pair(
    output: Path,
    clean_state: replay.CheckpointState,
    noisy_state: replay.CheckpointState,
    *,
    contract_sha256: str,
) -> None:
    replay.validate_checkpoint_pair(clean_state, noisy_state)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean": output / "clean-exposure.pt",
        "noisy": output / "noisy-exposure.pt",
    }
    temporary_paths: dict[str, Path] = {}
    for name, path in paths.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            _snapshot_payload(clean_state if name == "clean" else noisy_state, contract_sha256),
            temporary,
        )
        temporary_paths[name] = temporary
    for name, path in paths.items():
        os.replace(temporary_paths[name], path)
    _atomic_json(
        output / SNAPSHOT_MANIFEST,
        {
            "schema": SNAPSHOT_SCHEMA,
            "contract_sha256": contract_sha256,
            "clean": {"file": paths["clean"].name, "sha256": _sha256(paths["clean"])},
            "noisy": {"file": paths["noisy"].name, "sha256": _sha256(paths["noisy"])},
            "test_loaded": False,
        },
    )


def _load_snapshot(path: Path, contract_sha256: str) -> replay.CheckpointState:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"snapshot cannot be loaded: {path.name}") from error
    required = {"schema", "contract_sha256", "parameters", "optimizer", "next_epoch"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("snapshot schema mismatch")
    if payload["schema"] != SNAPSHOT_SCHEMA or payload["contract_sha256"] != contract_sha256:
        raise RuntimeError("snapshot contract mismatch")
    if not isinstance(payload["parameters"], dict) or not isinstance(payload["optimizer"], dict):
        raise RuntimeError("snapshot state is malformed")
    if not _plain_int(payload["next_epoch"]) or payload["next_epoch"] < 0:
        raise RuntimeError("snapshot epoch is malformed")
    return replay.CheckpointState(
        parameters=payload["parameters"],
        optimizer=payload["optimizer"],
        next_epoch=payload["next_epoch"],
    )


def load_snapshot_pair(
    output: Path, *, contract_sha256: str
) -> tuple[replay.CheckpointState, replay.CheckpointState]:
    manifest_path = output / SNAPSHOT_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("snapshot manifest cannot be loaded") from error
    required = {"schema", "contract_sha256", "clean", "noisy", "test_loaded"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise RuntimeError("snapshot manifest schema mismatch")
    if (
        manifest["schema"] != SNAPSHOT_SCHEMA
        or manifest["contract_sha256"] != contract_sha256
        or manifest["test_loaded"] is not False
    ):
        raise RuntimeError("snapshot manifest contract mismatch")
    states: list[replay.CheckpointState] = []
    for name in ("clean", "noisy"):
        record = manifest[name]
        expected_name = f"{name}-exposure.pt"
        if not isinstance(record, dict) or set(record) != {"file", "sha256"}:
            raise RuntimeError("snapshot manifest record is malformed")
        if record["file"] != expected_name:
            raise RuntimeError("snapshot filename mismatch")
        path = output / expected_name
        try:
            actual_sha = _sha256(path)
        except OSError as error:
            raise RuntimeError("snapshot file is missing") from error
        if actual_sha != record["sha256"]:
            raise RuntimeError("snapshot checksum mismatch")
        states.append(_load_snapshot(path, contract_sha256))
    replay.validate_checkpoint_pair(states[0], states[1])
    return states[0], states[1]


BatchPlan = tuple[int, tuple[int, ...]]


def _batch_plans(
    dataset_size: int, batch_size: int, seed: int, count: int
) -> list[BatchPlan]:
    if dataset_size <= 0 or batch_size <= 0 or count <= 0:
        raise ValueError("batch-plan dimensions must be positive")
    plans: list[BatchPlan] = []
    epoch = 0
    while len(plans) < count:
        rng = np.random.default_rng(np.random.SeedSequence([20_260_811, seed, epoch]))
        order = rng.permutation(dataset_size)
        for offset in range(0, dataset_size, batch_size):
            positions = tuple(int(value) for value in order[offset : offset + batch_size])
            plans.append((epoch, positions))
            if len(plans) == count:
                break
        epoch += 1
    return plans


def _materialize_plan(
    dataset: PairedTransportDataset, plan: BatchPlan
) -> tuple[Tensor, Tensor, Tensor]:
    epoch, positions = plan
    dataset.epoch = epoch
    rows = [dataset[position] for position in positions]
    return (
        torch.stack([row[0] for row in rows]),
        torch.tensor([row[1] for row in rows], dtype=torch.long),
        torch.tensor([row[2] for row in rows], dtype=torch.long),
    )


def _apply_plans(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataset: PairedTransportDataset,
    plans: list[BatchPlan],
    *,
    noisy_labels: bool,
    device: torch.device,
) -> None:
    for plan in plans:
        images, clean_labels, noisy = _materialize_plan(dataset, plan)
        replay._training_step(
            model,
            optimizer,
            criterion,
            images.to(device, non_blocking=True),
            (noisy if noisy_labels else clean_labels).to(device, non_blocking=True),
        )


def _optimizer_clock(state: replay.CheckpointState) -> int:
    raw = state.optimizer.get("state")
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("exposure optimizer state is empty")
    clocks: set[int] = set()
    for parameter_state in raw.values():
        if not isinstance(parameter_state, dict) or "step" not in parameter_state:
            raise RuntimeError("exposure optimizer clock is missing")
        value = parameter_state["step"]
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise RuntimeError("exposure optimizer clock is malformed")
            clocks.add(int(value.detach().cpu()))
        elif _plain_int(value):
            clocks.add(value)
        else:
            raise RuntimeError("exposure optimizer clock is malformed")
    if len(clocks) != 1:
        raise RuntimeError("exposure optimizer clocks diverged")
    return clocks.pop()


def _request_record(request: TransportRequest) -> dict[str, object]:
    value = asdict(request)
    # Round-trip through JSON so tuples in ``clean.Config`` compare exactly to
    # the lists written in a completed summary after a process restart.
    return json.loads(
        json.dumps(
            value,
            default=lambda item: str(item) if isinstance(item, Path) else item,
            sort_keys=True,
        )
    )


def _validate_existing_bundle(request: TransportRequest) -> dict[str, object] | None:
    summary_path = request.output / "summary.json"
    manifest_path = request.output / "sha256_manifest.json"
    if not summary_path.is_file() and not manifest_path.is_file():
        return None
    if not summary_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("partial causal transport result exists")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("completed causal transport result is unreadable") from error
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "succeeded"
        or summary.get("source_commit") != request.source_commit
        or summary.get("contract_sha256") != request.contract_sha256
        or summary.get("request") != _request_record(request)
        or summary.get("test_loaded") is not False
        or not isinstance(manifest, dict)
    ):
        raise RuntimeError("completed causal transport result does not match request")
    for name, expected in manifest.items():
        path = request.output / name
        if not isinstance(expected, str) or not path.is_file() or _sha256(path) != expected:
            raise RuntimeError("completed causal transport checksum failed")
    return summary


def _probe_and_tuning(
    request: TransportRequest,
) -> tuple[list[tuple[Tensor, Tensor]], DataLoader]:
    tuning = noisy.TuningDataset(request.tuning_store)
    if not 1 <= request.probe_size <= len(tuning):
        raise ValueError("probe_size is outside the tuning store")
    rng = np.random.default_rng(
        np.random.SeedSequence([20_260_811, request.seed, request.dose])
    )
    indices = sorted(rng.choice(len(tuning), request.probe_size, replace=False).tolist())
    probe_loader = DataLoader(
        Subset(tuning, indices),
        batch_size=request.config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    probe = [(images.clone(), labels.clone()) for images, labels in probe_loader]
    tuning_loader = DataLoader(
        tuning,
        batch_size=request.config.batch_size,
        shuffle=False,
        num_workers=request.config.workers,
        pin_memory=request.device == "cuda",
    )
    return probe, tuning_loader


def _validate_input_binding(request: TransportRequest) -> dict[str, object]:
    public = validate_public(request.noisy_bundle, request.image_store, request.tuning_store)
    actual = {
        "noise_bundle_sha256": _sha(request.noisy_bundle / "manifest.json"),
        "image_store_sha256": _sha(request.image_store / "manifest.json"),
        "tuning_store_sha256": _sha(request.tuning_store / "manifest.json"),
        "source_commit": public["source_commit"],
        "test_loaded": False,
    }
    if request.expected_input_binding is not None and actual != request.expected_input_binding:
        raise RuntimeError("frozen input binding mismatch")
    if request.expected_input_binding is None and public["source_commit"] != request.source_commit:
        raise RuntimeError("unfrozen input source commit mismatch")
    return actual


def run_bundle(request: TransportRequest) -> dict[str, object]:
    """Run or resume one paired-exposure crossed continuation bundle."""
    validate_request(request)
    existing = _validate_existing_bundle(request)
    if existing is not None:
        return existing
    started = time.monotonic()
    input_binding = _validate_input_binding(request)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    m0.seed_everything(request.seed)
    device = torch.device(request.device)
    provenance = m0.build_provenance(device)
    if provenance["source_commit"] != request.source_commit:
        raise RuntimeError("runtime provenance does not match request")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for formal transport evidence")
        torch.cuda.reset_peak_memory_stats(device)

    cifar = CIFAR100(root=request.data_dir, train=True, download=False, transform=None)
    dataset = PairedTransportDataset(
        request.image_store,
        request.noisy_bundle,
        clean_targets=np.asarray(cifar.targets),
        augmentation_seed=request.config.augmentation_seed,
    )
    total_plans = request.warmup_steps + request.dose + 512
    plans = _batch_plans(len(dataset), request.config.batch_size, request.seed, total_plans)
    warmup_plans = plans[: request.warmup_steps]
    exposure_plans = plans[request.warmup_steps : request.warmup_steps + request.dose]
    continuation_plans = plans[request.warmup_steps + request.dose :]

    model = repair_models.build_transport_model(request.config).to(device)
    optimizer = clean._build_optimizer(model, request.config)
    criterion = nn.CrossEntropyLoss()
    manifest_path = request.output / SNAPSHOT_MANIFEST
    if manifest_path.is_file():
        clean_state, noisy_state = load_snapshot_pair(
            request.output, contract_sha256=request.contract_sha256
        )
    else:
        unexpected = [
            request.output / "clean-exposure.pt",
            request.output / "noisy-exposure.pt",
        ]
        if any(path.exists() for path in unexpected):
            raise RuntimeError("partial exposure snapshot pair exists")
        _apply_plans(
            model,
            optimizer,
            criterion,
            dataset,
            warmup_plans,
            noisy_labels=False,
            device=device,
        )
        common = m0_core.BranchState(
            deepcopy(m0_core.capture_trainable_state(model)),
            deepcopy(m0_core.capture_optimizer_state(optimizer)),
            0.0,
            0.0,
        )
        common_rng = m0.capture_rng_state()
        m0_core.restore_branch(model, optimizer, common)
        m0.restore_rng_state(common_rng)
        _apply_plans(
            model,
            optimizer,
            criterion,
            dataset,
            exposure_plans,
            noisy_labels=False,
            device=device,
        )
        clean_state = replay.CheckpointState(
            deepcopy(m0_core.capture_trainable_state(model)),
            deepcopy(m0_core.capture_optimizer_state(optimizer)),
            0,
        )
        m0_core.restore_branch(model, optimizer, common)
        m0.restore_rng_state(common_rng)
        _apply_plans(
            model,
            optimizer,
            criterion,
            dataset,
            exposure_plans,
            noisy_labels=True,
            device=device,
        )
        noisy_state = replay.CheckpointState(
            deepcopy(m0_core.capture_trainable_state(model)),
            deepcopy(m0_core.capture_optimizer_state(optimizer)),
            0,
        )
        replay.validate_checkpoint_pair(clean_state, noisy_state)
        expected_clock = request.warmup_steps + request.dose
        if _optimizer_clock(clean_state) != expected_clock or _optimizer_clock(
            noisy_state
        ) != expected_clock:
            raise RuntimeError("exposure optimizer clock does not match frozen dose")
        write_snapshot_pair(
            request.output,
            clean_state,
            noisy_state,
            contract_sha256=request.contract_sha256,
        )

    replay.validate_checkpoint_pair(clean_state, noisy_state)
    m0_core.restore_trainable_state(model, clean_state.parameters)
    m0_core.restore_optimizer_state(optimizer, clean_state.optimizer)
    cached: list[tuple[Tensor, Tensor]] = []
    for plan in continuation_plans:
        images, clean_labels, noisy_labels = _materialize_plan(dataset, plan)
        cached.append(
            (
                images,
                noisy_labels if request.continuation == "noisy" else clean_labels,
            )
        )
    branches = replay.build_factorial_branches(clean_state, noisy_state)
    probe, tuning_loader = _probe_and_tuning(request)
    rows, measurement = replay.measure_factorial_replay(
        model, optimizer, branches, cached, probe, tuning_loader, device
    )
    elapsed = time.monotonic() - started
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    summary: dict[str, object] = {
        "schema": "causal-transport-bundle/1",
        "status": "succeeded",
        "request": _request_record(request),
        "source_commit": request.source_commit,
        "contract_sha256": request.contract_sha256,
        "input_binding": input_binding,
        "uv_lock_sha256": provenance["uv_lock_sha256"],
        "endpoints": measurement["endpoints"],
        "effects": measurement["effects"],
        "wall_clock_seconds": elapsed,
        "peak_vram_gb": peak_vram,
        "test_loaded": False,
    }
    request.output.mkdir(parents=True, exist_ok=True)
    replay._write_jsonl(request.output / "trajectory_metrics.jsonl", rows)
    clean._write_json(request.output / "factorial_effects.json", measurement["effects"])
    clean._write_json(
        request.output / "replay_manifest.json",
        {
            "source_commit": request.source_commit,
            "contract_sha256": request.contract_sha256,
            "common_rng_sha256": measurement["common_rng_sha256"],
            "horizons": list(replay.HORIZONS),
            "warmup_steps": request.warmup_steps,
            "dose": request.dose,
            "scheduler": "none",
            "test_loaded": False,
        },
    )
    clean._write_json(request.output / "summary.json", summary)
    sha_manifest = {
        path.name: _sha256(path)
        for path in sorted(request.output.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    clean._write_json(request.output / "sha256_manifest.json", sha_manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.request.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("request JSON must be an object")
    config_raw = raw.pop("config")
    if not isinstance(config_raw, dict):
        raise ValueError("request config must be an object")
    config_raw["betas"] = tuple(config_raw.get("betas", (0.9, 0.999)))
    config_raw["checkpoint_epochs"] = tuple(config_raw.get("checkpoint_epochs", ()))
    for key in ("data_dir", "image_store", "tuning_store", "noisy_bundle", "output"):
        raw[key] = Path(raw[key])
    request = TransportRequest(config=clean.Config(**config_raw), **raw)
    print(json.dumps(run_bundle(request), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
