from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Envelope(StrictModel):
    api_version: Literal["1.0"]
    request_id: str = Field(min_length=1)


class HealthResponse(Envelope):
    status: Literal["ok", "degraded"]


class MetaResponse(Envelope):
    server_version: str
    minimum_client_version: str
    features: set[str]


class StatusData(StrictModel):
    node_id: str
    state: Literal["ready", "degraded", "stopped"]
    active_tasks: int = Field(ge=0)
    waiting_approval: int = Field(ge=0)


class StatusResponse(Envelope):
    data: StatusData


class ProjectSummary(StrictModel):
    project_id: str
    name: str
    source_url: HttpUrl
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_platform: str


class ProjectListResponse(Envelope):
    items: list[ProjectSummary]
    next_cursor: str | None


class TaskSummary(StrictModel):
    task_id: str
    project_id: str
    title: str
    status: str
    version: int = Field(ge=1)


class TaskListResponse(Envelope):
    items: list[TaskSummary]
    next_cursor: str | None


ExperimentState = Literal[
    "submitted",
    "policy-validating",
    "waiting-approval",
    "policy-rejected",
    "queued",
    "running",
    "failed",
    "succeeded",
    "cancelled",
]


class CandidatePatch(StrictModel):
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patch: str = Field(min_length=1, max_length=1_000_000)

    @model_validator(mode="after")
    def validate_patch_hash(self) -> "CandidatePatch":
        if sha256(self.patch.encode()).hexdigest() != self.patch_sha256:
            raise ValueError("patch_sha256 does not match UTF-8 patch bytes")
        return self


class ExperimentMatrix(StrictModel):
    seeds: list[int] = Field(min_length=1, max_length=32)
    time_budget_seconds: int = Field(ge=30, le=7200)
    max_parallel: int = Field(ge=1, le=8)
    config_id: int | None = Field(default=None, ge=0, le=63)
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=256)
    experiment_kind: Literal["optimizer_state_m0"] | None = None
    optimizer: Literal["adamw", "sgd"] | None = None
    pulse: Literal["label_flip", "input_degradation"] | None = None
    warmup_steps: int | None = Field(default=None, ge=1, le=2000)
    replay_steps: int | None = Field(default=None, ge=1, le=512)
    probe_size: int | None = Field(default=None, ge=32, le=1000)

    @model_validator(mode="after")
    def validate_matrix(self) -> "ExperimentMatrix":
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be unique")
        if self.experiment_kind == "optimizer_state_m0":
            if self.optimizer is None:
                raise ValueError("optimizer is required for optimizer_state_m0")
            if self.pulse is None:
                raise ValueError("pulse is required for optimizer_state_m0")
            if self.config_id is not None:
                raise ValueError("config_id is forbidden for optimizer_state_m0")
        elif any(
            value is not None
            for value in (
                self.optimizer,
                self.pulse,
                self.warmup_steps,
                self.replay_steps,
                self.probe_size,
            )
        ):
            raise ValueError("M0 fields require experiment_kind=optimizer_state_m0")
        return self


class ExperimentSubmitRequest(StrictModel):
    project_id: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    title: str = Field(min_length=1, max_length=200)
    plan_id: str = Field(min_length=1, max_length=100)
    candidate: CandidatePatch
    matrix: ExperimentMatrix


class ExperimentCancelRequest(StrictModel):
    reason: str = Field(min_length=1, max_length=500)
    expected_version: int = Field(ge=1)


class SafeError(StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ExperimentResult(StrictModel):
    summary: str = Field(min_length=1)
    artifact_count: int = Field(ge=0)


class ExperimentSummary(StrictModel):
    experiment_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    title: str = Field(min_length=1)
    state: ExperimentState
    version: int = Field(ge=1)


class ExperimentDetail(ExperimentSummary):
    plan_id: str = Field(min_length=1)
    candidate_patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matrix: ExperimentMatrix
    result: ExperimentResult | None
    error: SafeError | None

    @model_validator(mode="after")
    def validate_terminal_data(self) -> "ExperimentDetail":
        if self.state == "succeeded" and self.result is None:
            raise ValueError("result is required when state is succeeded")
        if self.state == "failed" and self.error is None:
            raise ValueError("error is required when state is failed")
        if self.state not in {"succeeded", "failed"} and (self.result or self.error):
            raise ValueError("result and error are forbidden before a result terminal state")
        return self


class ExperimentEvent(StrictModel):
    event_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    event_type: str = Field(min_length=1)
    occurred_at: str = Field(min_length=1)
    state: ExperimentState


class ArtifactEntry(StrictModel):
    artifact_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1)


class ExperimentResponse(Envelope):
    data: ExperimentDetail


class ExperimentListResponse(Envelope):
    items: list[ExperimentSummary]
    next_cursor: str | None


class ExperimentEventListResponse(Envelope):
    items: list[ExperimentEvent]
    next_cursor: str | None


class ArtifactManifestResponse(Envelope):
    items: list[ArtifactEntry]


class ErrorData(StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ErrorResponse(Envelope):
    error: ErrorData
