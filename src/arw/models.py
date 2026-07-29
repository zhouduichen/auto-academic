from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


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
    next_cursor: str | None = None


class TaskSummary(StrictModel):
    task_id: str
    project_id: str
    title: str
    status: str
    version: int = Field(ge=1)


class TaskListResponse(Envelope):
    items: list[TaskSummary]
    next_cursor: str | None = None
