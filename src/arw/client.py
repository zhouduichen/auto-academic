import ssl
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, TypeVar
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import ValidationError

from arw.config import Settings
from arw.errors import NetworkError, ServerError, map_http_error
from arw.models import (
    ArtifactManifestResponse,
    ExperimentCancelRequest,
    ExperimentEventListResponse,
    ExperimentListResponse,
    ExperimentResponse,
    ExperimentSubmitRequest,
    MetaResponse,
    StrictModel,
)

ResponseModel = TypeVar("ResponseModel", bound=StrictModel)
RETRYABLE_STATUS = {502, 503, 504}


def _client_version() -> str:
    try:
        return version("auto-academic")
    except PackageNotFoundError:
        return "0.1.0"


def _version_key(value: str) -> tuple[int, int, int]:
    core = value.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise ServerError("server returned an invalid minimum client version")
    numbers = [int(part) for part in parts] + [0, 0]
    return numbers[0], numbers[1], numbers[2]


class ArwClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._sleeper = sleeper
        verify: ssl.SSLContext | bool = True
        if settings.ca_bundle is not None:
            verify = ssl.create_default_context(cafile=str(settings.ca_bundle))
        timeout = httpx.Timeout(
            connect=settings.connect_timeout,
            read=settings.read_timeout,
            write=settings.read_timeout,
            pool=settings.connect_timeout,
        )
        self._client = httpx.Client(
            base_url=settings.server,
            headers={
                "Authorization": f"Bearer {settings.api_token.get_secret_value()}",
                "X-ARW-Client-Version": _client_version(),
                "Accept": "application/json",
            },
            follow_redirects=False,
            timeout=timeout,
            transport=transport,
            verify=verify,
        )

    def __enter__(self) -> "ArwClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _validate(self, response: httpx.Response, model: type[ResponseModel]) -> ResponseModel:
        if response.is_redirect or not response.is_success:
            raise map_http_error(response)
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ServerError(
                "server returned an invalid response",
                request_id=response.headers.get("X-Request-ID"),
                status=response.status_code,
            ) from exc

    def _get(
        self,
        path: str,
        model: type[ResponseModel],
        *,
        params: dict[str, Any] | None = None,
    ) -> ResponseModel:
        for attempt in range(3):
            try:
                response = self._client.get(path, params=params)
            except httpx.RequestError as exc:
                if attempt < 2:
                    self._sleeper(0.1 * (attempt + 1))
                    continue
                raise NetworkError("unable to reach the ARW server") from exc
            if response.status_code in RETRYABLE_STATUS and attempt < 2:
                self._sleeper(0.1 * (attempt + 1))
                continue
            return self._validate(response, model)
        raise AssertionError("unreachable")

    def _post(
        self,
        path: str,
        payload: StrictModel,
        model: type[ResponseModel],
    ) -> ResponseModel:
        idempotency_key = str(uuid4())
        for attempt in range(2):
            try:
                response = self._client.post(
                    path,
                    json=payload.model_dump(mode="json"),
                    headers={"Idempotency-Key": idempotency_key},
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt == 0:
                    self._sleeper(0.1)
                    continue
                raise NetworkError("unable to reach the ARW server") from exc
            except httpx.RequestError as exc:
                raise NetworkError("unable to reach the ARW server") from exc
            return self._validate(response, model)
        raise AssertionError("unreachable")

    def meta(self) -> MetaResponse:
        return self._get("/api/v1/meta", MetaResponse)

    def require_feature(self, feature: str) -> None:
        metadata = self.meta()
        if _version_key(_client_version()) < _version_key(metadata.minimum_client_version):
            raise ServerError(
                f"server requires minimum client version {metadata.minimum_client_version}"
            )
        if feature not in metadata.features:
            raise ServerError(f"server does not advertise required feature: {feature}")

    def submit_experiment(self, request: ExperimentSubmitRequest) -> ExperimentResponse:
        self.require_feature("experiments.submit")
        return self._post("/api/v1/experiments", request, ExperimentResponse)

    def list_experiments(
        self, *, limit: int = 100, cursor: str | None = None
    ) -> ExperimentListResponse:
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        return self._get("/api/v1/experiments", ExperimentListResponse, params=params)

    def get_experiment(self, experiment_id: str) -> ExperimentResponse:
        identifier = quote(experiment_id, safe="")
        return self._get(f"/api/v1/experiments/{identifier}", ExperimentResponse)

    def list_events(
        self, experiment_id: str, *, limit: int = 100, cursor: str | None = None
    ) -> ExperimentEventListResponse:
        identifier = quote(experiment_id, safe="")
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        return self._get(
            f"/api/v1/experiments/{identifier}/events",
            ExperimentEventListResponse,
            params=params,
        )

    def cancel_experiment(
        self, experiment_id: str, request: ExperimentCancelRequest
    ) -> ExperimentResponse:
        self.require_feature("experiments.cancel")
        identifier = quote(experiment_id, safe="")
        return self._post(f"/api/v1/experiments/{identifier}/cancel", request, ExperimentResponse)

    def get_artifact_manifest(self, experiment_id: str) -> ArtifactManifestResponse:
        identifier = quote(experiment_id, safe="")
        return self._get(f"/api/v1/experiments/{identifier}/artifacts", ArtifactManifestResponse)

    @contextmanager
    def stream_artifact(self, experiment_id: str, artifact_id: str) -> Iterator[httpx.Response]:
        experiment = quote(experiment_id, safe="")
        artifact = quote(artifact_id, safe="")
        path = f"/api/v1/experiments/{experiment}/artifacts/{artifact}/content"
        timeout = httpx.Timeout(
            connect=self.settings.connect_timeout,
            read=self.settings.artifact_read_timeout,
            write=self.settings.read_timeout,
            pool=self.settings.connect_timeout,
        )
        try:
            with self._client.stream("GET", path, timeout=timeout) as response:
                if response.is_redirect or not response.is_success:
                    raise map_http_error(response)
                yield response
        except httpx.RequestError as exc:
            raise NetworkError("unable to stream the artifact") from exc
