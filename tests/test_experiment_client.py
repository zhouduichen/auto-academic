from collections.abc import Callable
from hashlib import sha256

import httpx
import pytest

from arw.client import ArwClient
from arw.config import Settings
from arw.errors import ConflictError, NetworkError, ServerError
from arw.models import (
    CandidatePatch,
    ExperimentCancelRequest,
    ExperimentMatrix,
    ExperimentSubmitRequest,
)


def settings() -> Settings:
    return Settings(server="https://node.example", api_token="top-secret")


def envelope_data(state: str = "submitted") -> dict[str, object]:
    return {
        "api_version": "1.0",
        "request_id": "req-1",
        "data": {
            "experiment_id": "exp-1",
            "project_id": "p",
            "source_commit": "a" * 40,
            "title": "t",
            "state": state,
            "version": 1,
            "plan_id": "plan",
            "candidate_patch_sha256": sha256(b"x").hexdigest(),
            "matrix": {"seeds": [1], "time_budget_seconds": 30, "max_parallel": 1},
            "result": None,
            "error": None,
        },
    }


def submit_request() -> ExperimentSubmitRequest:
    return ExperimentSubmitRequest(
        project_id="p",
        source_commit="a" * 40,
        title="t",
        plan_id="plan",
        candidate=CandidatePatch(patch_sha256=sha256(b"x").hexdigest(), patch="x"),
        matrix=ExperimentMatrix(seeds=[1], time_budget_seconds=30, max_parallel=1),
    )


def test_submit_checks_feature_and_sends_headers() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/meta":
            return httpx.Response(
                200,
                json={
                    "api_version": "1.0",
                    "request_id": "r",
                    "server_version": "1",
                    "minimum_client_version": "0.1.0",
                    "features": ["experiments.submit"],
                },
            )
        return httpx.Response(201, json=envelope_data())

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        assert client.submit_experiment(submit_request()).data.experiment_id == "exp-1"
    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[1].headers["Authorization"] == "Bearer top-secret"
    assert requests[1].headers["X-ARW-Client-Version"] == "0.1.0"
    assert requests[1].headers["Idempotency-Key"]


def test_missing_feature_sends_zero_writes() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json={
                "api_version": "1.0",
                "request_id": "r",
                "server_version": "1",
                "minimum_client_version": "0.1.0",
                "features": [],
            },
        )

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServerError, match="feature"):
            client.submit_experiment(submit_request())
    assert methods == ["GET"]


def test_incompatible_minimum_version_sends_zero_writes() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(
            200,
            json={
                "api_version": "1.0",
                "request_id": "r",
                "server_version": "1",
                "minimum_client_version": "99.0.0",
                "features": ["experiments.submit"],
            },
        )

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServerError, match="minimum client version"):
            client.submit_experiment(submit_request())
    assert methods == ["GET"]


def test_write_retry_reuses_key() -> None:
    keys: list[str] = []
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/api/v1/meta":
            return httpx.Response(
                200,
                json={
                    "api_version": "1.0",
                    "request_id": "r",
                    "server_version": "1",
                    "minimum_client_version": "0.1.0",
                    "features": ["experiments.cancel"],
                },
            )
        posts += 1
        keys.append(request.headers["Idempotency-Key"])
        if posts == 1:
            raise httpx.ConnectError("reset", request=request)
        return httpx.Response(200, json=envelope_data(state="cancelled"))

    with ArwClient(
        settings(), transport=httpx.MockTransport(handler), sleeper=lambda _: None
    ) as client:
        client.cancel_experiment(
            "exp-1", ExperimentCancelRequest(reason="stop", expected_version=1)
        )
    assert len(keys) == 2 and keys[0] == keys[1]


def test_separate_write_calls_use_distinct_idempotency_keys() -> None:
    keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/meta":
            return httpx.Response(
                200,
                json={
                    "api_version": "1.0",
                    "request_id": "r",
                    "server_version": "1",
                    "minimum_client_version": "0.1.0",
                    "features": ["experiments.submit"],
                },
            )
        keys.append(request.headers["Idempotency-Key"])
        return httpx.Response(201, json=envelope_data())

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        client.submit_experiment(submit_request())
        client.submit_experiment(submit_request())
    assert len(keys) == 2
    assert keys[0] != keys[1]


def test_get_retries_and_query_is_exact() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, request=request)
        assert request.url.path == "/api/v1/experiments"
        assert dict(request.url.params) == {"limit": "5", "cursor": "next"}
        return httpx.Response(
            200, json={"api_version": "1.0", "request_id": "r", "items": [], "next_cursor": None}
        )

    with ArwClient(
        settings(), transport=httpx.MockTransport(handler), sleeper=lambda _: None
    ) as client:
        client.list_experiments(limit=5, cursor="next")
    assert calls == 3


@pytest.mark.parametrize("status,error", [(302, ServerError), (409, ConflictError)])
def test_http_failures_are_mapped(status: int, error: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request)

    with ArwClient(
        settings(), transport=httpx.MockTransport(handler), sleeper=lambda _: None
    ) as client:
        with pytest.raises(error):
            client.get_experiment("exp-1")


def test_transport_failure_becomes_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret details", request=request)

    with ArwClient(
        settings(), transport=httpx.MockTransport(handler), sleeper=lambda _: None
    ) as client:
        with pytest.raises(NetworkError, match="unable to reach"):
            client.get_experiment("exp-1")


@pytest.mark.parametrize(
    "failure",
    [
        lambda request: httpx.ConnectError("TLS certificate verify failed", request=request),
        lambda request: httpx.ReadTimeout("read timed out", request=request),
    ],
)
def test_tls_and_timeout_failures_are_safe_network_errors(
    failure: Callable[[httpx.Request], httpx.RequestError],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure(request)

    with ArwClient(
        settings(), transport=httpx.MockTransport(handler), sleeper=lambda _: None
    ) as client:
        with pytest.raises(NetworkError, match="unable to reach") as captured:
            client.get_experiment("exp-1")
    assert "certificate" not in str(captured.value).lower()
    assert "timed out" not in str(captured.value).lower()


def test_response_validation_is_strict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "api_version": "1.0",
                "request_id": "r",
                "items": [],
                "next_cursor": None,
                "extra": True,
            },
        )

    with ArwClient(settings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServerError, match="invalid response"):
            client.list_experiments()
