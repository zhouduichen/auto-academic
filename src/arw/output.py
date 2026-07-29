import json
from typing import Any

from pydantic import BaseModel


def stable_json(kind: str, value: BaseModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        {"kind": kind, "response": payload, "schema_version": "1"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
