"""Strict JSON input and atomic output publication."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from src.models import FunctionCall, FunctionDefinition, Prompt


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently replacing values."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    """Reject nonstandard JSON numeric constants."""
    raise ValueError(f"invalid JSON constant: {value}")


def read_json(path: Path) -> Any:
    """Read strict UTF-8 JSON from a managed file handle."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_object,
                         parse_constant=reject_constant)


def load_inputs(
    definitions_path: Path, prompts_path: Path
) -> tuple[list[FunctionDefinition], list[Prompt]]:
    """Validate all inputs before loading expensive inference resources.

    Args:
        definitions_path: JSON array of available function definitions.
        prompts_path: JSON array of natural-language requests.

    Returns:
        Validated definitions and prompts in their original order.

    Raises:
        ValueError: JSON or schema validation fails, or names are duplicated.
        OSError: Either input cannot be opened or read.
    """
    definitions = TypeAdapter(list[FunctionDefinition]).validate_python(
        read_json(definitions_path), strict=True
    )
    prompts = TypeAdapter(list[Prompt]).validate_python(
        read_json(prompts_path), strict=True
    )
    if not definitions:
        raise ValueError("at least one function definition is required")
    if len({d.name for d in definitions}) != len(definitions):
        raise ValueError("duplicate function names")
    return definitions, prompts


def write_results(path: Path, calls: list[FunctionCall]) -> None:
    """Publish a complete batch atomically, preserving existing output.

    Args:
        path: Destination JSON file; parent directories are created as needed.
        calls: Already validated calls, in input order.

    Raises:
        OSError: Creation, flushing, or atomic publication fails.
        ValueError: A value cannot be serialized as strict JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump([call.model_dump() for call in calls], handle,
                      ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
