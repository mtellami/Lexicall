"""Validated definitions and strict recursive argument checking."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JSONValue = Any


class Schema(BaseModel):
    """Supported subset of JSON Schema for a function value."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal[
        "string", "number", "integer", "boolean", "null", "object", "array"
    ]
    description: str = ""
    enum: list[JSONValue] | None = None
    properties: dict[str, "Schema"] | None = None
    required: list[str] | None = None
    items: "Schema | None" = None
    additionalProperties: Literal[False] = False

    @model_validator(mode="after")
    def check_schema(self) -> "Schema":
        """Reject unsupported or internally inconsistent definitions."""
        if self.type == "object":
            if self.properties is None:
                raise ValueError("object schemas require properties")
            if self.required is not None:
                if len(set(self.required)) != len(self.required):
                    raise ValueError("duplicate required property")
                if not set(self.required) <= self.properties.keys():
                    raise ValueError("required contains unknown properties")
        elif self.properties is not None or self.required is not None:
            raise ValueError("properties/required need an object schema")
        if (self.type == "array") != (self.items is not None):
            raise ValueError("only array schemas must supply items")
        if self.enum is not None:
            if not self.enum:
                raise ValueError("enum must not be empty")
            for value in self.enum:
                validate_value(value, self, check_enum=False)
        return self


class FunctionDefinition(BaseModel):
    """An available function; every top-level parameter is required."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, Schema]
    returns: Schema


class Prompt(BaseModel):
    """One natural-language request."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str


class FunctionCall(BaseModel):
    """The exact public result shape."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str
    name: str
    parameters: dict[str, JSONValue]


def validate_value(
    value: JSONValue, schema: Schema, *, check_enum: bool = True
) -> None:
    """Check a value without coercing booleans, strings, or numbers.

    Args:
        value: Parsed JSON value to validate recursively.
        schema: Its declared argument schema.
        check_enum: Whether to enforce enum membership in addition to type.

    Raises:
        ValueError: A type, property, enum, or finite-number check fails.
        UnicodeError: A string contains an unpaired surrogate.
    """
    kind = schema.type
    valid = {
        "string": isinstance(value, str),
        "number": type(value) in (int, float),
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }[kind]
    if not valid:
        raise ValueError(f"expected {kind}, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numbers must be finite")
    if kind == "string":
        value.encode("utf-8", errors="strict")
    if check_enum and schema.enum is not None:
        if not any(value == item and type(value) is type(item)
                   for item in schema.enum):
            raise ValueError("value is outside enum")
    if kind == "object":
        properties = schema.properties or {}
        required = schema.required
        if required is None:
            required = list(properties)
        if not set(required) <= value.keys():
            raise ValueError("missing required properties")
        if not value.keys() <= properties.keys():
            raise ValueError("unexpected properties")
        for key, item in value.items():
            validate_value(item, properties[key])
    elif kind == "array":
        assert schema.items is not None
        for item in value:
            validate_value(item, schema.items)


def validate_call(
    call: FunctionCall, definitions: list[FunctionDefinition]
) -> None:
    """Validate the selected function and every argument recursively.

    Args:
        call: Generated call with the original prompt attached.
        definitions: Available function catalog for this request.

    Raises:
        ValueError: The name is unknown or arguments violate its schema.
    """
    definition = next((d for d in definitions if d.name == call.name), None)
    if definition is None:
        raise ValueError(f"unknown function: {call.name}")
    schema = Schema(type="object", properties=definition.parameters)
    validate_value(call.parameters, schema)
