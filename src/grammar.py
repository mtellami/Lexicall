"""Compile function schemas to a byte-level incremental JSON grammar."""

import json
from typing import Any

import regex
from pydantic import BaseModel, ConfigDict, Field

from src.models import FunctionDefinition, Schema

# Canonical generated JSON removes syntactic choices that carry no meaning.
WS = b""
HEX = rb"[0-9a-fA-F]"
UNICODE_ESCAPE = (
    rb"u(?:[0-9a-cA-Ce-fE-F]" + HEX + rb"{3}|[dD][0-7]" + HEX
    + rb"{2}|[dD][89aAbB]" + HEX
    + rb"{2}\\u[dD][c-fC-F]" + HEX + rb"{2})"
)
UTF8 = (
    rb"[\xc2-\xdf][\x80-\xbf]|"
    rb"\xe0[\xa0-\xbf][\x80-\xbf]|"
    rb"[\xe1-\xec\xee-\xef][\x80-\xbf]{2}|"
    rb"\xed[\x80-\x9f][\x80-\xbf]|"
    rb"\xf0[\x90-\xbf][\x80-\xbf]{2}|"
    rb"[\xf1-\xf3][\x80-\xbf]{3}|"
    rb"\xf4[\x80-\x8f][\x80-\xbf]{2}"
)
STRING = (rb'"(?:[\x20-\x21\x23-\x5b\x5d-\x7f]|' + UTF8
          + rb'|\\(?:["\\/bfnrt]|' + UNICODE_ESCAPE + rb'))*"')
INTEGER = rb"-?(?:0|[1-9][0-9]*)"
NUMBER = INTEGER + rb"(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"


def literal(value: Any) -> bytes:
    """Escape a canonical JSON value as literal regex bytes."""
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return bytes(regex.escape(encoded))


def alternatives(parts: list[bytes]) -> bytes:
    """Combine regex alternatives without a capturing group."""
    return b"(?:" + b"|".join(parts) + b")"


def object_pattern(
    properties: dict[str, Schema], required: list[str]
) -> bytes:
    """Allow optional properties in definition order with valid commas."""
    keys = list(properties)
    entries = [literal(key) + WS + b":" + WS
               + schema_pattern(properties[key]) for key in keys]
    choices: list[bytes] = []
    if not required:
        choices.append(b"")
    for first in range(len(keys)):
        if any(key in required for key in keys[:first]):
            break
        tail = entries[first]
        for index in range(first + 1, len(keys)):
            entry = WS + b"," + WS + entries[index]
            tail += (entry if keys[index] in required
                     else b"(?:" + entry + b")?")
        choices.append(tail)
    return b"\\{" + WS + alternatives(choices) + WS + b"\\}"


def schema_pattern(schema: Schema) -> bytes:
    """Compile the supported schema subset into a regular grammar."""
    if schema.enum is not None:
        return alternatives([literal(value) for value in schema.enum])
    simple = {"string": STRING, "integer": INTEGER, "number": NUMBER,
              "boolean": rb"(?:true|false)", "null": b"null"}
    if schema.type in simple:
        return simple[schema.type]
    if schema.type == "object":
        properties = schema.properties or {}
        required = schema.required
        return object_pattern(properties, list(properties)
                              if required is None else required)
    assert schema.items is not None
    item = schema_pattern(schema.items)
    return (rb"\[" + WS + b"(?:" + item + b"(?:" + WS + b"," + WS
            + item + b")*)?" + WS + rb"\]")


class Grammar(BaseModel):
    """Prefix membership and completion checks, bounded by an LRU cache."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    expression: Any
    cache: dict[bytes, tuple[bool, bool]] = Field(default_factory=dict)

    @classmethod
    def compile(cls, definitions: list[FunctionDefinition]) -> "Grammar":
        """Compile name-dependent parameter schemas for the whole catalog."""
        branches = []
        for definition in definitions:
            params = object_pattern(definition.parameters,
                                    list(definition.parameters))
            branches.append(
                rb'\{' + WS + literal("name") + WS + b":" + WS
                + literal(definition.name) + WS + b"," + WS
                + literal("parameters") + WS + b":" + WS + params
                + WS + rb"\}"
            )
        return cls(expression=regex.compile(alternatives(branches)))

    def state(self, prefix: bytes) -> tuple[bool, bool]:
        """Return (legal prefix, complete document), allowing partial UTF-8."""
        if prefix not in self.cache:
            match = self.expression.fullmatch(prefix, partial=True)
            if len(self.cache) >= 16384:
                self.cache.clear()
            self.cache[prefix] = (match is not None,
                                  match is not None and not match.partial)
        return self.cache[prefix]

    def forced_suffix(self, prefix: bytes) -> bytes:
        """Find syntax bytes with exactly one legal continuation."""
        output = bytearray()
        current = prefix
        while not self.state(current)[1]:
            choices = []
            for byte in range(256):
                if self.state(current + bytes([byte]))[0]:
                    choices.append(byte)
                    if len(choices) > 1:
                        break
            if len(choices) != 1:
                break
            output.append(choices[0])
            current += bytes([choices[0]])
        # Diverging Unicode literals can share an incomplete byte prefix.
        while output:
            try:
                (prefix + output).decode("utf-8")
                break
            except UnicodeDecodeError:
                output.pop()
        return bytes(output)
