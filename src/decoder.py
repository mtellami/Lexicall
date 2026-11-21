"""Greedy constrained selection with bounded, explicit recovery."""

import json
import math
import sys
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.grammar import Grammar
from src.models import FunctionCall, FunctionDefinition, validate_call
from src.tokenizer import BPETokenizer

MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct")


def selected_name(prefix: bytes) -> str | None:
    """Read a completed name field from a possibly incomplete call."""
    try:
        name, _ = json.JSONDecoder().raw_decode(prefix[8:].decode("utf-8"))
    except (ValueError, UnicodeError):
        return None
    return name if isinstance(name, str) else None


def chat_prompt(
    prompt: str, definitions: list[FunctionDefinition], model: str,
    reminder: bool = False,
) -> str:
    """Build a model-specific chat prompt without private SDK helpers."""
    catalog = json.dumps([{
        "name": definition.name,
        "description": definition.description,
        "parameters": {key: value.model_dump(exclude_none=True,
                                             exclude_defaults=True)
                       for key, value in definition.parameters.items()},
    } for definition in definitions], ensure_ascii=False,
        separators=(",", ":"))
    system = (
        'Select a function and extract its arguments from the request. '
        'Do not calculate answers. Return only '
        '{"name":"function_name","parameters":{...}} with exact names '
        'and types. For regex replacement, extract the source, pattern, and '
        'replacement. Functions:\n' + catalog
    )
    if reminder:
        system += ("\nFinish a concise valid call. Include every required "
                   "parameter. Copy literal string arguments faithfully.")
    text = ("<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n")
    if model == MODELS[0]:
        text += "<think>\n\n</think>\n\n"
    return text


class Decision(BaseModel):
    """A generation choice retained for bounded backtracking."""

    prefix: bytes
    context: str
    ids: list[int]
    token_id: int
    alternative: int | None = None


class ConstrainedDecoder(BaseModel):
    """Generate calls through public SDK logits and legal token selection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    sdk: Any
    tokenizer: BPETokenizer
    grammar: Grammar
    definitions: list[FunctionDefinition]
    model: str = MODELS[0]
    max_new_tokens: int = 512
    visualize: bool = False
    trie: dict[int, Any] = Field(default_factory=dict)
    memo: dict[str, FunctionCall] = Field(default_factory=dict)
    inference_calls: int = 0
    cache_hits: int = 0

    def build_trie(self) -> None:
        """Index every ordinary vocabulary token by its raw bytes."""
        for token_id, piece in self.tokenizer.pieces.items():
            if not piece:
                continue
            node = self.trie
            for byte in piece:
                node = node.setdefault(byte, {})
            node.setdefault(-1, []).append(token_id)

    def legal_ids(self, prefix: bytes) -> list[int]:
        """Enumerate legal tokens through shared trie prefix checks."""
        if not self.trie:
            self.build_trie()
        result: list[int] = []
        stack = [(self.trie, prefix)]
        while stack:
            node, candidate = stack.pop()
            result.extend(node.get(-1, []))
            for byte, child in node.items():
                if byte != -1:
                    extended = candidate + bytes([byte])
                    if self.grammar.state(extended)[0]:
                        stack.append((child, extended))
        return result

    def candidates(self, prefix: bytes, ids: list[int]) -> list[int]:
        """Find the top two legal tokens using exact lazy logit masking.

        Descending score order permits stopping after two legal candidates:
        every higher-scoring rejected token has been masked, while all lower
        scores cannot affect the greedy choice or its recovery alternative.
        """
        self.inference_calls += 1
        logits = np.asarray(self.sdk.get_logits_from_input_ids(ids),
                            dtype=np.float64)
        if logits.ndim != 1 or not logits.size:
            raise ValueError("SDK returned empty or non-vector logits")
        if np.isnan(logits).any() or np.isposinf(logits).any():
            raise ValueError("SDK returned invalid logits")
        result: list[int] = []
        for raw_id in np.argsort(-logits, kind="stable"):
            token_id = int(raw_id)
            if not math.isfinite(float(logits[token_id])):
                break
            piece = self.tokenizer.pieces.get(token_id)
            if piece is None or not piece:
                logits[token_id] = -np.inf
                continue
            candidate = prefix + piece
            legal, complete = self.grammar.state(candidate)
            if legal and complete:
                try:
                    self.parse(candidate, "")
                except (ValueError, UnicodeError, OverflowError):
                    legal = False
            if not legal:
                logits[token_id] = -np.inf
                continue
            result.append(token_id)
            if len(result) == 2:
                break
        return result

    def parse(self, raw: bytes, prompt: str) -> FunctionCall:
        """Validate a generated call and attach the original prompt."""
        value = json.loads(raw)
        call = FunctionCall(prompt=prompt, **value)
        validate_call(call, self.definitions)
        return call

    def generate(self, prompt: str) -> FunctionCall:
        """Generate, backtrack once, then restart once before failing.

        Args:
            prompt: Original natural-language request, retained verbatim.

        Returns:
            A complete function call validated against this decoder's catalog.

        Raises:
            ValueError: Generation exhausts its bounded recovery attempts.
        """
        if prompt in self.memo:
            self.cache_hits += 1
            return self.memo[prompt].model_copy(deep=True)
        started = time.perf_counter()
        decisions: list[Decision] = []
        context = chat_prompt(prompt, self.definitions, self.model)
        prefix = b'{"name":"'
        ids = self.tokenizer.encode(context + prefix.decode())
        reason = "generation failed"
        for attempt in range(3):
            if attempt == 1:
                choice = next((item for item in reversed(decisions)
                               if item.alternative is not None), None)
                if choice is None:
                    reason = "no alternative token for backtracking"
                    continue
                assert choice.alternative is not None
                ids = choice.ids + [choice.alternative]
                context = choice.context
                prefix = (choice.prefix
                          + self.tokenizer.pieces[choice.alternative])
            elif attempt == 2:
                prefix = b'{"name":"'
                context = chat_prompt(
                    prompt, self.definitions, self.model, reminder=True
                )
                ids = self.tokenizer.encode(context + prefix.decode())
            decisions = []
            try:
                for step in range(self.max_new_tokens):
                    # Preserve model-selected tokenization throughout the
                    # name: prefilled shared prefixes can bias routing.
                    name = selected_name(prefix)
                    forced = b""
                    if name is not None:
                        forced = self.grammar.forced_suffix(prefix)
                        prefix += forced
                        name = selected_name(prefix)
                    extraction_context = context
                    if name is not None:
                        selected_definition = next(
                            item for item in self.definitions
                            if item.name == name
                        )
                        extraction_context = chat_prompt(
                            prompt, [selected_definition], self.model,
                            reminder=attempt == 2,
                        )
                    if forced or extraction_context != context:
                        context = extraction_context
                        ids = self.tokenizer.encode(
                            context + prefix.decode("utf-8")
                        )
                        if self.visualize and forced:
                            print(f"forced syntax: {forced!r}",
                                  file=sys.stderr)
                    if len(self.tokenizer.encode(
                        prefix.decode("utf-8", errors="replace")
                    )) > self.max_new_tokens:
                        raise ValueError("generated token budget exceeded")
                    if self.grammar.state(prefix)[1]:
                        call = self.parse(prefix, prompt)
                        self.memo[prompt] = call.model_copy(deep=True)
                        return call
                    choices = self.candidates(prefix, ids)
                    if not choices:
                        raise ValueError("no finite legal next-token score")
                    selected = choices[0]
                    decisions.append(Decision(
                        prefix=prefix, context=context, ids=ids[:],
                        token_id=selected,
                        alternative=choices[1] if len(choices) > 1 else None,
                    ))
                    if self.visualize:
                        allowed = len(self.legal_ids(prefix))
                        piece = self.tokenizer.pieces[selected]
                        elapsed = time.perf_counter() - started
                        print(
                            f"attempt={attempt + 1} step={step + 1} "
                            f"token={selected} bytes={piece!r} "
                            f"state=prefix:{len(prefix)}B "
                            f"allowed={allowed} elapsed={elapsed:.3f}s",
                            file=sys.stderr,
                        )
                    prefix += self.tokenizer.pieces[selected]
                    ids.append(selected)
                if self.grammar.state(prefix)[1]:
                    call = self.parse(prefix, prompt)
                    self.memo[prompt] = call.model_copy(deep=True)
                    return call
                raise ValueError(
                    f"token budget {self.max_new_tokens} exceeded"
                )
            except (ValueError, UnicodeError, OverflowError) as error:
                reason = str(error)
                if self.visualize:
                    print(f"recovery: {reason}", file=sys.stderr)
        raise ValueError(f"generation exhausted recovery attempts: {reason}")
