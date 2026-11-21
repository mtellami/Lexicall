"""Public, handwritten byte-level BPE tokenizer for the supported Qwens."""

import json
import unicodedata
from pathlib import Path
from typing import Any, Literal

import regex
from pydantic import BaseModel, ConfigDict, Field


def byte_alphabet() -> dict[int, str]:
    """Return the reversible GPT byte-to-Unicode alphabet."""
    values = list(range(33, 127))
    values += list(range(161, 173)) + list(range(174, 256))
    characters = list(values)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


class BPETokenizer(BaseModel):
    """Encode/decode without importing an existing tokenizer engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    vocab: dict[str, int]
    ranks: dict[tuple[str, str], int]
    pattern: Any
    added: dict[str, int]
    special_ids: set[int]
    normalization: Literal["NFC"] | None = None
    alphabet: dict[int, str] = Field(default_factory=byte_alphabet)
    pieces: dict[int, bytes] = Field(default_factory=dict)
    added_by_id: dict[int, str] = Field(default_factory=dict)
    cache: dict[str, list[int]] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        """Read the public tokenizer.json asset provided by the SDK.

        Args:
            path: Local asset path returned by the SDK's public method.

        Returns:
            A validated tokenizer with vocabulary, merges, and byte mappings.

        Raises:
            ValueError: The asset declares an unsupported tokenizer setting.
        """
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        model = data["model"]
        normalizer = data.get("normalizer")
        if model["type"] != "BPE" or normalizer not in (
            None, {"type": "NFC"}
        ):
            raise ValueError("unsupported tokenizer model or normalization")
        stages = data["pre_tokenizer"]
        if stages["type"] != "Sequence":
            raise ValueError("expected Split + ByteLevel pre-tokenization")
        split, byte_level = stages["pretokenizers"]
        if (split["type"] != "Split" or split.get("invert")
                or split["behavior"] != "Isolated"
                or byte_level["type"] != "ByteLevel"
                or byte_level.get("add_prefix_space")
                or byte_level.get("use_regex")):
            raise ValueError("unsupported byte-level pre-tokenizer settings")
        pattern = regex.compile(split["pattern"]["Regex"])
        merges = model["merges"]
        pairs = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
                 for m in merges]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("invalid BPE merge pair")
        added = {item["content"]: item["id"]
                 for item in data["added_tokens"]}
        for item in data["added_tokens"]:
            if any(item.get(flag) for flag in
                   ("lstrip", "rstrip", "single_word")):
                raise ValueError("unsupported added-token matching settings")
        tokenizer = cls(
            vocab=model["vocab"],
            ranks={(a, b): rank for rank, (a, b) in enumerate(pairs)},
            pattern=pattern, added=added,
            normalization="NFC" if normalizer else None,
            special_ids={item["id"] for item in data["added_tokens"]
                         if item["special"]},
        )
        reverse = {char: byte for byte, char in tokenizer.alphabet.items()}
        tokenizer.added_by_id = {value: key for key, value in added.items()}
        tokenizer.pieces = {
            token_id: bytes(reverse[char] for char in token)
            for token, token_id in tokenizer.vocab.items()
            if token_id not in tokenizer.added_by_id
        }
        return tokenizer

    def encode_piece(self, piece: str) -> list[int]:
        """Merge the lowest-rank adjacent pair until no merge applies."""
        if piece in self.cache:
            return self.cache[piece][:]
        symbols = [self.alphabet[byte] for byte in piece.encode("utf-8")]
        while len(symbols) > 1:
            candidates = [
                (self.ranks.get((left, right), float("inf")), index)
                for index, (left, right) in
                enumerate(zip(symbols, symbols[1:]))
            ]
            rank, index = min(candidates)
            if rank == float("inf"):
                break
            symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
        result = [self.vocab[symbol] for symbol in symbols]
        if len(self.cache) >= 8192:
            self.cache.clear()
        self.cache[piece] = result
        return result[:]

    def encode(self, text: str) -> list[int]:
        """Encode text, recognizing added tokens before pre-tokenization.

        Args:
            text: Unicode text, including any explicit added-token strings.

        Returns:
            Token IDs suitable for the SDK's public logits method.
        """
        expression = "|".join(regex.escape(token) for token in
                              sorted(self.added, key=len, reverse=True))
        parts = regex.split(f"({expression})", text) if expression else [text]
        result: list[int] = []
        for part in parts:
            if part in self.added:
                result.append(self.added[part])
                continue
            if self.normalization:
                part = unicodedata.normalize(self.normalization, part)
            pieces = self.pattern.findall(part)
            if "".join(pieces) != part:
                raise ValueError("pre-tokenization did not consume all text")
            for piece in pieces:
                result.extend(self.encode_piece(piece))
        return result

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode full byte sequences, optionally removing special tokens.

        Args:
            ids: Vocabulary token IDs in generation order.
            skip_special_tokens: Whether to omit designated special tokens.

        Returns:
            Decoded text, replacing incomplete or invalid UTF-8 sequences.
        """
        output = bytearray()
        for token_id in ids:
            if skip_special_tokens and token_id in self.special_ids:
                continue
            if token_id in self.added_by_id:
                output.extend(self.added_by_id[token_id].encode("utf-8"))
            else:
                output.extend(self.pieces[token_id])
        return output.decode("utf-8", errors="replace")
