*This project has been created as part of the 42 curriculum by mtellami.*

# Lexicall

## Description

Lexicall translates natural-language requests into typed function calls
with a small local language model. It selects a function and extracts its
arguments; it never executes the function or computes the requested answer.

For “What is the sum of 2 and 3?”, the result is:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2, "b": 3}
}
```

The implementation includes schema-constrained generation, a handwritten
byte-level BPE tokenizer, nested arguments, bounded error recovery, caching,
two model profiles, and an optional terminal generation trace.

## Instructions

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use
Python 3.10 or later (the repository selects Python 3.12 by default). From the
repository root, run the evaluator's commands:

```bash
uv sync
uv run python -m src
```

The equivalent Makefile workflow uses the committed lockfile:

```bash
make install
make run
```

`uv sync` creates `.venv` and installs the application, development tools, and
the bundled SDK from `llm_sdk/`. `uv.lock` records the resolved dependencies;
`make install` uses `uv sync --locked` to verify that it is current. To select
another supported interpreter when creating an environment, use
`uv sync --python python3.12`. To select another environment directory, set
`UV_PROJECT_ENVIRONMENT=/path/to/venv` for uv or Makefile commands.
The SDK declares and uses PyTorch, Transformers, and Accelerate for inference;
application code does not import those libraries or inspect SDK private attributes.

The first inference run downloads model and tokenizer files from Hugging Face.
Allow several gigabytes of disk space for dependencies and model caches.
Subsequent runs reuse the downloaded files. CPU inference works without a GPU;
the SDK also selects CUDA or MPS automatically when available. CPU thread count
defaults to four and can be overridden with `OMP_NUM_THREADS`.

### Example usage

```bash
uv run python -m src --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

make run ARGS="--model Qwen/Qwen2.5-0.5B-Instruct"
make run ARGS="--device cpu --visualize"
make run ARGS="--max-new-tokens 256"
make debug ARGS="--device cpu"
make lint
make lint-strict
make format
make clean
```

`install` sets up the environment and installs dependencies. `run` starts the
CLI, and `debug` starts it in Python's built-in debugger (`pdb`); use `n` to
step, `c` to continue, and `q` to quit. `lint` runs flake8 and mypy with the
assignment's required flags; `lint-strict` runs flake8 and strict mypy.
`clean` removes Python and mypy caches while keeping the installed environment.
`format` applies autopep8 to `src/` to fix style issues. Run `lint` afterward
to check for any remaining issues that require manual changes.

The default files are exactly those in the first command. Output directories
are created automatically and excluded from Git.
Diagnostics and traces go to stderr; the output file contains only JSON.

Exit codes: `0` for a complete published batch, `1` for a handled failure,
`2` for invalid command-line arguments, and `130` for interruption.

## Input and output contracts

The prompts file is an array of objects containing exactly a string `prompt`.
The definitions file is a nonempty array of unique function definitions:

```json
[
  {
    "name": "create_record",
    "description": "Create a named record with an enabled flag.",
    "parameters": {
      "record": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "enabled": {"type": "boolean"}
        }
      }
    },
    "returns": {"type": "boolean"}
  }
]
```

Supported value types are `string`, `number`, `integer`, `boolean`, `null`,
`object`, and `array`. Schemas can include `description` and nonempty `enum`.
Objects must declare `properties`; arrays must declare `items`. All top-level
parameters are required. Nested `required` lists can make individual properties
optional; omitting the list makes all declared properties required. Extra
properties are forbidden. `additionalProperties: false` is accepted explicitly.
Unsupported JSON Schema keywords are rejected rather than silently ignored.

Each output record has exactly `prompt`, `name`, and `parameters`, in input
order. Duplicate requests retain duplicate output records. An empty prompt
array produces `[]` without loading a model. Missing, ambiguous, or unrelated
requests still use the supplied function catalog: this format has no no-call
sentinel and cannot guarantee semantic correctness for such requests.

## Algorithm explanation

### Handwritten tokenization

`BPETokenizer.from_file` reads `tokenizer.json` using the SDK's public
`get_path_to_tokenizer_file()` method. It checks supported tokenizer settings,
loads vocabulary and merge ranks, and separates added tokens from ordinary
token bytes. `encode` recognizes added tokens, applies the declared NFC
normalization when present, then applies the declared Unicode
pre-tokenization regex, maps UTF-8 bytes into the reversible byte alphabet, and
repeatedly merges the lowest-ranked adjacent BPE pair. `decode` reverses that
mapping and assembles bytes before decoding Unicode. Special-token removal
matches the SDK's public decode behavior.

The production pipeline calls neither SDK `encode` nor SDK `decode`. Its public
`encode(text) -> list[int]` and `decode(ids) -> str` methods feed IDs directly to
`get_logits_from_input_ids`. Model-specific chat profiles supply the assistant
prefix; Qwen3 uses a closed, empty thinking block for non-thinking output.

### Schema compilation and constrained decoding

The catalog compiles into one byte-oriented regular expression with a branch
for each function. A branch fixes the function name and its exact parameter
schema. Generated syntax is minified. Objects emit properties in definition
order, avoiding permutations;
optional nested properties can be omitted. Arrays accept repeated item schemas.
An incremental partial match distinguishes legal prefixes from complete calls.
This is a grammar implementation using `regex`, not a third-party constrained
generation framework.

The fixed `{"name":"` prefix is prefilled. The function-name content is
generated through model logits, preserving its generated tokenization.
After name selection, syntax with exactly one legal byte continuation is
inserted directly. No inference is needed to choose a fixed key or delimiter.
The extraction context then contains only the selected function, reducing
repeated inference work while preserving the model's original catalog choice.
The full context is retokenized after such insertions, preserving BPE boundaries.
Argument choices remain constrained by the same grammar.

At each remaining step, the SDK returns next-token logits. Candidates are checked in
descending score order. Invalid candidates receive negative infinity. Search
stops after the two highest-scoring legal candidates: the first is the exact
greedy constrained choice and the second is retained for recovery. Checking
lower-scoring tokens cannot change either selection. This lazy masking avoids
materializing the entire vocabulary mask on every inference step.

Checks apply to the entire token byte sequence. A token can contain multiple
JSON delimiters or only part of a Unicode character. String grammar enforces
JSON escapes, UTF-8 validity, and paired surrogate escapes. Numeric grammar
enforces JSON number syntax. A candidate completing a call also undergoes
strict semantic validation, rejecting overflow to infinity and incorrect types.

Generation stops at a complete document. The original prompt is copied into
the result and Pydantic models plus recursive validation check the final call.
The standard JSON serializer publishes the result; no prose extraction,
JSON repair, type coercion, or hardcoded function-selection rules are used.

## Design decisions and recovery

Application classes use Pydantic models. Input validation happens before model
loading. JSON parsing rejects duplicate keys and nonstandard numeric constants.
Final validation distinguishes booleans from integers and never converts a
numeric string into a number.

The decoder permits three attempts: initial greedy generation, backtracking to
the latest retained alternative, and a fresh generation with a schema reminder.
Each attempt has a token budget. Exhausted recovery, loading errors, malformed
inputs, filesystem errors, and interruptions produce clear messages without a
traceback at the CLI boundary. No placeholder function calls are invented.

A failed batch never replaces an existing output file. A successful batch is
written to a temporary file beside the destination, flushed, and atomically
renamed. Input files cannot be used as the output path.

Caches include BPE pieces, grammar prefix checks, and completed duplicate
requests within a decoder instance. Each decoder fixes its model, catalog,
and generation settings; separate runs share no result cache. A token-byte trie
shares prefix checks when counting legal tokens for visualization. The supplied
SDK recomputes the full context for each generated token and exposes no public
batch or KV-cache API. Optimization therefore stays within its public surface.

With `--visualize`, stderr shows the attempt, step, selected token ID and bytes,
prefix byte count, exact grammar-allowed token count, and elapsed time. Counting
the complete legal set is additional work, so tracing is disabled for benchmarks.

## Testing strategy

Development validation covered grammar prefixes, schema-dependent function
names, nested arguments, optional fields, escapes, Unicode, invalid types,
invalid model scores, illegal highest-logit tokens, recovery limits, caching,
atomic writes, malformed input, and CLI behavior. Tokenizer parity checks use
only public SDK methods and included deterministic randomized Unicode strings.
A temporary suite of 61 checks passed. Both supported models passed 113 public-SDK
tokenizer parity cases, including decomposed Unicode and added tokens.

The development test suite and benchmark helpers have been removed. The
validation results below record checks performed during implementation.

For manual lint and type checks, run:

```bash
uv run flake8 .
uv run mypy . --warn-return-any --warn-unused-ignores \
  --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
uv run mypy . --strict
```

The supplied SDK is treated as vendor code and excluded from application
lint/type-check coverage.

## Performance analysis

Measured on an Intel Core i7-8665U CPU with four inference threads, CPU float32,
and cached model files. The default CLI run includes model initialization and
output publication.

| Model and evaluation | Correct complete calls | Schema-valid | Time |
| --- | --- | --- | --- |
| Qwen3-0.6B, 11 supplied prompts, default CLI | 11/11 (100%) | 11/11 | 181.97 s |
| Qwen3-0.6B, 20 held-out prompts and eight unfamiliar functions | 19/20 (95%) | 20/20 | 250.46 s |
| Qwen2.5-0.5B-Instruct, 11 supplied prompts | 11/11 (100%) | 11/11 | 141.09 s |

Structural validity and semantic accuracy are measured separately: constrained
decoding guarantees schema-valid published calls, while choosing the right
function and extracting the right values depend on the model. Hardware and
download state affect runtime. The five-minute target is measured with cached
model files, including initialization; first-download time is separate.
The development benchmark also verified repeated-request caching: replaying a prompt
used zero additional logits calls (about 0.1 ms on the measured run).

Known extraction limitation: Qwen3 can interpret the unquoted phrase “empty
string” as literal text instead of `""`. Schema validation catches structural
and type errors, but cannot detect this semantic mistake. Function selection
was correct for all 20 held-out requests; this argument error was the sole miss.

## Challenges faced

- Vocabulary entries represent bytes, not necessarily standalone Unicode
  strings. Grammar checks operate on bytes and accept incomplete UTF-8 prefixes.
- Tokens can cross JSON syntax boundaries. Full-token checks prevent an
  apparently safe first character from hiding an invalid suffix.
- Small models need both schema guidance and precise extraction instructions.
  Function definitions are supplied dynamically, and generated names activate
  their corresponding argument grammar.
- Function-name tokenization affects routing. Name tokens retain the model's
  generated sequence; forced syntax and context narrowing start after the
  complete name has been selected. Prefilling the opening quote prevents
  combined quote-and-name tokens from favoring a shared name prefix.
- SDK inference has no reusable attention-cache API. Lazy masking and compact
  prompts reduce application overhead without accessing private internals.

## Resources

### AI assistance

AI assistance was used to inspect and explain the implementation, describe
tokenization and constrained generation, draft `ARCHITECTURE.md` and its Mermaid
diagrams, and audit the repository against the assignment PDF. It also assisted
with restoring the required directory layout, uv dependency setup, Makefile
commands, and documentation, and with running validation checks. These statements
describe the assistance recorded in the current development conversation; they
do not establish how earlier versions of the implementation were authored.
The author remains responsible for understanding and verifying the submitted code.

### References

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
  and [tokenizer configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json).
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- [Original GPT-2 byte-level BPE implementation](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259).
- [Pydantic documentation](https://docs.pydantic.dev/latest/).
- [regex partial matching documentation](https://github.com/mrabarnett/mrab-regex#partial-matches).
