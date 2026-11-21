"""Lexicall converts natural language into typed function calls."""

import argparse
import os
import sys
import time
from pathlib import Path

from src.decoder import MODELS, ConstrainedDecoder
from src.files import load_inputs, write_results
from src.grammar import Grammar
from src.models import FunctionCall
from src.tokenizer import BPETokenizer


def positive_integer(value: str) -> int:
    """Parse a positive generation budget."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    """Build the required command line and optional bonus controls."""
    cli = argparse.ArgumentParser(prog="Lexicall", description=__doc__)
    cli.add_argument("--functions_definition", type=Path,
                     default=Path(
                         "assets/data/input/functions_definition.json"))
    cli.add_argument("--input", type=Path,
                     default=Path(
                         "assets/data/input/function_calling_tests.json"))
    cli.add_argument("--output", type=Path,
                     default=Path(
                         "assets/data/output/function_calling_results.json"))
    cli.add_argument("--model", choices=MODELS, default=MODELS[0])
    cli.add_argument("--device", choices=("cpu", "cuda", "mps"))
    cli.add_argument("--visualize", action="store_true")
    cli.add_argument("--max-new-tokens", type=positive_integer, default=512)
    return cli


def main(argv: list[str] | None = None) -> int:
    """Run a complete batch and return a conventional process exit code."""
    options = parser().parse_args(argv)
    started = time.perf_counter()
    try:
        definitions, prompts = load_inputs(options.functions_definition,
                                           options.input)
        if options.output.resolve() in {
            options.input.resolve(), options.functions_definition.resolve()
        }:
            raise ValueError("output must not overwrite either input file")
        grammar = Grammar.compile(definitions)
        calls: list[FunctionCall] = []
        if prompts:
            # CPU oversubscription is especially costly for small models.
            os.environ.setdefault("OMP_NUM_THREADS", "4")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from llm_sdk import Small_LLM_Model

            print(f"Loading {options.model}...", file=sys.stderr)
            sdk = Small_LLM_Model(model_name=options.model,
                                  device=options.device,
                                  trust_remote_code=False)
            tokenizer = BPETokenizer.from_file(
                sdk.get_path_to_tokenizer_file()
            )
            decoder = ConstrainedDecoder(
                sdk=sdk, tokenizer=tokenizer,
                grammar=grammar, definitions=definitions,
                model=options.model, max_new_tokens=options.max_new_tokens,
                visualize=options.visualize,
            )
            for index, request in enumerate(prompts, start=1):
                try:
                    calls.append(decoder.generate(request.prompt))
                except Exception as error:
                    raise ValueError(f"prompt {index}: {error}") from error
                print(f"Processed {index}/{len(prompts)}", file=sys.stderr)
        write_results(options.output, calls)
        elapsed = time.perf_counter() - started
        print(f"Wrote {len(calls)} calls to {options.output} "
              f"in {elapsed:.2f}s", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("error: interrupted; output was not published", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
