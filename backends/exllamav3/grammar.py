import json
import typing
from collections import OrderedDict
from copy import copy, deepcopy
from typing import List, Optional
import traceback

from exllamav3 import (
    Tokenizer,
    Filter,
    FormatronFilter,
)
from formatron.extractor import NonterminalExtractor
from formatron.formatter import FormatterBuilder
from formatron.schemas import json_schema
from common.logger import xlogger


# Each cached prototype pins a kbnf engine (~150 MB with a 250k vocabulary) and the
# tokenizer it was compiled against. A handful of distinct grammars per model is the
# realistic ceiling, so this only guards against unbounded growth.
MAX_CACHED_FILTER_PROTOTYPES = 8


def clone_filter(prototype: FormatronFilter) -> FormatronFilter:
    """
    Produce a request-local copy of an already-built filter.

    Building a FormatronFilter compiles the grammar into token-level automata over the
    whole vocabulary, which costs seconds. Copying the resulting kbnf engine is a Rust-side
    clone measured in microseconds, so every request gets its own parse state without
    paying the compile again.
    """

    clone = copy(prototype)
    clone._formatter = deepcopy(prototype._formatter)
    clone._formatter.reset()
    clone._zeros = None
    clone.job = None
    clone.generator = None
    clone.vocab_size = None
    clone.is_active = prototype.trigger_token is None
    return clone


class FilterPrototypeCache:
    """
    LRU cache of built filters. Prototypes are never handed to a job - callers always
    receive a clone - so a prototype's parse state stays pristine and concurrent requests
    sharing a grammar cannot corrupt each other.
    """

    def __init__(self, max_entries: int = MAX_CACHED_FILTER_PROTOTYPES):
        self.max_entries = max_entries
        self._prototypes: "OrderedDict[tuple, FormatronFilter]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._prototypes)

    def get(self, key: tuple) -> Optional[FormatronFilter]:
        """Return a fresh clone of the cached prototype, or None on a miss."""

        prototype = self._prototypes.get(key)
        if prototype is None:
            return None

        self._prototypes.move_to_end(key)
        return clone_filter(prototype)

    def put(self, key: tuple, prototype: FormatronFilter) -> FormatronFilter:
        """Store a freshly built prototype and return the clone to use for this request."""

        self._prototypes[key] = prototype
        self._prototypes.move_to_end(key)
        while len(self._prototypes) > self.max_entries:
            self._prototypes.popitem(last=False)

        return clone_filter(prototype)

    def clear(self) -> None:
        """Drop every prototype. Must run on model unload: the automata and the strong
        tokenizer reference they hold are only valid for the vocabulary they were built
        against."""

        self._prototypes.clear()


# Shared across requests for the lifetime of the loaded model.
schema_filter_cache = FilterPrototypeCache()


class CFGExtractor(NonterminalExtractor):
    """Extractor class for KBNF context-free grammar"""

    def __init__(self, nonterminal: str, kbnf_string: str):
        super().__init__(nonterminal)
        self.kbnf_string = kbnf_string

    # Return the entire input string as the extracted string
    def extract(self, input_str: str) -> typing.Optional[tuple[str, typing.Any]]:
        return "", input_str

    @property
    def kbnf_definition(self) -> str:
        return self.kbnf_string.replace("start", self.nonterminal)


class ExLlamaV3Grammar:
    """ExLlamaV3 class for various grammar filters/parsers."""

    filters: List[Filter]

    def __init__(self):
        self.filters = []

    def _add_cached_filter(
        self,
        key: tuple,
        formatter_builder: FormatterBuilder,
        tokenizer: Tokenizer,
        trigger_token_id: Optional[int],
        eos_after_completed: bool,
    ):
        """Append a filter for the given grammar, building it only on a cache miss."""

        cached = schema_filter_cache.get(key)
        if cached is not None:
            self.filters.append(cached)
            return

        prototype = FormatronFilter(
            tokenizer,
            eos_after_completed=eos_after_completed,
            formatter_builder=formatter_builder,
            trigger_token=trigger_token_id,
        )
        self.filters.append(schema_filter_cache.put(key, prototype))

    def add_json_schema_filter(
        self,
        schema: dict,
        tokenizer: Tokenizer,
        trigger_token_id: int = None,
    ):
        """Adds an ExllamaV3 filter based on a JSON schema."""

        try:
            # Get named schema nested in from OAI response format config
            if "schema" in schema and "name" in schema:
                schema = schema["schema"]

            # Add fields required by formatron if not present
            if "$id" not in schema:
                schema["$id"] = "https://example.com/example.json"
            if "$schema" not in schema:
                schema["$schema"] = "http://json-schema.org/draft-07/schema#"

            leading_character = "[" if schema.get("type") == "array" else "{"

            # Keyed on the normalized schema, so property ordering and the OAI envelope
            # do not fragment the cache
            schema_key = json.dumps(schema, sort_keys=True)

            # Validate schema and create formatter
            schema = json_schema.create_schema(schema)

        except Exception:
            traceback.print_exc()
            xlogger.error(
                "Skipping because the JSON schema couldn't be parsed. "
                "Please read the above error for more information.",
                {"schema": schema, "exception": traceback.format_exc()},
            )
            return

        f = FormatterBuilder()
        f.append_line(f"{f.json(schema)}")
        self._add_cached_filter(
            ("json_schema", schema_key, trigger_token_id, id(tokenizer)),
            f,
            tokenizer,
            trigger_token_id,
            eos_after_completed=True,
        )

        # Additional constraint to force leading character
        f = FormatterBuilder()
        f.append_line(leading_character)
        self._add_cached_filter(
            ("leading_character", leading_character, trigger_token_id, id(tokenizer)),
            f,
            tokenizer,
            trigger_token_id,
            eos_after_completed=False,
        )

    def add_regex_filter(
        self,
        pattern: str,
        tokenizer: Tokenizer,
        trigger_token_id: Optional[int] = None,
    ):
        """Adds an ExllamaV3 filter based on a regular expression."""

        try:
            # Validate regex and create formatter
            f = FormatterBuilder()
            f.append_line(f"{f.regex(pattern)}")
        except Exception:
            traceback.print_exc()
            xlogger.error(
                "Skipping because the regex pattern couldn't be parsed. "
                "Please read the above error for more information.",
                {"pattern": pattern, "exception": traceback.format_exc()},
            )
            return

        self._add_cached_filter(
            ("regex", pattern, trigger_token_id, id(tokenizer)),
            f,
            tokenizer,
            trigger_token_id,
            eos_after_completed=True,
        )

    def add_kbnf_filter(
        self,
        kbnf_string: str,
        tokenizer: Tokenizer,
        trigger_token_id: Optional[int] = None,
    ):
        """Adds an ExllamaV3 filter based on KBNF grammar."""

        try:
            # Validate KBNF and create formatter
            f = FormatterBuilder()
            f.append_line(
                f"""{f.extractor(lambda nonterminal: CFGExtractor(nonterminal, kbnf_string))}"""
            )
        except Exception:
            traceback.print_exc()
            xlogger.error(
                "Skipping because the KBNF string couldn't be parsed. "
                "Please read the above error for more information.",
                {"kbnf_string": kbnf_string, "exception": traceback.format_exc()},
            )
            return

        self._add_cached_filter(
            ("kbnf", kbnf_string, trigger_token_id, id(tokenizer)),
            f,
            tokenizer,
            trigger_token_id,
            eos_after_completed=True,
        )
