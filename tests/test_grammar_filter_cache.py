import json
import unittest
from unittest.mock import patch

import kbnf

from backends.exllamav3.grammar import (
    ExLlamaV3Grammar,
    FilterPrototypeCache,
    schema_filter_cache,
)

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

PET_SCHEMA = {
    "type": "object",
    "properties": {"pet": {"type": "string"}},
    "required": ["pet"],
}

LIST_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
}

# A GPT2-style vocabulary: formatron autodetects the mangling from the Ġ density,
# and refuses to build a vocabulary when no processor matches.
VOCAB_PIECES = [
    "{", "}", "[", "]", ":", ",", '"', "name", "age", "pet",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "a", "b", "c", "Ġ", "Ċ", "true", "false", "null", "<eos>",
] + [f"Ġ{piece}" for piece in ("a", "b", "c", "name", "age", "pet", "0", "1", "true", "null")]


class FakeInnerTokenizer:
    def __init__(self, pieces):
        self._pieces = pieces

    def decode(self, tokens):
        return "".join(self._pieces[token] for token in tokens)


class FakeTokenizer:
    """Minimal stand-in for exllamav3.Tokenizer: vocab dict plus a decoder."""

    def __init__(self, pieces=VOCAB_PIECES):
        self._pieces = list(pieces)
        self.tokenizer = FakeInnerTokenizer(self._pieces)

    def get_vocab_dict(self):
        return {piece: index for index, piece in enumerate(self._pieces)}

    def piece_id(self, piece):
        return self._pieces.index(piece)

    def single_id(self, piece):
        return self.piece_id(piece)


def allowed_tokens(grammar_filter):
    grammar_filter._formatter.compute_allowed_tokens()
    return set(grammar_filter._formatter.get_allowed_tokens_since_last_computation())


class EngineBuildCounter:
    """Counts kbnf.Engine constructions - the multi-second step being cached."""

    def __init__(self):
        self.count = 0
        self._real = kbnf.Engine

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._real(*args, **kwargs)


class GrammarFilterCacheTests(unittest.TestCase):
    def setUp(self):
        schema_filter_cache.clear()
        self.tokenizer = FakeTokenizer()

    def tearDown(self):
        schema_filter_cache.clear()

    def add_schema(self, schema, tokenizer=None, trigger_token_id=None):
        handler = ExLlamaV3Grammar()
        handler.add_json_schema_filter(
            json.loads(json.dumps(schema)),
            tokenizer or self.tokenizer,
            trigger_token_id=trigger_token_id,
        )
        return handler

    def test_identical_schema_does_not_rebuild_the_engine(self):
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            first = self.add_schema(PERSON_SCHEMA)
            after_first = counter.count
            second = self.add_schema(PERSON_SCHEMA)

        # Two filters per request (schema + leading character), built once only.
        self.assertEqual(after_first, 2)
        self.assertEqual(counter.count, 2)
        self.assertEqual(len(first.filters), 2)
        self.assertEqual(len(second.filters), 2)

    def test_a_different_schema_builds_its_own_engine(self):
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            self.add_schema(PERSON_SCHEMA)
            self.add_schema(PET_SCHEMA)

        # Both requests build a schema engine; the leading "{" filter is shared.
        self.assertEqual(counter.count, 3)

    def test_cached_filters_do_not_share_parse_state(self):
        first = self.add_schema(PERSON_SCHEMA).filters[0]
        start_state = allowed_tokens(first)

        first._formatter.accept_token(self.tokenizer.piece_id("{"))
        self.assertNotEqual(allowed_tokens(first), start_state)

        second = self.add_schema(PERSON_SCHEMA).filters[0]
        self.assertEqual(allowed_tokens(second), start_state)

        third = self.add_schema(PERSON_SCHEMA).filters[0]
        self.assertEqual(allowed_tokens(third), start_state)
        self.assertIsNot(second, third)
        self.assertIsNot(second._formatter, third._formatter)

    def test_filters_are_not_shared_between_requests(self):
        first = self.add_schema(PERSON_SCHEMA).filters
        second = self.add_schema(PERSON_SCHEMA).filters

        for left, right in zip(first, second):
            self.assertIsNot(left, right)
            self.assertIsNot(left._formatter, right._formatter)

    def test_oai_envelope_shares_the_entry_with_the_bare_schema(self):
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            self.add_schema(PERSON_SCHEMA)
            self.add_schema(
                {"name": "person", "strict": True, "schema": PERSON_SCHEMA}
            )

        self.assertEqual(counter.count, 2)

    def test_key_ignores_property_ordering(self):
        reordered = {
            "required": ["name", "age"],
            "properties": {
                "age": {"type": "integer"},
                "name": {"type": "string"},
            },
            "type": "object",
        }
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            self.add_schema(PERSON_SCHEMA)
            self.add_schema(reordered)

        self.assertEqual(counter.count, 2)

    def test_wrapped_array_schema_forces_a_leading_bracket(self):
        handler = self.add_schema(
            {"name": "fruits", "strict": True, "schema": LIST_SCHEMA}
        )
        leading = handler.filters[1]

        self.assertEqual(
            allowed_tokens(leading), {self.tokenizer.piece_id("[")}
        )

    def test_object_schema_forces_a_leading_brace(self):
        leading = self.add_schema(PERSON_SCHEMA).filters[1]

        self.assertEqual(
            allowed_tokens(leading), {self.tokenizer.piece_id("{")}
        )

    def test_trigger_token_is_part_of_the_key(self):
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            self.add_schema(PERSON_SCHEMA, trigger_token_id=None)
            self.add_schema(PERSON_SCHEMA, trigger_token_id=3)

        self.assertEqual(counter.count, 4)

    def test_trigger_token_survives_a_cache_hit(self):
        trigger = self.tokenizer.piece_id("{")
        self.add_schema(PERSON_SCHEMA, trigger_token_id=trigger)
        cached = self.add_schema(PERSON_SCHEMA, trigger_token_id=trigger).filters[0]

        self.assertEqual(cached.trigger_token, trigger)
        self.assertFalse(cached.is_active)

    def test_a_different_tokenizer_does_not_reuse_the_entry(self):
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            self.add_schema(PERSON_SCHEMA)
            self.add_schema(PERSON_SCHEMA, tokenizer=FakeTokenizer())

        self.assertEqual(counter.count, 4)

    def test_clear_drops_every_entry(self):
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            self.add_schema(PERSON_SCHEMA)
            schema_filter_cache.clear()
            self.add_schema(PERSON_SCHEMA)

        self.assertEqual(counter.count, 4)

    def test_regex_filters_are_cached(self):
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            first = ExLlamaV3Grammar()
            first.add_regex_filter("[abc]+", self.tokenizer)
            second = ExLlamaV3Grammar()
            second.add_regex_filter("[abc]+", self.tokenizer)

        self.assertEqual(counter.count, 1)
        self.assertEqual(len(second.filters), 1)
        self.assertIsNot(first.filters[0], second.filters[0])

    def test_kbnf_filters_are_cached(self):
        grammar = 'start ::= "{" "}";'
        counter = EngineBuildCounter()
        with patch.object(kbnf, "Engine", counter):
            first = ExLlamaV3Grammar()
            first.add_kbnf_filter(grammar, self.tokenizer)
            second = ExLlamaV3Grammar()
            second.add_kbnf_filter(grammar, self.tokenizer)

        self.assertEqual(counter.count, 1)
        self.assertIsNot(first.filters[0], second.filters[0])

    def test_an_unparsable_schema_adds_no_filter_and_caches_nothing(self):
        handler = ExLlamaV3Grammar()
        handler.add_json_schema_filter({"type": "not-a-real-type"}, self.tokenizer)

        self.assertEqual(handler.filters, [])
        self.assertEqual(len(schema_filter_cache), 0)

    def test_an_unparsable_regex_adds_no_filter(self):
        handler = ExLlamaV3Grammar()
        handler.add_regex_filter("[unterminated", self.tokenizer)

        self.assertEqual(handler.filters, [])


class FilterPrototypeCacheEvictionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.cache = FilterPrototypeCache(max_entries=2)

    def build(self, key):
        handler = ExLlamaV3Grammar()
        handler.add_kbnf_filter(f'start ::= "{key}";', self.tokenizer)
        return handler.filters[0]

    def test_least_recently_used_entry_is_evicted(self):
        self.cache.put("a", self.build("a"))
        self.cache.put("b", self.build("b"))
        self.cache.put("c", self.build("c"))

        self.assertIsNone(self.cache.get("a"))
        self.assertIsNotNone(self.cache.get("b"))
        self.assertIsNotNone(self.cache.get("c"))
        self.assertEqual(len(self.cache), 2)

    def test_a_hit_refreshes_recency(self):
        self.cache.put("a", self.build("a"))
        self.cache.put("b", self.build("b"))
        self.cache.get("a")
        self.cache.put("c", self.build("c"))

        self.assertIsNotNone(self.cache.get("a"))
        self.assertIsNone(self.cache.get("b"))

    def test_put_returns_a_clone_and_keeps_the_prototype_pristine(self):
        prototype = self.build("a")
        start_state = allowed_tokens(prototype)

        handed_out = self.cache.put("a", prototype)
        self.assertIsNot(handed_out, prototype)
        handed_out._formatter.accept_token(self.tokenizer.piece_id("a"))

        self.assertEqual(allowed_tokens(self.cache.get("a")), start_state)


if __name__ == "__main__":
    unittest.main()
