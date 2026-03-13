"""
Utility classes and functions for use with MLLMs for parsing addresses.
"""
import json
import re
import difflib
from abc import ABC, abstractmethod
from utils import ParsedAddressResultBuilder
import transformers
import sentence_transformers
import pandas as pd
from collections import OrderedDict
from typing import Any

_STREET_SUFFIX_RE = re.compile(
    r'^(straße|strasse|gasse|weg|allee|platz|ring|damm|chaussee|steig|stieg|'
    r'pfad|trift|promenade|avenue|av\.)$',
    re.IGNORECASE,
)
_STR_ABBREV_RE = re.compile(r'^str\.$', re.IGNORECASE)
_DIST_TOKENS   = {'kr.', 'krs.', 'kreis', 'landkreis'}
_REGBEZ_TOKENS = {'reg.', 'bez.', 'regierungsbezirk'}
_PREP_TOKENS   = {'i.', 'b.', 'bei', 'a.', 'an', 'am', 'v.', 'im', 'in',
                  'o.', 'ob', 'n.', 'nr.', 'nähe'}

# unicode-letter word (optional trailing ".") | digits (optional trailing letter) |
# punctuation char | whitespace run
_ADDR_TOKEN_RE = re.compile(r'[^\W\d_]+\.?|\d+[a-zA-Z]?|[^\w\s]|\s+', re.UNICODE)

# word tokens for Jaccard similarity (lowercase, no punctuation)
_WORD_TOKEN_RE = re.compile(r'[^\W_]+', re.UNICODE)


def _token_set(address: str) -> frozenset:
    return frozenset(_WORD_TOKEN_RE.findall(address.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _classify_word(tok: str) -> str:
    tl = tok.lower()
    if tl in _DIST_TOKENS:
        return 'DIST'
    if tl in _REGBEZ_TOKENS:
        return 'REGBEZ'
    if tl in _PREP_TOKENS:
        return 'PREP'
    if _STREET_SUFFIX_RE.match(tok) or _STR_ABBREV_RE.match(tok):
        return 'STREET'
    return 'WORD'


def address_to_pattern(address: str) -> str:
    """Convert an address string to an abstract structural pattern.

    Examples:
        "Karl-Marx-Straße 21/3, Berlin"     → "WORD-WORD-STREET NUM/NUM, WORD"
        "Krs. Breslau, Hauptstr. 4"         → "DIST WORD, WORD-STREET NUM"
        "Berlin-Marienfelde, Alt Marienfelde 4" → "WORD-WORD, WORD WORD NUM"
    """
    parts = []
    for tok in _ADDR_TOKEN_RE.findall(address):
        if tok.isspace():
            parts.append(' ')
        elif tok[0].isdigit():
            parts.append('NUM')
        elif tok[0].isalpha() or ord(tok[0]) > 127:   # letter (incl. unicode)
            parts.append(_classify_word(tok))
        else:
            parts.append(tok)                          # punctuation verbatim
    return re.sub(r' {2,}', ' ', ''.join(parts)).strip()

class ExampleMatchingStrategy(ABC):
    def bulk_find_examples(self, addresses: list[str]) -> list[tuple[list[tuple[str, dict]], Any | None]]:
        return [self.find_examples(addr) for addr in addresses]

    @abstractmethod
    def find_examples(self, address: str) -> tuple[list[tuple[str, dict]], Any | None]:
        raise Exception("Not implemented")
    
class FixedExamples(ExampleMatchingStrategy):
    def __init__(self, examples: list[tuple[str, dict]]):
        self.examples = examples

    def find_examples(self, address: str) -> tuple[list[tuple[str, dict]], Any | None]:
        return self.examples, None

class ZeroShot(ExampleMatchingStrategy):
    def find_examples(self, address: str) -> tuple[list[tuple[str, dict]], Any | None]:
        return [], None

class SimilarExamples(ExampleMatchingStrategy):
    def __init__(self,
                 example_addresses: pd.Series,
                 example_labels: pd.DataFrame,
                 num_examples : int,
                 labels_to_include: list[str],
                 embeeding_model="multi-qa-mpnet-base-dot-v1",
                 similarity_threshold: float = None,
                 try_match_order : bool = True,
                 device=None):
        self.example_addresses = example_addresses
        self.example_labels = example_labels[labels_to_include]
        assert len(self.example_addresses) == len(self.example_labels), "example_addresses and example_labels must have the same length"
        if len(self.example_labels) < num_examples:
            print(f"Warning: num_examples {num_examples} is greater than "
                  f"the number of available examples {len(self.example_labels)}. "
                  f"Reducing num_examples to {len(self.example_labels)}.")
            self.num_examples = len(self.example_labels)
        else:
            self.num_examples = num_examples
        self.device = device
        self.model = sentence_transformers.SentenceTransformer(embeeding_model, device=device)
        self.similarity_threshold = similarity_threshold
        self.try_match_order = try_match_order
        self.example_embeddings = self.model.encode(self.example_addresses, convert_to_tensor=True)
        self.example_embeddings = self.example_embeddings.to(device)
        self.example_embeddings = sentence_transformers.util.normalize_embeddings(self.example_embeddings)

    def _get_example(self, index):
        address = self.example_addresses.iloc[index]
        labels = []
        for label, part in self.example_labels.iloc[index].items():
            if not pd.isna(part):
                address_start = address.find(part)
                labels.append((address_start, label, part))
        if self.try_match_order:
            labels.sort()
        labels = OrderedDict((x[1], x[2]) for x in labels)
        return address, labels

    def _hit_filter(self, hit):
        if self.similarity_threshold is not None and hit["score"] < self.similarity_threshold:
            return False
        return True

    def bulk_find_examples(self, addresses):
        address_embeddings = self.model.encode(addresses, convert_to_tensor=True)
        address_embeddings = address_embeddings.to(self.device)
        address_embeddings = sentence_transformers.util.normalize_embeddings(address_embeddings)
        bulk_hits = sentence_transformers.util.semantic_search(
            address_embeddings, self.example_embeddings, top_k=self.num_examples)
        bulk_examples = []
        for example_hits in bulk_hits:
            examples = []
            metadatas = []
            for hit in example_hits:
                example = self._get_example(hit["corpus_id"])
                metadata = hit | {"address": example[0], "example": example[1], "included": False}
                if self._hit_filter(hit):
                    examples.append(self._get_example(hit["corpus_id"]))
                    metadata["included"] = True
                metadatas.append(metadata)
            bulk_examples.append((examples, metadatas))
        return bulk_examples


    def find_examples(self, address: str):
        return self.bulk_find_examples([address])[0]


class PatternTokenSimilarExamples(ExampleMatchingStrategy):
    """Few-shot selection combining structural pattern similarity and lexical
    token similarity:

        score = pattern_weight * pattern_sim + (1 - pattern_weight) * token_sim

    *pattern_sim* – ``difflib.SequenceMatcher`` ratio between the abstract
                    address patterns produced by ``address_to_pattern()``,
                    e.g. "WORD-WORD-STREET NUM/NUM, WORD".
    *token_sim*   – Jaccard similarity on the lowercased word-token sets of
                    the raw address strings.

    Both scores are computed against the full training set (no GPU needed).
    """

    def __init__(
        self,
        example_addresses: pd.Series,
        example_labels: pd.DataFrame,
        num_examples: int,
        labels_to_include: list[str],
        pattern_weight: float = 0.6,
        similarity_threshold: float = None,
        try_match_order: bool = True,
    ):
        self.example_addresses    = example_addresses.reset_index(drop=True)
        self.example_labels       = example_labels[labels_to_include].reset_index(drop=True)
        assert len(self.example_addresses) == len(self.example_labels)
        self.num_examples         = min(num_examples, len(self.example_labels))
        self.pattern_weight       = pattern_weight
        self.similarity_threshold = similarity_threshold
        self.try_match_order      = try_match_order

        # Precompute patterns and token sets for all training examples
        self.example_patterns   = [address_to_pattern(a) for a in self.example_addresses]
        self.example_token_sets = [_token_set(a)         for a in self.example_addresses]

    def _get_example(self, index: int):
        address = self.example_addresses.iloc[index]
        labels = []
        for label, part in self.example_labels.iloc[index].items():
            if not pd.isna(part):
                labels.append((address.find(part), label, part))
        if self.try_match_order:
            labels.sort()
        return address, OrderedDict((x[1], x[2]) for x in labels)

    def bulk_find_examples(self, addresses: list[str]):
        results = []
        for addr in addresses:
            query_pattern  = address_to_pattern(addr)
            query_tokens   = _token_set(addr)

            scored = []
            for idx, (ex_pattern, ex_tokens) in enumerate(
                zip(self.example_patterns, self.example_token_sets)
            ):
                pat_score = difflib.SequenceMatcher(None, query_pattern, ex_pattern).ratio()
                tok_score = _jaccard(query_tokens, ex_tokens)
                combined  = self.pattern_weight * pat_score + (1 - self.pattern_weight) * tok_score
                scored.append((combined, tok_score, pat_score, idx))

            scored.sort(reverse=True)

            examples  = []
            metadatas = []
            for combined, tok_score, pat_score, idx in scored[:self.num_examples]:
                example  = self._get_example(idx)
                included = self.similarity_threshold is None or combined >= self.similarity_threshold
                metadatas.append({
                    "corpus_id":       idx,
                    "score":           combined,
                    "token_score":     tok_score,
                    "pattern_score":   pat_score,
                    "address":         example[0],
                    "example":         example[1],
                    "included":        included,
                    "query_pattern":   query_pattern,
                    "example_pattern": self.example_patterns[idx],
                })
                if included:
                    examples.append(example)

            results.append((examples, metadatas))

        return results

    def find_examples(self, address: str):
        return self.bulk_find_examples([address])[0]


class PromptTemplate(ABC):
    def __init__(self, template: str, 
                 examples_prefix = "Consider the following examples:\n", 
                 example_template = "Address: %(address)s\n%(example)s\n"):
        self.template = template
        self.examples_prefix = examples_prefix
        self.example_template = example_template

    def make_prompt(self, address: str, examples : list[tuple[str, dict]]) -> dict:
        formatted_examples = [
            self.example_template % {"address": addr, "example": self.format_example(example)} 
            for addr, example in examples
        ]
        if len(formatted_examples) > 0:
            examples = self.examples_prefix + "".join(formatted_examples)
        else:
            examples = ""
        template_parameters = {"address": address, "examples": examples}
        return self.template % template_parameters

    @abstractmethod
    def format_example(self, example) -> str:
        raise Exception("Not implemented")

    @abstractmethod
    def parse_output(self, response: str, original_address: str) -> dict:
        raise Exception("Not implemented")

def extract_json_block(model_response : str):
    limit_chars = [('{', '}'), ('[', ']'), ('"', '"')]
    json_str = model_response
    parts = model_response.split("```")
    parts = [subpart for part in parts for subpart in part.split("`")]
    if len(parts) >= 2: # single code block or malformed code block
        json_str = parts[1]
    elif len(parts) >= 3:
        for part in parts:
            part = part.strip()
            if part.startswith("json") or any(part.startswith(lim[0]) and part.endswith(lim[1]) for lim in limit_chars):
                json_str = part
                break
    if json_str.startswith("json"):
        json_str = json_str[4:].strip()
    return json_str

class JsonDictPromptTemplate(PromptTemplate):
    def format_example(self, example):
        return "```json" + json.dumps(example, ensure_ascii=False) + "```"
    
    def parse_output(self, response, original_address):
        json_str = extract_json_block(response)
        obj = json.loads(json_str)
        result_builder = ParsedAddressResultBuilder(original_address)
        for k, v in obj.items():
            result_builder.add_part(k, v)
        data = result_builder.build()
        return data
    
class JSONTuplesPromptTemplate(PromptTemplate):
    def __init__(self, template: str, 
                 examples_prefix = "Consider the following examples:\n", 
                 example_template = "Address: %(address)s\n%(example)s\n",
                 ignore_other = True):
        super().__init__(template, examples_prefix, example_template)
        self.ignore_other = ignore_other

    def format_example(self, example):
        return "```json" + json.dumps([[v, k] for k, v in example.items()], ensure_ascii=False) + "```"
    
    def parse_output(self, response, original_address):
        ignore_key = None
        if self.ignore_other:
            ignore_key = "Other"
            if isinstance(self.ignore_other, str):
                ignore_key = self.ignore_other
        json_str = extract_json_block(response)
        tuples = json.loads(json_str)
        result_builder = ParsedAddressResultBuilder(original_address)
        for part, ptype in tuples:
            if ptype != ignore_key:
                result_builder.add_part(ptype, part)
        data = result_builder.build()
        return data


class LlamaAddressParsingModel:
    def __init__(self, 
                 model_name, 
                 prompt : PromptTemplate, 
                 example_strategy : ExampleMatchingStrategy | dict, 
                 batch_size=32, 
                 device=None):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, padding_side='left', device=device)
        self.pipe = transformers.pipeline("text-generation", model=model_name, 
                                          batch_size=batch_size, tokenizer=tokenizer)
        if getattr(self.pipe.tokenizer, "pad_token_id", None) is None:
            eos_token_id = self.pipe.model.config.eos_token_id
            if not isinstance(eos_token_id, int): eos_token_id = eos_token_id[0]
            self.pipe.tokenizer.pad_token_id = eos_token_id
        self.example_strategy : ExampleMatchingStrategy
        if isinstance(example_strategy, dict):
            self.example_strategy = example_strategy["factory"](
                *example_strategy.get("factory_args", []),
                **example_strategy.get("factory_kargs", {})
            )
        else:
            self.example_strategy = example_strategy
        self.prompt = prompt

    def _parse_output(self, conversation, original_address: str, example_metadata=None):
        model_response = None
        output_dict = None
        try:
            model_response = conversation[0]["generated_text"][1]["content"]
            parsed = self.prompt.parse_output(model_response, original_address=original_address)
            parsed["fullConversation"] = json.dumps(conversation, ensure_ascii=False)
            output_dict = parsed
        except Exception as e:
            print(f"Error parsing model output for address '{original_address}': {e}\n"
                  f"Full Conversation: {conversation}\n"
                  f"Model response: {repr(model_response)}")
            output_dict = {"error": str(e), "fullConversation": json.dumps(conversation)}
        if example_metadata is not None:
            output_dict["___example_metadata"] = example_metadata
        return output_dict

    def parse_addresses(self, addresses : list[str]) -> str:
        bulk_examples = self.example_strategy.bulk_find_examples(addresses)
        messages = [[
            {
                "role": "user", 
                "content": self.prompt.make_prompt(address, address_examples)
            }
        ] for address, (address_examples, _) in zip(addresses, bulk_examples)]
        bulk_examples_metadata = [metadata for _, metadata in bulk_examples]
        result = self.pipe(messages)
        responses = [
            self._parse_output(r, original_address=addr, example_metadata=example_metadata) 
            for r, addr, example_metadata in zip(result, addresses, bulk_examples_metadata)]
        return responses