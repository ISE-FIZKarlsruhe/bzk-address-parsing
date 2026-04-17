"""
Utility classes and functions for use with MLLMs for parsing addresses.
"""
import json
import re
from abc import ABC, abstractmethod
from utils import ParsedAddressResultBuilder
import transformers
import sentence_transformers
import pandas as pd
from collections import OrderedDict
from typing import Any
import re

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


_ADDR_TOKEN_RE = re.compile(r'[^\W\d_]+\.?|\d+[a-zA-Z]?|[^\w\s]|\s+', re.UNICODE)


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
    def __init__(self, examples: list[tuple[str, dict]], labels_to_include: list[str] | None = None):
        self.examples = examples
        self.labels_to_include = labels_to_include

    def find_examples(self, address: str) -> tuple[list[tuple[str, dict]], Any | None]:
        examples = self.examples
        if self.labels_to_include is not None:
            examples = [
                (addr, {k: v for k, v in example.items() if k in self.labels_to_include})
                for addr, example in examples
            ]
        return examples, None

class ZeroShot(ExampleMatchingStrategy):
    def find_examples(self, address: str) -> tuple[list[tuple[str, dict]], Any | None]:
        return [], None

class SimilarExamples(ExampleMatchingStrategy):
    def __init__(self,
                 example_addresses: pd.Series,
                 example_labels: pd.DataFrame,
                 num_examples : int,
                 labels_to_include: list[str],
                 embedding_model="multi-qa-mpnet-base-dot-v1",
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
        if isinstance(embedding_model, str):
            self.model = sentence_transformers.SentenceTransformer(embedding_model, device=device)
        else:
            self.model = embedding_model
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


class HybridSimilarExamples(ExampleMatchingStrategy):
    """Few-shot selection that draws candidates from two pools (embedding + pattern),
    then keeps the top-num_examples by score across both pools.

    For each address:
      1. Fetch pool_size candidates from the embedding strategy.
      2. Fetch pool_size candidates from the pattern strategy.
      3. Merge both pools, deduplicate by corpus_id (first occurrence wins),
         sort by score descending, return the top num_examples.
    """

    def __init__(
        self,
        embedding_strategy: "SimilarExamples",
        pattern_strategy: "NERPatternSimilarExamples",
        num_examples: int = 3,
        pool_size: int = 3,
    ):
        self.embedding_strategy = embedding_strategy
        self.pattern_strategy   = pattern_strategy
        self.num_examples       = num_examples
        self.pool_size          = pool_size

    def bulk_find_examples(self, addresses: list[str]):
        orig_embedding_n = self.embedding_strategy.num_examples
        orig_pattern_n   = self.pattern_strategy.num_examples
        self.embedding_strategy.num_examples = self.pool_size
        self.pattern_strategy.num_examples   = self.pool_size

        embedding_bulk = self.embedding_strategy.bulk_find_examples(addresses)
        pattern_bulk   = self.pattern_strategy.bulk_find_examples(addresses)

        self.embedding_strategy.num_examples = orig_embedding_n
        self.pattern_strategy.num_examples   = orig_pattern_n

        results = []
        for (em_examples, em_metas), (pt_examples, pt_metas) in zip(embedding_bulk, pattern_bulk):
            for m in em_metas:
                m["source"] = "embedding"
            for m in pt_metas:
                m["source"] = "pattern"

            # Merge, deduplicate 
            seen_ids = set()
            merged_examples, merged_metas = [], []
            for ex, m in zip(em_examples + pt_examples, em_metas + pt_metas):
                if m["corpus_id"] not in seen_ids:
                    seen_ids.add(m["corpus_id"])
                    merged_examples.append(ex)
                    merged_metas.append(m)

            # Sort by score descending and keep top num_examples
            paired = sorted(zip(merged_examples, merged_metas), key=lambda x: x[1]["score"], reverse=True)
            paired = paired[:self.num_examples]

            if paired:
                top_examples, top_metas = zip(*paired)
                results.append((list(top_examples), list(top_metas)))
            else:
                results.append(([], []))

        return results

    def find_examples(self, address: str):
        return self.bulk_find_examples([address])[0]


# ---------------------------------------------------------------------------
# Fixed demo examples – curated hard cases for fallback
# ---------------------------------------------------------------------------
# Covers the structural patterns the model most often gets wrong:

FIXED_DEMO_EXAMPLES: list[tuple[str, dict]] = [
    # 1a. Slash stays in city – /Main qualifier
    (
        "Frankfurt/Main, Voltastr. 51",
        {"HouseNumber": "51", "StreetName": "Voltastr.", "City": "Frankfurt/Main",
         "District": "", "State": "", "Country": ""},
    ),
    # 1b. Slash stays in city – a.M. qualifier
    (
        "Frankfurt a.M.",
        {"HouseNumber": "", "StreetName": "", "City": "Frankfurt a.M.",
         "District": "", "State": "", "Country": ""},
    ),
    # 2a. City/Country slash – German region abbreviation is NOT State/Country
    (
        "Weener/Ostfr.",
        {"HouseNumber": "", "StreetName": "", "City": "Weener",
         "District": "", "State": "", "Country": ""},
    ),
    # 2b. City/Country slash – country extracted correctly
    (
        "Bergen/Norwegen, Rich. Nordrakagate 4",
        {"HouseNumber": "4", "StreetName": "Rich. Nordrakagate", "City": "Bergen",
         "District": "", "State": "", "Country": "Norwegen"},
    ),
    # 3a. HouseNumber with roman-numeral floor suffix
    (
        "Gräfelfing b.München, Hartnagelstr.1/I",
        {"HouseNumber": "1/I", "StreetName": "Hartnagelstr.", "City": "Gräfelfing",
         "District": "", "State": "", "Country": ""},
    ),
    # 3b. Block-style house number
    (
        "Asdod-Iam, Block 1049/8 Israel",
        {"HouseNumber": "Block 1049/8", "StreetName": "", "City": "Asdod-Iam",
         "District": "", "State": "", "Country": "Israel"},
    ),
    # 4.  Foreign address with State + Country
    (
        "Fall River/Mass 995 Walnut Street U.S.A.",
        {"HouseNumber": "995", "StreetName": "Walnut Street", "City": "Fall River",
         "District": "", "State": "Mass", "Country": "U.S.A."},
    ),
]


class FallbackExamplesStrategy(ExampleMatchingStrategy):
    """Wraps a primary strategy and replaces its output with fixed curated
    demo examples whenever the average retrieval score drops below threshold.
    """

    def __init__(
        self,
        primary: ExampleMatchingStrategy,
        labels_to_include: list[str],
        demo_examples: list[tuple[str, dict]] | None = None,
        threshold: float = 0.92,
        num_examples: int = 3,
    ):
        self.primary = primary
        self.num_examples = num_examples
        self.threshold = threshold
        raw = demo_examples if demo_examples is not None else FIXED_DEMO_EXAMPLES
        # Filter to requested labels, drop empty values so prompt stays clean
        self._fixed: list[tuple[str, OrderedDict]] = []
        for addr, labels in raw:
            filtered = OrderedDict(
                (k, v) for k, v in labels.items()
                if k in labels_to_include and v
            )
            self._fixed.append((addr, filtered))

    def _fixed_results(self) -> tuple[list, list]:
        examples = self._fixed[: self.num_examples]
        metadata = [
            {"source": "demo_fixed", "score": 1.0, "address": addr}
            for addr, _ in examples
        ]
        return list(examples), metadata

    def bulk_find_examples(self, addresses: list[str]):
        primary_results = self.primary.bulk_find_examples(addresses)
        out = []
        for examples, metadata in primary_results:
            scores = [m.get("score", 0) for m in metadata if isinstance(m, dict)]
            avg_score = sum(scores) / len(scores) if scores else 0
            if avg_score < self.threshold:
                out.append(self._fixed_results())
            else:
                out.append((examples, metadata))
        return out

    def find_examples(self, address: str):
        return self.bulk_find_examples([address])[0]


# ---------------------------------------------------------------------------
# NER-model-based pattern extraction
# ---------------------------------------------------------------------------

_BZK_LABELS = frozenset([
    "HouseNumber", "StreetName", "Neighborhood",
    "City", "District", "State", "Country",
])


def ner_address_to_pattern(address: str, nlp) -> str:
    """Convert an address to a structural pattern using a trained NER model.

    Entity spans are replaced with their label exactly once.
    Tokens in the gaps between entities fall back to the regex-based token
    If no entities are found the function falls back to address_to_pattern.

    Example:
        "Berlin-Marienfelde, Teichstr. 9"  →  "City, StreetName HouseNumber"
    """
    doc = nlp(address)
    ents_all = sorted(
        [e for e in doc.ents if e.label_ in _BZK_LABELS],
        key=lambda e: e.start,
    )

    if not ents_all:
        return address_to_pattern(address)

    def _gap_tokens(tokens):
        """Emit pattern parts for a sequence of non-entity tokens."""
        result = []
        for tok in tokens:
            if tok.is_space:
                result.append(" ")
            elif tok.like_num or tok.pos_ == "NUM":
                result.append("NUM")
            elif tok.is_punct or tok.is_bracket or tok.is_currency:
                result.append(tok.text)
            else:
                result.append("WORD")
            result.append(tok.whitespace_)
        return result

    def _entity_label_for(text: str) -> str:
        """Run a quick sub-prediction on a token part to resolve its entity label."""
        sub = nlp(text)
        return sub.ents[0].label_ if sub.ents else "WORD"

    def _expand_hyphenated(token, primary_label: str) -> str:
        """Split a single hyphenated token into a pattern like City-Neighborhood.
        """
        if "-" not in token.text:
            return primary_label
        sub_parts = token.text.split("-")
        result = [primary_label]
        for part in sub_parts[1:]:
            if not part:
                result.append("")
                continue
            result.append(_entity_label_for(part))
        return "-".join(result)

    parts = []
    ents = ents_all

    # Tokens before the first entity
    parts += _gap_tokens(t for t in doc if t.i < ents[0].start)

    for idx, ent in enumerate(ents):
        # Single hyphenated token
        if len(ent) == 1 and "-" in ent[0].text:
            parts.append(_expand_hyphenated(ent[0], ent.label_))
        else:
            parts.append(ent.label_)
        parts.append(ent[-1].whitespace_)

        # Tokens between this entity and the next (or end of doc)
        next_start = ents[idx + 1].start if idx + 1 < len(ents) else len(doc)
        parts += _gap_tokens(t for t in doc if ent.end <= t.i < next_start)

    return re.sub(r" {2,}", " ", "".join(parts)).strip()


class NERPatternSimilarExamples(ExampleMatchingStrategy):
    """Pattern-based few-shot selection using a trained spaCy NER model.
    """

    def __init__(
        self,
        example_addresses: pd.Series,
        example_labels: pd.DataFrame,
        num_examples: int,
        labels_to_include: list[str],
        model_dir: str = "models/ner_bzk",
    ):
        import spacy as _spacy
        self.nlp = _spacy.load(model_dir)

        self.example_addresses = example_addresses.reset_index(drop=True)
        self.example_labels    = example_labels[labels_to_include].reset_index(drop=True)
        self.num_examples      = min(num_examples, len(self.example_labels))
        self.labels_to_include = labels_to_include

        print(f"Computing NER patterns for {len(self.example_addresses)} training examples...")
        self.example_patterns = [
            ner_address_to_pattern(addr, self.nlp)
            for addr in self.example_addresses
        ]

    def bulk_find_examples(self, addresses: list[str]):
        from difflib import SequenceMatcher
        results = []
        for address in addresses:
            query_pattern = ner_address_to_pattern(address, self.nlp)

            candidates = []
            for idx, (example_addr, example_pattern) in enumerate(
                zip(self.example_addresses, self.example_patterns)
            ):
                score = SequenceMatcher(None, query_pattern, example_pattern).ratio()
                example_labels = {
                    k: v
                    for k, v in self.example_labels.iloc[idx].to_dict().items()
                    if pd.notna(v) and v != ""
                }
                candidates.append((score, idx, example_addr, example_labels, example_pattern))

            candidates.sort(key=lambda x: x[0], reverse=True)
            top = candidates[: self.num_examples]

            examples, metadatas = [], []
            for score, idx, example_addr, example_labels, example_pattern in top:
                included = score > 0.0  # any non-zero similarity is included
                metadatas.append({
                    "corpus_id":       idx,
                    "score":           score,
                    "pattern_score":   score,
                    "address":         example_addr,
                    "example":         example_labels,
                    "included":        included,
                    "query_pattern":   query_pattern,
                    "example_pattern": example_pattern,
                })
                if included:
                    examples.append((example_addr, example_labels))

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

def _compile_tuple_regex(separator):
    separator_escaped = re.escape(separator)
    # \"(?P<key>.*)\"\s*,\s*\"(?P<value>.*)\"
    return re.compile(
        r"\"(?P<key>.*)\"\s*" + 
        separator_escaped + 
        r"\s*\"(?P<value>.*)\""
    )

PRE_COMPILED_REGEXES = {s : _compile_tuple_regex(s) for s in [":", ","]}

def extract_tuples(model_response : str, separator=":"):
    regex = PRE_COMPILED_REGEXES.get(separator)
    if regex is None:
        regex = _compile_tuple_regex(separator)
    tuples = []
    for match in regex.finditer(model_response):
        tuples.append((match.group("key"), match.group("value")))
    return tuples

class JsonDictPromptTemplate(PromptTemplate):
    def format_example(self, example):
        return "```json" + json.dumps(example, ensure_ascii=False) + "```"
    
    def parse_output(self, response, original_address):
        try:
            json_str = extract_json_block(response)
            tuples = json.loads(json_str).items()
        except:
            tuples = extract_tuples(response, separator=":")
        result_builder = ParsedAddressResultBuilder(original_address)
        for k, v in tuples:
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
        try:
            json_str = extract_json_block(response)
            tuples = json.loads(json_str)
        except:
            tuples = extract_tuples(response, separator=",")
        result_builder = ParsedAddressResultBuilder(original_address)
        for part, ptype in tuples:
            if ptype != ignore_key:
                result_builder.add_part(ptype, part)
        data = result_builder.build()
        return data



class LLMAddressParsingModel:
    def __init__(self, 
                 model_name, 
                 prompt : PromptTemplate, 
                 example_strategy : ExampleMatchingStrategy | dict, 
                 *,
                 batch_size=32,
                 device=None,
                 max_new_tokens=512):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, padding_side='left', device=device)
        self.pipe = transformers.pipeline(
            "text-generation", model=model_name, 
            batch_size=batch_size, tokenizer=tokenizer, 
            device=device)
        self.generation_config = transformers.GenerationConfig(max_new_tokens=max_new_tokens)
        self.generate_kwargs = dict(generation_config=self.generation_config)
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
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks emitted by reasoning models"""
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _parse_output(self, conversation, original_address: str, example_metadata=None):
        model_response = None
        output_dict = None
        try:
            model_response = self._get_response(conversation)
            model_response = self._strip_thinking(model_response)
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

    @classmethod
    def _get_response(cls, conversation):
        return conversation[0]["generated_text"][1]["content"]

    def _make_conversation(self, address: str, examples : list[tuple[str, dict]]):
        return [
            {
                "role": "user", 
                "content": self.prompt.make_prompt(address, examples)
            }
        ]
    
    def _invoke_model(self, conversations):
        return self.pipe(conversations, **self.generate_kwargs)
    
    def parse_addresses(self, addresses : list[str]) -> list[dict[str, str]]:
        bulk_examples = self.example_strategy.bulk_find_examples(addresses)
        messages = [
            self._make_conversation(address, address_examples) 
            for address, (address_examples, _) in zip(addresses, bulk_examples)
        ]
        bulk_examples_metadata = [metadata for _, metadata in bulk_examples]
        result = self._invoke_model(messages)
        responses = [
            self._parse_output(r, original_address=addr, example_metadata=example_metadata)
            for r, addr, example_metadata in zip(result, addresses, bulk_examples_metadata)]
        return responses
    
class LlamaAddressParsingModel(LLMAddressParsingModel):
    pass

class MistralAddressParsingModel(LLMAddressParsingModel):
    pass

class DeepSeekAddressParsingModel(LLMAddressParsingModel):
    pass

class QwenAddressParsingModel(LLMAddressParsingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # According to https://huggingface.co/Qwen/Qwen3.5-9B#best-practices 
        # non thinking mode for general tasks recommendations
        self.generation_config.update(
            max_new_tokens=32_768 // 4, # Too many tokens can cause large latency
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0
        )

    @classmethod
    def _get_response(cls, conversation):
        response = conversation[0]["generated_text"].split("<|im_start|>assistant")[-1]
        return response

    def _invoke_model(self, conversations):
        # Claude's suggestion to disable thinking mode
        conversations = self.pipe.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return super()._invoke_model(conversations)

    def _make_conversation(self, address, examples):
        return [
            {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": self.prompt.make_prompt(address, examples)
                }]
            }
        ]
    

