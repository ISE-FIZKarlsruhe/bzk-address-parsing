import json
from utils import StrictMergeParsedResultBuilder, SEPARATOR_CHARS
import transformers
from transformers.pipelines.token_classification import AggregationStrategy, TokenClassificationPipeline
import numpy as np
import re
import warnings

FIGHTING_CRIME_LABEL_MAPPING = {
    "Country" : "Country",
    "CountryCode" : "Country",
    "Province" : "State",
    "Municipality" : "City",
    "StreetName" : "StreetName",
    "StreetNumber" : "HouseNumber"
}

_FIGHTING_CRIME_TAG2ID = {
 'B-Country': 0,
 'B-CountryCode': 1,
 'B-HardSep': 2,
 'B-Municipality': 3,
 'B-Name': 4,
 'B-OOV': 5,
 'B-PostalCode': 6,
 'B-Province': 7,
 'B-StreetName': 8,
 'B-StreetNumber': 9,
 'B-Unit': 10,
 'I-Country': 11,
 'I-Municipality': 12,
 'I-Name': 13,
 'I-OOV': 14,
 'I-PostalCode': 15,
 'I-Province': 16,
 'I-StreetName': 17,
 'I-StreetNumber': 18,
 'I-Unit': 19
}



_FIGHTING_CRIME_ID2TAG = {f"LABEL_{v}": k for k, v in _FIGHTING_CRIME_TAG2ID.items()}

WORD_REGEX = re.compile(r"(\w+)|(\W+)") # Separates words from word boundaries, creating a separate match for each

class CustomTokenClassificationPipeline(TokenClassificationPipeline):
    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: list[tuple[int, int]] | None,
        special_tokens_mask: np.ndarray,
        aggregation_strategy: AggregationStrategy,
        word_ids: list[int | None] | None = None,
        word_to_chars_map: list[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """CUSTOM IMPLEMENTATION - Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        # This is a custom implementation of the gather_pre_entities method from transformers.pipelines.token_classification.TokenClassificationPipeline
        # It is modified to distinguish subwords without relying on the tokenizer to be word aware
        # It may not work on every situation but maybe I should create a pull request to transformers anyway
        pre_entities = []
        word_spans = None
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens
            if special_tokens_mask[idx]:
                continue
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]

                # If the input is pre-tokenized, we need to rescale the offsets to the absolute sentence.
                if word_ids is not None and word_to_chars_map is not None:
                    word_index = word_ids[idx]
                    if word_index is not None:
                        start_char, _ = word_to_chars_map[word_index]
                        start_ind += start_char
                        end_ind += start_char

                if not isinstance(start_ind, int):
                    start_ind = start_ind.item()
                    end_ind = end_ind.item()
                word_ref = sentence[start_ind:end_ind]
                if getattr(self.tokenizer, "_tokenizer", None) and getattr(
                    self.tokenizer._tokenizer.model, "continuing_subword_prefix", None
                ):
                    # This is a BPE, word aware tokenizer, there is a correct way
                    # to fuse tokens
                    is_subword = len(word) != len(word_ref)
                else:
                    # This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                    if aggregation_strategy in {
                        AggregationStrategy.FIRST,
                        AggregationStrategy.AVERAGE,
                        AggregationStrategy.MAX,
                    }:
                        warnings.warn(
                            "Tokenizer does not support real words, using fallback heuristic",
                            UserWarning,
                        )
                    # Original heuristic for detecting subwords
                    # is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1]

                    # New regex span relying heuristic (Not guaranteed to work but a better heuristic)
                    # Split sentence into words (or word boundaries) if not already done
                    word_spans = word_spans or [(m.start(), m.end()) for m in WORD_REGEX.finditer(sentence)]
                    is_subword = not any(
                        start_ind <= word_start and word_start < max(end_ind, start_ind + 1)
                        for word_start, word_end in word_spans 
                    )

                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False
            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

class TokenClassifierAddressParser:
    def __init__(self, model_name: str, 
                 label_mapping: dict[int, str] = FIGHTING_CRIME_LABEL_MAPPING,
                 tag_bio_label_mapping: dict[str, int] = _FIGHTING_CRIME_ID2TAG,
                 batch_size: int = 16, 
                 device = None,
                 strip_separators = True,
                 aggregation_strategy = "simple"
                 ):
        self.model_name = model_name
        self.label_mapping = label_mapping
        self.tag_bio_label_mapping = tag_bio_label_mapping
        self.strip_separators = strip_separators
        self.aggregation_strategy = aggregation_strategy
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
        pipeline_class = None
        if not (getattr(self.tokenizer, "_tokenizer", None) and getattr(
                self.tokenizer._tokenizer.model, "continuing_subword_prefix", None
            )) and aggregation_strategy in ["first", "average", "max"]:
            warnings.warn(
                "Tokenizer is not word aware and aggregation strategy " + aggregation_strategy + 
                " requires word awareness. Modifying transformers pipeline implementation"
                " to support it. This will likely break accross transformers library updates."
            )
            # patch the pipeline to use the custom gather_pre_entities method 
            #that does not rely on the tokenizer to be word aware
            pipeline_class = CustomTokenClassificationPipeline
        self.pipeline = transformers.pipeline("ner", model=self.model, 
                                              tokenizer=self.tokenizer, 
                                              batch_size=batch_size, device=device,
                                              aggregation_strategy=aggregation_strategy,
                                              pipeline_class=pipeline_class)
        
    
    def _solve_label(self, label):
        prefix = ""
        if self.tag_bio_label_mapping:
            label = self.tag_bio_label_mapping.get(label, label)
        if label.startswith("B-") or label.startswith("I-"):
            prefix = label[:2]
            label = label[2:]  # Remove the "B-" or "I-" prefix
        if self.label_mapping:
            label = self.label_mapping.get(label, label)
        return prefix, label
    
    def parse_addresses(self, addresses : list[str]) -> str:
        model_results = self.pipeline(addresses)
        results = []
        for address, result in zip(addresses, model_results):
            result_builder = StrictMergeParsedResultBuilder(address)
            #result = self._aggregate(result, address)
            for entity in result:
                label = self._solve_label(entity.get("entity_group") or entity['entity'])[1]
                if label is not None:
                    part = address[entity['start']:entity['end']]
                    result_builder.add_part(label, part, entity['start'])
            built_result = result_builder.build()
            if self.strip_separators:
                for k, v    in built_result.items():
                    v = v.lstrip(SEPARATOR_CHARS)
                    # words ending in a dot are likely abbreviations and the dot should be preserved
                    v = v.rstrip(SEPARATOR_CHARS.replace(".", ""))
                    built_result[k] = v
            results.append(built_result)
        return results
