import json
from utils import ParsedAddressResultBuilder, SEPARATOR_CHARS
import transformers

FIGHTING_CRIME_LABEL_MAPPING = {
    "Country" : "Country",
    "Province" : "State",
    "Municipality" : "City",
    "StreetName" : "StreetName",
    "StreetNumber" : "HouseNumber"
}
_FIGHTING_CRIME_TAG2ID = {'B-Country': 0,
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

for k, v in _FIGHTING_CRIME_TAG2ID.items():
    label = k[2:]  # Remove the "B-" or "I-" prefix
    new_label_mapping = {
        f"LABEL_{v}" : FIGHTING_CRIME_LABEL_MAPPING
    }
    if label in FIGHTING_CRIME_LABEL_MAPPING:
        FIGHTING_CRIME_LABEL_MAPPING[f"LABEL_{v}"] = FIGHTING_CRIME_LABEL_MAPPING[label]



class TokenClassifierAddressParser:
    def __init__(self, model_name: str, 
                 label_mapping: dict[int, str] = FIGHTING_CRIME_LABEL_MAPPING, 
                 batch_size: int = 16, 
                 device = None,
                 strip_separators = True
                 ):
        self.model_name = model_name
        self.label_mapping = label_mapping
        self.strip_separators = strip_separators
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipeline = transformers.pipeline("ner", model=self.model, 
                                              tokenizer=self.tokenizer, 
                                              batch_size=batch_size, device=device)

    def parse_addresses(self, addresses : list[str]) -> str:
        model_results = self.pipeline(addresses)
        results = []
        for address, result in zip(addresses, model_results):
            result_builder = ParsedAddressResultBuilder(address)
            for entity in result:
                tag = entity.get("entity_group") or entity['entity']
                if tag.startswith("B-") or tag.startswith("I-"):
                    tag = tag[2:]  # Remove the "B-" or "I-" prefix
                label = self.label_mapping.get(tag)
                if label is not None:
                    part = address[entity['start']:entity['end']]
                    result_builder.add_part(label, part)
            built_result = result_builder.build()
            if self.strip_separators:
                for k, v    in built_result.items():
                    v = v.lstrip(SEPARATOR_CHARS)
                    # words ending in a dot are likely abbreviations and the dot should be preserved
                    v = v.rstrip(SEPARATOR_CHARS.replace(".", ""))
                    built_result[k] = v
            results.append(built_result)
        return results
