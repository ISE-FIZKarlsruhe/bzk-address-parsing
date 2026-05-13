import deepparse.parser
from modules.utils import ParsedAddressResultBuilder


DEEPPARSE_LABEL_MAPPING = {
    "StreetNumber": "HouseNumber",
    "StreetName": "StreetName",
    "Municipality": "City",
    #"Province": "State", # note: DeepParse does not distinguish between state, province, region or country
    "Province": "Country",
    "PostalCode" : "PostalCode"
}

class DeepParseParser:
    def __init__(self, label_mapping = DEEPPARSE_LABEL_MAPPING, device = None, **kwargs):
        self.label_mapping = label_mapping
        self.parser = deepparse.parser.AddressParser(device=device, **kwargs)

    def _transform_results(self, parsed_addresses, addresses):
        results = []
        for parsed, addr in zip(parsed_addresses, addresses):
            result_builder = ParsedAddressResultBuilder(addr)
            tuples = parsed.to_list_of_tuples()
            for part, label in tuples:
                result_builder.add_part(self.label_mapping.get(label, label), part)
            results.append(result_builder.build())
        return results

    
    def parse_addresses(self, addresses):
        parsed_results = self.parser(addresses)
        return self._transform_results(parsed_results, addresses)