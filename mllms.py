"""
Utility classes and functions for use with MLLMs for parsing addresses.
"""
import json
from abc import ABC, abstractmethod
from utils import ParsedAddressResultBuilder
import transformers

class ExampleMatchingStrategy(ABC):
    @abstractmethod
    def find_examples(self, address: str) -> list[tuple[str, dict]]:
        raise Exception("Not implemented")
    
class FixedExamples(ExampleMatchingStrategy):
    def __init__(self, examples: list[tuple[str, dict]]):
        self.examples = examples

    def find_examples(self, address: str) -> list[tuple[str, dict]]:
        return self.examples
    
class ZeroShot(ExampleMatchingStrategy):
    def find_examples(self, address: str) -> list[tuple[str, dict]]:
        return []

class PromptTemplate(ABC):
    def __init__(self, template: str, 
                 examples_prefix = "Consider the following examples:\n", 
                 example_template = "Address: {address}\n{example}\n"):
        self.template = template
        self.examples_prefix = examples_prefix
        self.example_template = example_template

    def make_prompt(self, address: str, examples : list[tuple[str, dict]]) -> dict:
        formatted_examples = [self.example_template.format(address=addr, example=self.format_example(example)) for addr, example in examples]
        if len(formatted_examples) > 0:
            examples = self.examples_prefix + "".join(formatted_examples)
        else:
            examples = ""
        return self.template % {"address": address, "examples": examples}

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
        return "```json" + json.dumps(example) + "```"
    
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
                 example_template = "Address: {address}\n{example}\n",
                 ignore_other = True):
        super().__init__(template, examples_prefix, example_template)
        self.ignore_other = ignore_other

    def format_example(self, example):
        return "```json" + json.dumps([[v, k] for k, v in example.items()]) + "```"
    
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
    def __init__(self, model_name, prompt : PromptTemplate, example_strategy : ExampleMatchingStrategy, batch_size=32, device=None):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, padding_side='left', device=device)
        self.pipe = transformers.pipeline("text-generation", model=model_name, 
                                          batch_size=batch_size, tokenizer=tokenizer)
        self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id[0]
        self.example_strategy = example_strategy
        self.prompt = prompt

    def _parse_output(self, conversation, original_address: str):
        model_response = None
        try:
            model_response = conversation[0]["generated_text"][1]["content"]
            parsed = self.prompt.parse_output(model_response, original_address=original_address)
            parsed["fullConversation"] = json.dumps(conversation)
            return parsed
        except Exception as e:
            print(f"Error parsing model output for address '{original_address}': {e}\n"
                  f"Full Conversation: {conversation}\n"
                  f"Model response: {repr(model_response)}")
            return {"error": str(e), "fullConversation": json.dumps(conversation)}

    def parse_addresses(self, addresses : list[str]) -> str:
        messages = [[
            {
                "role": "user", 
                "content": self.prompt.make_prompt(address, self.example_strategy.find_examples(address))
            }
        ] for address in addresses]
        result = self.pipe(messages)
        responses = [self._parse_output(r, original_address=addr) for r, addr in zip(result, addresses)]
        return responses