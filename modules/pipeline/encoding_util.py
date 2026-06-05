from json import JSONEncoder, JSONDecoder
from dataclasses import fields, is_dataclass, asdict
from enum import Enum
from typing import NamedTuple
import types
import inspect
from contextlib import contextmanager
import pandas as pd
from pprint import pformat


def is_type_decodeable(t) -> bool:
    # TODO
    raise NotImplementedError()

class PipelineDataEncoderDecoder:
    """
        Utility class for encoding and decoding pipeline 
        data structures to and from basic python data types 
        (dict, list, str, int, float, bool, None)
        To be used for JSON serialization and deserialization and 
        hierarchical DataFrame conversion
    """
    def __init__(self, minimize_objects_if_possible : bool = True):
        self.minimize_objects_if_possible = minimize_objects_if_possible


    def encode(self, obj) -> dict:
        if self.minimize_objects_if_possible and hasattr(obj, "minimize") and callable(getattr(obj, "minimize")):
            obj = obj.minimize()
        if hasattr(obj, "__dict_encode__") and callable(getattr(obj, "__dict_encode__")):
            return obj.__dict_encode__(default_encoder=self)
        if is_dataclass(obj):
            values = {
                k: self.encode(v) for k, v in asdict(obj).items()
            }
            return values
        elif isinstance(obj, NamedTuple):
            values = {
                k: self.encode(getattr(obj, k)) for k in obj._fields
            }
            return values
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (str, int, float, bool, type(None), list, tuple, dict)):
            return obj
        elif issubclass(type(obj), (list, tuple, dict)):
            return obj
        else:
            raise TypeError(f"Object of type {obj.__class__.__name__} cannot be represented as a dict")

    def decode(self, data , expected_type : type):
        object_stack_trace = []
        @contextmanager
        def _push_to_stack(obj_repr):
            object_stack_trace.append(obj_repr)
            try: yield
            finally: object_stack_trace.pop()
        def _decode(data, expected_type : type):
            if isinstance(expected_type, str):
                raise TypeError(f"String literal type '{expected_type}' cannot be decoded, "
                                f"replace with explicit type {expected_type}")
            elif hasattr(expected_type, "__dict_decode__") and callable(getattr(expected_type, "__dict_decode__")):
                nested = {}
                for k, v in data.items():
                    with _push_to_stack(f".{k}"):
                        nested[k] = v
                return expected_type.__dict_decode__(data)
            elif not isinstance(expected_type, type):
                raise TypeError(f"Expected type must be a type object, got {expected_type} of type {type(expected_type)}")
            elif isinstance(expected_type, types.GenericAlias):
                if type(data) != expected_type.__origin__ and not isinstance(data, expected_type.__origin__):
                    raise TypeError(
                        f"Data type {type(data)} "
                        f"does not match target type {expected_type}"
                    )
                if expected_type.__origin__ in (list, tuple) or issubclass(expected_type.__origin__, (list, tuple)):
                    result_list = expected_type.__origin__()
                    for i, item in enumerate(data):
                        with _push_to_stack(f"[{i}]"):
                            item_type = expected_type.__args__[0] if len(expected_type.__args__) == 1 else expected_type.__args__[i]
                            result_list.append(_decode(item, item_type))
                    return result_list
                elif expected_type.__origin__ == dict or issubclass(expected_type.__origin__, dict):
                    value_type = expected_type.__args__[1]
                    result_dict = expected_type.__origin__()
                    for key, value in data.items():
                        with _push_to_stack(f"[{repr(key)}]"):
                            result_dict[key] = _decode(value, value_type)
                    return result_dict
            elif is_dataclass(expected_type):
                field_types = {f.name: f.type for f in fields(expected_type)}
                init_values = {}
                for key, value in data.items():
                    with _push_to_stack(f".{key}"):
                        init_values[key] = _decode(value, field_types[key])
                return expected_type(**init_values)
            elif issubclass(expected_type, Enum):
                return expected_type(data)
            elif issubclass(expected_type, NamedTuple):
                field_types = expected_type._field_types
                init_values = {}
                for key, value in data.items():
                    with _push_to_stack(f".{key}"):
                        init_values[key] = _decode(value, field_types[key])
                return expected_type(**init_values)
            elif expected_type in (str, int, float, bool, type(None)):
                if not isinstance(data, expected_type):
                    if pd.isna(data) and expected_type == type(None):
                        return None
                    else:
                        raise TypeError(
                            f"Data type {type(data)} does not match target type {expected_type}, "
                            f"at {''.join(object_stack_trace)}"
                        )
                return data
            elif issubclass(expected_type, (list, tuple)):
                if not isinstance(data, expected_type):
                    return expected_type(*data)
                else:
                    return data
            elif issubclass(expected_type, dict):
                if not isinstance(data, expected_type):
                    return expected_type(**data)
                else:
                    return data
            else:
                raise TypeError(f"Unsupported type {expected_type}")
        try:
            return _decode(data, expected_type)
        except:
            # Attach more exception context
            raise Exception(
                f"Error decoding data of expected_type {expected_type}"
                 f" at {''.join(object_stack_trace)}:\n{pformat(data)}"
            )