from json import JSONEncoder, JSONDecoder
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import NamedTuple
import types
import inspect
from contextlib import contextmanager
import pandas as pd
from pprint import pformat
from datetime import datetime
from typing import Optional, Union
import typing


# TODO consider using or extending existing libraries for this, 
# e.g. pydantic, marshmallow, dataclasses-json, etc.
# Preliminary analysis reveals that none of the listed have quite the same applicability


def encode_as_dict(obj) -> dict:
    def _encode(obj, skip_custom = False):
        annots = getattr(obj, '__annotations__', None) or getattr(type(obj), '__annotations__', None)
        if not skip_custom and hasattr(obj, "__dict_encode__") and callable(getattr(obj, "__dict_encode__")):
            def default_encoder(obj):
                return _encode(obj, skip_custom=True)
            return obj.__dict_encode__(default_encoder)
        elif isinstance(obj, Enum):
            return obj.name
        elif is_dataclass(obj):
            values = {
                f.name: _encode(getattr(obj, f.name)) for f in fields(obj)
            }
            return values
        elif annots is not None:
            values = {
                k: encode_as_dict(getattr(obj, k)) for k in annots.keys()
            }
            return values
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)) or issubclass(type(obj), (list, tuple)):
            return [ encode_as_dict(v) for v in obj ]
        elif isinstance(obj, dict) or issubclass(type(obj), dict):
            return { encode_as_dict(k): encode_as_dict(v) for k, v in obj.items() }
        else:
            raise TypeError(f"Object of type {obj.__class__.__name__} cannot be represented as a dict")
    return _encode(obj, skip_custom = False)
    

@dataclass
class DecodingOptions:
    none_to_empty_list : bool = True
    none_to_false : bool = True
    none_to_nan : bool = True
    none_to_zero : bool = True

default_decoding_options = DecodingOptions()

def _stack_trace_to_str(object_stack_trace : list[str]):
    return ''.join(object_stack_trace) if len(object_stack_trace) > 0 else "<root>"

def decode_from_dict(data, expected_type : type, options: Optional[DecodingOptions] = None):
    decoding_options = options if options is not None else default_decoding_options
    @contextmanager
    def _push_to_stack(obj_repr, object_stack_trace : list[str]):
        object_stack_trace.append(obj_repr)
        try:
            yield ''
            object_stack_trace.pop()
        except: raise
    def _decode(data, expected_type : type, object_stack_trace : list[str], skip_custom = False):
        origin_type = typing.get_origin(expected_type)
        arg_types = typing.get_args(expected_type)

        # String type hints
        if isinstance(expected_type, str):
            raise TypeError(f"String literal type '{expected_type}' cannot be decoded, "
                            f"replace with explicit type {expected_type}")
        
        # Custom decoding behaviour
        elif not skip_custom and hasattr(expected_type, "__dict_decode__") and callable(getattr(expected_type, "__dict_decode__")):
            def default_decoder(data, expected_type):
                return _decode(data, expected_type, object_stack_trace, skip_custom=True)
            return expected_type.__dict_decode__(data, arg_types, default_decoder)
        
        # Not a type
        elif not isinstance(expected_type, type) and origin_type is None:
            raise TypeError(f"Expected type must be a type object, got {expected_type} of type {type(expected_type)}")
        
        # Optional types
        elif origin_type in [typing.Union, types.UnionType] and len(arg_types) == 2 and type(None) in arg_types:
            non_none_type = arg_types[0] if arg_types[1] is type(None) else arg_types[1]
            if data is None:
                return None
            else:
                return _decode(data, non_none_type, object_stack_trace)

        # Unions and similar types
        elif origin_type in [typing.Union, types.UnionType]:
            exceptions = []
            for arg_type in arg_types:
                try:
                    sub_stack_trace = object_stack_trace.copy()
                    return _decode(data, arg_type, sub_stack_trace)
                except Exception as e:
                    exceptions.append((arg_type, e))
                    continue
            raise TypeError(
                f"Data type {type(data)} does not match any of the possible types in {expected_type}."
                f" Tried decoding as:\n" + "\n".join(f"  - {arg_type}: failed at {_stack_trace_to_str(sub_stack_trace)} with {repr(e)}" for arg_type, e in exceptions)
            )
        
        # typed lists and tuples
        elif origin_type in (list, tuple) or issubclass(origin_type.__class__, (list, tuple)):
            if decoding_options.none_to_empty_list and data is None:
                data = []
            elif not isinstance(data, (list, tuple)):
                raise TypeError(
                    f"Data type {type(data)} does not match target type {expected_type}, "
                    f"at {''.join(object_stack_trace)}"
                )
            result_list = []
            if len(arg_types) == 2 and arg_types[1] is Ellipsis:
                arg_types = (arg_types[0],)
            for i, item in enumerate(data):
                with _push_to_stack(f"[{i}]", object_stack_trace):
                    item_type = arg_types[0] if len(arg_types) == 1 else arg_types[i]
                    result_list.append(_decode(item, item_type, object_stack_trace))
            return expected_type(result_list)
        
        # typed dicts (basic generic type parameters, not the TypedDict from typing)
        elif origin_type == dict or issubclass(origin_type.__class__, dict):
            if not isinstance(data, dict):
                raise TypeError(
                    f"Data type {type(data)} does not match target type {expected_type}, "
                    f"at {''.join(object_stack_trace)}"
                )
            key_type = arg_types[0]
            value_type = arg_types[1]
            result_dict = origin_type()
            for key, value in data.items():
                with _push_to_stack(f"[{repr(key)}]", object_stack_trace):
                    result_dict[_decode(key, key_type, object_stack_trace)] = _decode(value, value_type, object_stack_trace)
            return result_dict
        
        # enums
        elif issubclass(expected_type, Enum):
            try:
                return expected_type[data]
            except Exception as e:
                try:
                    return expected_type(data)
                except:
                    raise e

        # dataclasses
        elif is_dataclass(expected_type):
            field_types = {f.name: f.type for f in fields(expected_type)}
            init_values = {}
            for key, field_type in field_types.items():
                with _push_to_stack(f".{key}", object_stack_trace):
                    init_values[key] = _decode(data[key], field_type, object_stack_trace)
            return expected_type(**init_values)
        
        

        # Datetime objects
        elif expected_type == datetime or issubclass(expected_type, datetime):
            if isinstance(data, expected_type):
                return data
            elif not isinstance(data, str):
                raise TypeError(
                    f"Data type {type(data)} does not match target type {expected_type}, "
                    f"at {''.join(object_stack_trace)}"
                )
            return datetime.fromisoformat(data)
        
        # Other dataclass like objects (eg. NamedTuple)
        elif hasattr(expected_type, '__annotations__'):
            field_types = expected_type.__annotations__
            init_values = {}
            for key, field_type in field_types.items():
                with _push_to_stack(f".{key}", object_stack_trace):
                    init_values[key] = _decode(data[key], field_type, object_stack_trace)
            return expected_type(**init_values)
        
        # Primitive types
        elif expected_type in (str, int, float, bool, type(None)):
            if isinstance(data, expected_type):
                return data
            
            # Try to conform data to expected type
            if pd.isna(data):
                data = None
            if data is None and expected_type is not type(None):
                if expected_type is int and decoding_options.none_to_zero:
                    return 0
                elif expected_type in (float, int) and decoding_options.none_to_nan:
                    return float('nan')
                elif expected_type is bool and decoding_options.none_to_false:
                    return False
            if isinstance(data, expected_type):
                return data
            else:
                raise TypeError(
                    f"Data type {type(data)} does not match target type {expected_type}, "
                    f"at {''.join(object_stack_trace)}"
                )
        
        # not typed lists and tuples
        elif issubclass(expected_type, (list, tuple)):
            if not isinstance(data, expected_type):
                return expected_type(data)
            else:
                return data
        
        # not typed dicts
        elif issubclass(expected_type, dict):
            if not isinstance(data, expected_type):
                return expected_type(**data)
            else:
                return data
            

        else:
            raise TypeError(f"Unsupported type {expected_type}")

    root_stack_trace = []
    try:
        return _decode(data, expected_type, root_stack_trace)
    except:
        # Attach more exception context
        raise Exception(
            f"Error decoding data of expected_type {expected_type}"
            f" at {_stack_trace_to_str(root_stack_trace)}:\n{pformat(data)}"
        )

def assert_same(obj1, obj2):
    stack_trace = []
    def _assert_same(obj1, obj2):
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            if obj1.keys() != obj2.keys():
                raise AssertionError(f"Key mismatch: {obj1.keys()} != {obj2.keys()} at {_stack_trace_to_str(stack_trace)}")
            for k in obj1.keys():
                stack_trace.append(f"[{repr(k)}]")
                _assert_same(obj1[k], obj2[k])
                stack_trace.pop()
        elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
            if len(obj1) != len(obj2):
                raise AssertionError(f"Length mismatch: {len(obj1)} != {len(obj2)} at {_stack_trace_to_str(stack_trace)}")
            for i in range(len(obj1)):
                stack_trace.append(f"[{i}]")
                _assert_same(obj1[i], obj2[i])
                stack_trace.pop()
        elif type(obj1) != type(obj2):
            raise AssertionError(f"Type mismatch: {type(obj1)} != {type(obj2)} at {_stack_trace_to_str(stack_trace)}")
        elif is_dataclass(obj1):
            for f in fields(obj1):
                stack_trace.append(f".{f.name}")
                _assert_same(getattr(obj1, f.name), getattr(obj2, f.name))
                stack_trace.pop()
        elif obj1 != obj2:
            raise AssertionError(f"Value mismatch: {obj1} != {obj2} at {_stack_trace_to_str(stack_trace)}")
    _assert_same(obj1, obj2)