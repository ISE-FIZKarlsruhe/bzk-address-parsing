from typing import Mapping, get_args


class FrozenDict[K, V](Mapping[K, V]):
    """
    A simple immutable dictionary implementation.
    """

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash

    def __eq__(self, other):
        if isinstance(other, FrozenDict):
            return self._dict == other._dict
        return False

    def __repr__(self):
        return f"FrozenDict({self._dict})"
    
    def __dict_encode__(self, default_encoder):
        return default_encoder(self._dict.copy())

    @classmethod
    def __dict_decode__(cls, data: dict, targs, default_decoder):
        if len(targs) == 2:
            data = default_decoder(data, dict[targs[0], targs[1]])
        else:
            data = default_decoder(data, dict)
        return FrozenDict(data)