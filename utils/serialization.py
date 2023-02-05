from typing import Any, Dict, Type, TypeVar

import cattrs
import msgpack
import msgpack_numpy as m
import numpy as np
import numpy as onp
from cattr import GenConverter


def msgpack_dumps(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True, default=m.encode)


def msgpack_loads(data: bytes):
    return msgpack.unpackb(data, raw=False, object_hook=m.decode)


def to_native_types(obj: Any) -> Dict[str, Any]:
    return cattrs.unstructure(obj)


_CONVERTER = None


def _get_converter_singleton():
    global _CONVERTER

    if _CONVERTER is None:
        converter = GenConverter()

        converter.register_structure_hook_func(
            lambda t: getattr(t, "__origin__", None) is np.ndarray,
            lambda v, t: v
        )

        _CONVERTER = converter

    return _CONVERTER


T = TypeVar('T')


def from_native_types(data: Dict[str, Any], target_type: Type[T]) -> T:
    return _get_converter_singleton().structure(data, target_type)


def experiment(obj):

    the_in = msgpack.packb(obj, use_bin_type=True, default=m.encode)
    the_out = msgpack.unpackb(the_in, raw=False, object_hook=m.decode)

    print(f"{the_in=}")
    print(f"{the_out=}")


if __name__ == '__main__':
    experiment([1, 2, 3])
    an_array_ok = onp.array([1, 2, 3, 4], onp.float32)
    experiment(an_array_ok)


