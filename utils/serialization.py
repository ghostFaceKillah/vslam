from typing import Any, Dict

import cattrs
import msgpack
import msgpack_numpy as m
import numpy as onp


def msgpack_dumps(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True, default=m.encode)


def msgpack_loads(data: bytes):
    return msgpack.unpackb(data, raw=False, object_hook=m.decode)


def to_native_types(obj: Any) -> Dict[str, Any]:
    return cattrs.unstructure(obj)


def experiment(obj):

    the_in = msgpack.packb(obj, use_bin_type=True, default=m.encode)
    the_out = msgpack.unpackb(the_in, raw=False, object_hook=m.decode)

    print(f"{the_in=}")
    print(f"{the_out=}")


if __name__ == '__main__':
    experiment([1, 2, 3])

    an_array_ok = onp.array([1, 2, 3, 4], onp.float32)
    experiment(an_array_ok)

