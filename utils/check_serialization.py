import attr
import numpy as onp

from utils.custom_types import BGRImageArray


@attr.define
class FakeMessage:
    img_left: BGRImageArray = attr.ib(repr=False)
    right_left: BGRImageArray = attr.ib(repr=False)
    timestamp: int

    @classmethod
    def fake(cls):
        return cls(
            img_left=onp.random.rand(100, 100, 3),
            right_left=onp.random.rand(100, 100, 3),
            timestamp=123456789,
        )




from sim.sim_types import CameraSpecs
from cattr import GenConverter
import cattrs
import numpy as np


def test_camera_specs_serialization():
    camera_specs = CameraSpecs.from_default()
    data = cattrs.unstructure(camera_specs)


    converter = GenConverter()

    converter.register_structure_hook_func(
        lambda t: getattr(t, "__origin__", None) is np.ndarray,
        lambda v, t: v
    )

    re_camera_specs = converter.structure(data, CameraSpecs)
    print(re_camera_specs)



if __name__ == '__main__':
    fake_msg = FakeMessage.fake()
    print(fake_msg)