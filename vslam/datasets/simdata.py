from typing import Iterable

import attr

from sim.sim_types import Recording, Observation, CameraSpecs
from utils.serialization import msgpack_loads, from_native_types
from vslam.cam import CameraIntrinsics


@attr.define
class SimDataStreamer:
    """Simulated data streamer. """
    recorded_data: Recording

    @classmethod
    def from_dataset_path(cls, dataset_path):
        """Create a data streamer from a dataset path. """

        with open(dataset_path, 'rb') as f:
            raw_data = f.read()

        dict_data = msgpack_loads(raw_data)
        data = from_native_types(dict_data, Recording)

        return cls(recorded_data=data)

    def get_cam_intrinsics(self) -> CameraIntrinsics:
        return self.recorded_data.camera_specs.intrinsics

    def get_cam_specs(self) -> CameraSpecs:
        return self.recorded_data.camera_specs

    def stream(self) -> Iterable[Observation]:

        for obs in self.recorded_data.observations:
            yield obs

