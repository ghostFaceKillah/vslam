import attr
from typing import Iterable, Optional

from sim.sim_types import Recording, Observation, CameraSpecs
from utils.serialization import msgpack_loads, from_native_types
from vslam.cam import CameraIntrinsics


@attr.define
class SimDataStreamer:
    """Simulated data streamer. """
    recorded_data: Recording
    max_obs: Optional[int] = None

    @classmethod
    def from_dataset_path(
            cls,
            dataset_path: str,
            max_obs: Optional[int] = None
    ):
        """Create a data streamer from a dataset path. """

        with open(dataset_path, 'rb') as f:
            raw_data = f.read()

        dict_data = msgpack_loads(raw_data)
        data = from_native_types(dict_data, Recording)

        return cls(recorded_data=data, max_obs=max_obs)

    def get_cam_intrinsics(self) -> CameraIntrinsics:
        return self.recorded_data.camera_specs.intrinsics

    def get_cam_specs(self) -> CameraSpecs:
        return self.recorded_data.camera_specs

    def stream(self) -> Iterable[Observation]:

        for i, obs in enumerate(self.recorded_data.observations):
            yield obs

            if self.max_obs is not None and i > self.max_obs:
                break

