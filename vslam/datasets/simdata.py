import os
from typing import Iterable

import attr

from defs import ROOT_DIR
from sim.sim_types import Recording, Observation
from utils.serialization import msgpack_loads, from_native_types


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

    def stream(self) -> Iterable[Observation]:

        for obs in self.recorded_data.observations:
            yield obs


if __name__ == '__main__':
    dataset_path = os.path.join(ROOT_DIR, 'data/short_recording_2023-02-04--17-08-25.msgpack')

    data_streamer = SimDataStreamer.from_dataset_path(dataset_path=dataset_path)

    for obs in data_streamer.stream():
        print(obs.timestamp)


