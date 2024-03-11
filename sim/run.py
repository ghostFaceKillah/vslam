import lz4.frame

from sim.actor_simulation import TriangleSceneRenderer, PreRecordedActor, Simulation, ManualActor
from utils.file_utils import easy_filename
from utils.profiling import just_time
from utils.serialization import to_native_types, msgpack_dumps


def run_simulation(
    manual: bool = False,
    save_recording: bool = True,
    short_trip: bool = False
):
    scene_renderer = TriangleSceneRenderer.from_default()

    if manual:
        actor = ManualActor.from_default()
    else:
        actor = PreRecordedActor.from_a_nice_trip(short_trip=short_trip)

    sim = Simulation.from_defaults(
        actor=actor,
        scene_renderer=scene_renderer
    )

    with just_time('simulating'):
        recording = sim.simulate()

    with just_time('serializing'):
        native_types_data = to_native_types(recording)
        data = msgpack_dumps(native_types_data)
        print(f"size of recording is {len(data) / 1024 / 1024:.2f} mb")
        print(f"size of compressed recording is {len(lz4.frame.compress(data)) / 1024 / 1024:.2f} mb")

    if save_recording:
        fpath = easy_filename('short_recording.msgpack')
        print(f"writing to {fpath}...")

        with open(fpath, 'wb') as f:
            f.write(data)


if __name__ == '__main__':
    """
    On mac air m1, short trip = False
    ... Elapsed 352.4s in: simulating for rendering and recording 1214 frames
    size of recording is 6382121789  ~ 6 Gbs of data, whaat
    size of compressed recording is 74317519 ~ 70 mbs of data
    around 0.29 s per frame
    around 5 mb per non-compressed frame
    around 0.058 mb per compressed frame
    """
    run_simulation(manual=True, save_recording=False)
