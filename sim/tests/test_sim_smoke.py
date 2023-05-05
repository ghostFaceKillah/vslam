from sim.run import run_simulation


def test_sim_smoke():
    """ Check if simulation runs at all. """
    run_simulation(manual=False, short_trip=True, save_recording=False)
