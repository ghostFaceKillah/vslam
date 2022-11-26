import contextlib
import time

_JUST_TIME_IT_DEPTH = 0


@contextlib.contextmanager
def just_time(what='timer', verbose=True):
    global _JUST_TIME_IT_DEPTH
    depth = _JUST_TIME_IT_DEPTH
    _JUST_TIME_IT_DEPTH += 1
    resu_state = {}
    if verbose:
        print(f'{4 * depth}Entering: {what} ...')
    start_time = time.perf_counter()
    try:
        yield resu_state
    finally:
        _JUST_TIME_IT_DEPTH-=1
        elapsed = time.perf_counter() - start_time
        resu_state['elapsed'] = elapsed
        if verbose:
            print(f'{4 * depth}... Elapsed {elapsed:.4g}s in: {what}')
