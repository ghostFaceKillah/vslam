VSLAM
=====

Intro
-----

Here's an easy-to-understand Visual Simultaneous Localization And Mapping (VSLAM) algorithm.

![render](imgs/gui.gif)

I made a whole [youtube video series](https://www.youtube.com/playlist?list=PLENZR8id1crgCG9Dmr0uKE5mz5eE8S_0-) to explain it.

If you want to quickly get to the meat of the code, go to [`vslam/frontend.py` and read `Frontend.track()`](https://github.com/ghostFaceKillah/vslam/blob/main/vslam/frontend.py#L238) function - that's what gets called on every iteration to resolve pose from image input.

It works on top of data coming from kinda-easy-to-understand triangle-based scene rendering from scratch.

The VSLAM part of this repo is largely reinterpretation of tutorials presented in an ***excellent*** book 
*"Introduction to Visual SLAM: From Theory to Practice"*. See the associated
[github repo](https://github.com/gaoxiang12/slambook) provided by the authors of the book.
They also generously [provided pdf of the book itself in a realted repo](https://github.com/gaoxiang12/slambook-en/blob/master/slambook-en.pdf).
This is very awesome, and I am grateful for that.

Main entry point into VSLAM demo is:

```
pipenv shell
python -m lessons.ex_03_full_frontend
```

This live-renders the environment. That's a bit too slow to feel real-time (at around 1 fps).
It's better to first pre-generate the data by `python -m sim.run` and the run it from saved data.

Structure
----------

- `vslam`
    - `lessons` - scripts that run the framework piece by piece
    - `vslam`  - the vslam library - doesn't use jax.
        - `keyframe` - contains the most important functions that drive the SLAM algorithm
            - `def estimate_keyframe()`
            - `def estimate_pose_wrt_keyframe()`
        - `frontend` - Is the primary VSLAM state holder. Pulls everything together.
    - `sim` - rendering framework. Uses `jax`.
        - `egocentric_render` - contains the most important functions that drive rendering
            - `def parallel_z_buffer_render()` - that's the function that does the object drawing



Installing
----------

I recommend using `pipenv`.

```
python -m pip install pipenv
pipenv --python 3.10   # this assumes that you have python 3.10 installed
pipenv shell
pip install -r requirements.txt
```

To briefly summarize, we depend mostly on `attrs`, `numpy`, `jax` (!), `opencv-python`, `pandas` and `scipy`.
`jax` is not critical and could be done away with in favour of numpy, but it keeps things fast.


Rendering
----------

I made a small triangle rendering library to make data for VSLAM.

This way we fully control the data coming into the algorithm and we get to learn the camera equations
from the "inverse problem" side. Indeed, rendering is inverse of SLAM in a way.
Below command runs the entry point of the interactive rendering code.

```
pipenv shell
python -m sim.render
```
Sorry it's too slow to feel smooth to humans!
It seems we would need to rewrite in c++ to be proper fast.

Jax doesn't like modifying arrays in place.

- Experiment with WSAD, QE and arrow keys. 
- Escape to quit.
- Change `__main__` to save data.
One eye image looks more or less like this:

![render](imgs/triangles.gif)

Here's an older toy render of a cube.

![render](imgs/render.gif)
