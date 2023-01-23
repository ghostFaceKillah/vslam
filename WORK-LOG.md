2023-01-22
----------

- [ ] Less triangles, more evenly spaced, better color choices
  - birdseye renderer for debugging this
  - make movie maker script to not have to click through it by hand
  
- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general
  
- [ ] Record a datastructure of the rendering to use as data for VSLAM
  - choose: msgpack, capnproto, protobuf, sth else?
  
- [ ] add unit tests, mypy, black and so on on github
  
Later
- [ ] Render per triangle refactor
  - [ ] add unit tests for rendering
  
- [ ] lacma inspired rainbowy lines

Which serialization format?
- msgpack is easy to integrate with dicts
- msgpack would rely on custom object deserialization
- protobuf is relatively heavy dependency
- I wonder how fast protobuf deserialization is ? msgpack deserialized things 
  might live as typed dicts for now, but in the long run, they tend to need serious deserialization
  

My own triangle rendering situation
-----------------------------------------

I made my own triangle index finder situation. It takes 0.0045s for one triangle

```python
with just_time('inside triangle computation'):
    bary = compute_barycentric_coordinates_of_pixels(triangles_in_img_coords, px_center_coords_in_img_coords)
    inside_triangle_pixel_filter = np.all(bary > 0, axis=-1)
```
Takes 0.04369s for 12 triangles
Takes 0.4792s for 100 triangles



2023-01-16
----------
- [ ] Less triangles, more evenly spaced, better color choices
- [ ] Record a video of the rendering to use as data for VSLAM
    msgpack, capnproto, protobuf
- [ ] add unit tests, mypy, black and so on on github


2023-01-15
----------

TODO:

- [X] Spawn the scene!
  - make it look pretty!
    - [X] add a background,
    - [X] make it brighter
    - [ ] Hide triangles under the ground
- [X] Refactor render to make it look nice
- [X] make some basic README to make it already presentable
- [X] There's still a bug in lighting - sometimes triangle flips between yellow and red
  Removed lighting :)
- [ ] add unit tests, mypy, black and so on on github

* How to figure out if given pixel is "ground" or "sky"?

We have its coordinates in camera.
We need to treat it as a ray going from optical center towards image plane.
We need to compute this rays coordinates in world frame.

For the orientation coordinates in world frame, if the pitch is zero or higher, it's sky,
otherwise it's ground.


2023-01-14
----------

TODO:
- [ ] Refactor render to make it look nic**e
- Spawn the scene!
  - make it look pretty! (add a background, make it brighter, etc.)
- [X] make some basic README to make it already presentable
- add unit tests, mypy, black and so on



2023-01-09
----------

Need to do triangle visibility clipping

Need to compute clipping volume.


Need to compute coordinates of edges of the screen in camera coords.

a--e--b
|     |
g     h
|     |
c--f--d


then we can take {e,f,g,h} - origin and take vectors orthogonal to them


Then we can take vectors orthogonal sides of the screen 



2023-01-07
-----------

- refactor render to make it clear

- features
  - make a simulated room
  
- bugs
  - tune camera intrinsics to get rid of weird scaling
  - bug in rendering - uneccessary / incorrect transform of light to camera frame ?
  - visibility bug (we render things which are behind us :o)
  
- set up proper CICD

did a small sanity check to see if we will even have matching keypoints in a simulated environment.


2023-01-05
-----------
What should the next steps be?

Our ultimate goal is to have mobile phone on a stick robot, that navigates around arbitrary indoor environment.
Why? 
- to start a company selling cheap robots
- if starting company too ambitious or doesn't work, at least I would have a nice blog post out of it :))
- learn VSLAM 
- practice C++ 
- learn jax <--

What concrete tickets could we do to hit this goal?

- start building the robot - huge
  - pick two cameras (or one), pick compute module, pick the wheel base and try to get it to actually move
    - can we do all of this on a mobile phone ?
    - all this para-hardware work is streamable on Twitch using go pro
    
    - Roomba as a base
      - the vacuum can be bought from craiglist for around 100 bucks
      - most roomba's (notably 600/700/800/900 series) have debug ports that accept external messages
        with some hacking, there's hope of being able to control the base
        https://www.irobot.lv/uploaded_files/File/iRobot_Roomba_500_Open_Interface_Spec.pdf
      Also, just google "roomba open interface"
        
    - IRobot create 3 as a base
      - it's only 300 USD new, and it looks like it might save a lot of time
      - but it uses ROS :(
      100 USD + / - per hour
        
    - compute platform
      - qualcomm
      - Nvidia Xavier, etc
  
    - cameras for VSLAM
      - super high quality leopard cameras (likely huge overkill)
      - one or two mobile phones - a lot on ebay etc, but hard to access raw data stream, 
        pay a lot for sensors you don't need
      - microsoft kinnect
      - intel realsense
      - qualcomm prebuilt robotics kit
  
  - continue building naive VSLAM on top of simulated dataset
    - speed up the rendering in jax - 2-4 weeks
      - disadvantage: adds big dependency to otherwise simple to understand and use framework
    - don't speed up rendering and just live with current 0.4 s renders or incorrect occlusion rendering
    
  - clean up Orb SLAM V3, keep dataset performance defended by unit tests / provided benchmarks 
  and refactor it to be nice
    
    + doing the angel's work: taking best VSLAM method and making it usable in real life
    - huge time investment and likely not directly related to building the real life robot.
      Likely it will contain high amount of stuff not really needed for our usecase.
      
  In summary, as of today, it's probably a bit suboptimal, but I feel like doing jax.
  Specifically, speeding up the rendering thing in jax.


2023-01-02
----------
I have figured out a nice rendering methods that handles overlaps, etc.
It looks a bit slow: 640x480, 12 triangles renders in 0.37 seconds.
There's a couple of different possibilities of rendering it, many would work.
I think I will try to do it based on jax.

2022-12-05
----------
Cool! I have first draft of stupid rendering :)

2022-11-30
----------
To properly debug bundle adjustment, I feel we need to have our own simulated environment rendering.

- don't make it a rabbithole
- don't need tight rendering and keyboard input interaction
- avoid adding new dependencies

conclusion: try realtime rendering with opencv.

2022-11-29
----------
- tried SBGM based depth estimation (see 4.4.1 Stereo Vision)
  it was brittle, not understandable to me, dropping it for now
  
- probably this is way too overkill, but I think to understand the frontend stuff well I need
  to have a synthetic dataset with rendering, etc, and assert 100% convergence

todo:
- core:
    - implement frontend
- support
    - even more docs around types - would beginner be able to read them ?
    - unit tests and mypy
    - dep inject kitti fpath
    - kitti fix calibration
    - add unit tests around transforms

2022-11-27
----------

done:
  - refactor feature extraction
  - depth estimation and show
  
todo:
  - support
    - even more docs around types
    - unit tests and mypy
    - dep inject kitti fpath
    - kitti fix calibration
  - core:
    - implement frontend
  - ui

2022-11-25
----------

todo: 
- refactor feature extraction
- depth estimation and show
- ui
- unit tests etc

2022-11-24
----------
Added nice debugging tool for evaluating matches.
- [ ] Add UI: general displaying of information
- [X] Need to add text rendering and display match details

Actually turns out that filtering by low hamming distance was pretty good

2022-11-08
----------

What 'Intro to VSLAM' book did:

1) take GFTT feature extractor and extract features
2) use LK optical flow to figure out feature matching between left eye and right eye
3) based on this matching, do triangulation and initialize first baseline map from it

I guess they don't want to burn too much compute on the feature matching ...
I wonder:
- how much better ORB features are for tracking
- how much faster is GFTT vs ORB+ brute force match +RANSAC 




2022-11-07
==========

Probably need a nice camera library.

Would be super nice to have cool visualizations of the camera matrix,
projections, etc.

Maybe just in pangolin, right ?


2022-11-06
==========

Intro
-----

We are going to build VSLAM system for
"mobile phone on a stick" robot.

The robot will navigate from vision !
No LIDARs (!!!).
Why ?
1) Because I already know how to build navigation from Lidars, that's what I have at work.

2) Also, I a curious about how to properly engineer around system that has

- python
- cpp 
- cuda
- neural nets, not neccesarily cuda


Software concrete plan
----------------------

- prep work:
  - environment
  - array typing
- load kitti
- find keypoints
- do optical flow & visualize it :)


Software considerations
-----------------------

- probably start from kitti dataset
- can also take comma's challange

- go slow with deep understanding

  - stress pretty implementation & proper code


- goal:
  - is to make nice artifact for internet
  - prepare to build navigate-from-vision robot


Hardware research questions
---------------------------

- What compute:

  - snapdragon - price?
  - NVIDIA Xavier

- what camera
  - comma ai
      - AR0231 Automotive Camera (x3)
      - 1080p Resolution
      - 3x3µm size/pixel (“large” pixel)
      - 30 fps
      - x2 185° FOV Lens (driver cam & front wide cam)

      ~400 USD devkit / retail, probably overkill for us - too good / expensive
      it will go in volume for like 100 ? so maybe not overkill


    https://enjoi.dev/posts/2021-12-20-comma-3-camera-views/


- cheap lidar to bootstrap navigation? for example, to provide training
  data for occupancy / drivable surface esimation networks.

  - roborock lidar
  - SLAMTEC cheap lidar 100-500 USD - risy
  - 3d lidar 3.5k


- how widely mounted do the cameras need to be?
 
  1) from Kitti we know that 60 cm is enough
  2) comma has very narrowly spaced, but this is because
     they have access to radar info (so no need to estimate depth well, I guess)

