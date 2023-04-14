2023-04-13
----------

- [ ] fix frontend bugs: why is it going wrong on the big dataset?
- [ ] make vis pretty:
  - [ ] draw matches
  - [ ] global scene display
    - show more global context:
      - all of ground truth path
      - current pose estimate
- [ ] Test, type, lint, refactor etc
- [ ] Lectures & blogpost


2023-04-11
----------

- [ ] fix frontend bugs: why is it going wrong on the big dataset?
- [ ] make vis pretty:
  - [ ] draw matches
  - [X] Align sizes of images
  - [X] Add image labels
  - [X] Add more data to simulation
  - [ ] Clean up the scene display
    - don't show all keyframes, just the recent one
    - show more global context: 
      - all of ground truth path
      - current pose estimate
- [ ] Test, type lint, etc
- [ ] Lectures & blogpost

2023-04-10
----------

- [X] keyframe switching
- [X] More advanced visualizations
  - [X] off-white background point n stick plots of tracking
  - [ ] cumulative tracking error (rabbithole warning)
  - [X] keyframe image vs current image
  - [X] birdseyeview of keyframe
- [ ] Test, type lint, etc
- [ ] Lectures & blogpost


2023-04-07
----------

Looks great! Error looks ~0. Let me draw the path of the system from birds eye view.

- [X] Birdseye view of the path
- [ ] keyframe switching
- [ ] Test, type lint, etc
- [ ] Lectures & blogpost


2023-04-04
----------

Let me debug feature matching / pose estimation on second frame.
- feature matching looks great

Then, let me plot details of pnp.


2023-04-02
----------

When I cast a ray from optical center of left eye to (visual) apex of triangle, I see that the lines don't overlap.

Error can come from may different sources.
- [X] mapping from image coords to real coords
- [no] birdeye view rendering
  
As a way to cut through noise, let us draw lines to triangle ends

birdeye view rendering looks perfect

looks like conversion from camera px coords to world coords is next obvious candidate on the list
we can do the classic idempotency check.

- [X] test to_cam(from_cam(world_coords)) == world_coords

the error was difference in beginning pose assumption

- [ ] add beginning pose assumption to serialized simulation recording
- [.] debug the actual depth finding, we now see that it might have a bug
- [ ] rewrite equations from piece of paper to ipad
- [ ] then solve it symbolically and then only use the solution in xy plane, it looks so good, I am surprised 
      that it's not solved already



2023-04-01
----------
Nice, I wrote the keyframe making debug and sth is clearly off.
The projection lines don't intersect where they should.

Maybe cam intrinsics are off ? 
I don't get it ... yet!


2023-03-31
----------
I see a bunch of interesting ways to debug

1) keyframe making debug
   a) draw birds eye view of depth estimation, etc
2) pose tracking debugging

After making these debuggers, I should make blogpost finally.


2023-03-30
----------

I had really a lot of changes in my life! That was pretty crazy.
Let me try this back again.



2023-02-28
-----------
Ok, it seems to work!

Now I need to plot it on some kind of overhead plotting

- [ ] For the sake of good quality optimization, simulate the feature matches
- [ ] Visual Debugger for past poses 
    - Birdseye view of past poses, as figured out by optimization
- [ ] Debugger for Gaussian optimization
    - reprojection errors, etc visualization
- [ ] parametrize x y theta Gauss Newton instead of blindly overriding it
  - Charles has mentioned something called conjugate gradient method

- [ ] Implement the actual VSLAM
  - [.] Implement frontend
      - [ ] do tracking inliers etc
  - [ ] Implement backend (remember, don't overcomplicate!)
  - [X] g2o TrackPose reimplementation

- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general

- [ ] code quality retrieval
  - [ ] add unit tests, mypy, black and so on on github
  - [ ] Clean up the codez
  - [ ] Expose `to_native` and `from_native`

- [N] sqlite based recorder and streaming replayer to be able to record "big data" ?
- [ ] lacma inspired rainbowy lines as features




2023-02-26
----------

Further debugging!
depths look incorrect - they are far too far.
Let us debug this.
SOLVED!
the coordinates were flipped again :)

 Moved to simpler scene.
A lot of bad matches!
Hamming distance / feature descriptors seem to be not great.
They tend to get a lot of false matches along the diagonal of the square,
which is the longer edge of the triangle.


Now the pose is better, but still off. Hardcore debug continues.

```
> # our current estimate
> (keyframe.pose @ new_pose_estimate).round(2)
array([[ 1.  , -0.  , -0.01, -2.19],
       [ 0.  ,  1.  , -0.02, -0.95],
       [ 0.01,  0.02,  1.  ,  0.1 ],
       [ 0.  ,  0.  ,  0.  ,  1.  ]])

> # ground truth
> obs.camera_pose   
array([[ 1. ,  0. ,  0. , -2.4],
       [ 0. ,  1. ,  0. ,  0. ],
       [ 0. ,  0. ,  1. ,  0. ],
       [ 0. ,  0. ,  0. ,  1. ]])
```

Ideas:

- Fake feature matches! - you have access to ground truth data, just do ground truth matches
- Another debugging idea - compare against ground truth depth.
- Another debugging idea tool - reprojection error visualizer.

Thing to make sure about: 
- depth estimates from naive triangulation: are those in cam one or cam two ?

2023-02-24
----------

I am doing end-to-end frontend for our data.
The first pose is super-diverged, it must be a bug.
As always, careful debug of each step is required.

If we run `overfit_one_point` with keyframe pose as target, we get division by zero depth
type of situation. We should become resilient to those.



2023-02-20
----------

I have solved all the problems in the reprojection error minimization.
It all works well now!
I am now working on the frontend.
I think I know how to do backend as well. The sparse g2o part is a bit annoying to do, but in the end doable.

- [ ] Implement the actual VSLAM
  - [ ] Implement frontend
  - [ ] Implement backend (remember, don't overcomplicate!)
  - [X] g2o TrackPose reimplementation

- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general

- [ ] code quality retrieval
  - [ ] add unit tests, mypy, black and so on on github
  - [ ] Clean up the codez
  - [ ] Expose `to_native` and `from_native`

- [N] sqlite based recorder and streaming replayer to be able to record "big data" ?
- [ ] lacma inspired rainbowy lines


2023-02-14
----------
I am "implementing g2o" in the sense of minimizing the reprojection error and it's going well actually.
I have some errors in the reprojection error minimization.
These are definitely worth digging into. Cool stuff


2023-02-06
----------

- [ ] Implement the actual VSLAM
    - [X] Understand their code again
    - [ ] g2o TrackPose reimplementation
      - [ ] do I reimplement g2o, lol ?
    
- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general

- [ ] code quality retrieval
  - [ ] add unit tests, mypy, black and so on on github
  - [ ] Clean up the codez
  
- [ ] sqlite based recorder and streaming replayer to be able to record "big data" ?

- [ ] lacma inspired rainbowy lines
  


2023-02-04
----------

Started implementing VSLAM.

Feature matching works perfect.

Triangulation seems to be working OK - the depths returned look good, but many times, 
we get NaNs and failed triangulation. Would be nice to understand why exactly.

Update: Understood failed triangulation well based on equations.

- Scale estimation per dimensions and then averaging can be interpreted as L2 MSE estimation.
- If scale estimations based on dimensions are too off it means that it's a bad match,
  so it's a very nice secondary filter.
- Hardcoding dropping first dimensions is a bad idea, because that assumes that points are coplanar in horizontal 
  dimension in frame of the reference image (usually left)
- Excluding s = - b / a  where b or a are close to zero makes sense, because that would introduce very noisy estimates.

Triangulation outputs look very good and at the same time are really brittle wrt to bad features matches.
Brittle in the sense that for far off points it gives high distance errors.
This kinda replicates the classic results, so I am not too worried.


- [ ] Implement the actual VSLAM
    - [ ] Understand their code again
    - [ ] Will need g2o (or reimplementation, lol)
    
- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general

- [ ] code quality retrieval
  - [ ] add unit tests, mypy, black and so on on github
  - [ ] Clean up the codez
  
- [ ] sqlite based recorder and streaming replayer to be able to record "big data" ?

- [ ] lacma inspired rainbowy lines
  
- [X] Record datastruct


2023-02-02
----------

Need unit tests for stuff not to start falling apart soon
I have solved serialization :)  Using cattrs, so that's awesome.

Likely worth introducing sqlite based recorder to be able to record "big data" ?

- [ ] Record datastruct
- [X] make movie maker script to not have to click through it by hand
  (Doing this, 50% done :) )

- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general

- [ ] code quality retrieval
  - [ ] add unit tests, mypy, black and so on on github
  - [ ] Clean up the codez

- [X] Record a datastructure of the rendering to use as data for VSLAM
  - choose: msgpack, capnproto, protobuf, sth else?

- [ ] lacma inspired rainbowy lines


2023-02-01
----------

We have added simulation driver - and wrapped the manual interaction loop in ManualActor.
We will write prerecorded motions actor and record what it sees to msgpack.
Then we will learn how to VSLAM on top of it!

- [.] make movie maker script to not have to click through it by hand
  (Doing this, 50% done :) )

- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general

- [ ] code quality retrieval
  - [ ] add unit tests, mypy, black and so on on github
  - [ ] Clean up the codez

- [ ] Record a datastructure of the rendering to use as data for VSLAM
  - choose: msgpack, capnproto, protobuf, sth else?

- [ ] lacma inspired rainbowy lines


2023-01-30
----------

What do we want to save?

- image left, image right
- pose
- fake timestamp in ns



2023-01-26
-----------

- [ ] make movie maker script to not have to click through it by hand
  
- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general

- [ ] code quality retrieval
  - [ ] add unit tests, mypy, black and so on on github
  - [ ] Clean up the codez
  
- [ ] Record a datastructure of the rendering to use as data for VSLAM
  - choose: msgpack, capnproto, protobuf, sth else?

- [ ] lacma inspired rainbowy lines

- [X] Less triangles, more evenly spaced, better color choices
- [X] birdseye renderer for debugging this

2023-01-22
----------
  
- [X] Less triangles, more evenly spaced, better color choices
  - [X] birdseye renderer for debugging this
  - [ ] make movie maker script to not have to click through it by hand
  
- [ ] Make a tutorial video about rendering & geometry of rendering
- [ ] Make a blogpost about this thing in general
  
- [ ] Record a datastructure of the rendering to use as data for VSLAM
  - choose: msgpack, capnproto, protobuf, sth else?
  
- [ ] add unit tests, mypy, black and so on on github
  
Later
- [ ] Render per triangle refactor
  - [ ] add unit tests for rendering
    this has failed :(
  
- [ ] lacma inspired rainbowy lines

Which serialization format?
- msgpack is easy to integrate with dicts
- msgpack would rely on custom object deserialization
- protobuf is relatively heavy dependency
- I wonder how fast protobuf deserialization is ? msgpack deserialized things 
  might live as typed dicts for now, but in the long run, they tend to need serious deserialization
  

My own triangle rendering situation
-----------------------------------------

I needed simple (depth ignoring) rendering of triangles.
I had vague hopes of speeding up the main 3d rendering loop by getting rid of the barycentric computation
for each pixel for each triangle.
I made my own triangle index finder situation.

It has super failed, it seems! It takes 0.0045s for one triangle, which is 2 orders of magnitude slower then cv2.

Barycentric computation:
```python
with just_time('inside triangle computation'):
    bary = compute_barycentric_coordinates_of_pixels(triangles_in_img_coords, px_center_coords_in_img_coords)
    inside_triangle_pixel_filter = np.all(bary > 0, axis=-1)
```
Takes 0.04369s for 12 triangles
Takes 0.4792s for 100 triangles

In summary, cv2 has super destroyed my 'smart' jax implementation, by two orders of magnitude.

count  100.000000  100.000000
mean     0.007035    0.000054
std      0.024462    0.000037


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

