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

