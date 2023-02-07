"""
We are going to learn go PnP.
PnP = perspective and point.

We are going to take 3 pictures
- left eye, right eye
- next frame from left eye

we are going to match 3d points and 2d points
by minimizing reprojection error using manual Gauss-Netwon implementation.

see this https://github.com/gaoxiang12/slambook2/blob/master/ch7/pose_estimation_3d2d.cpp#L172
bundleAdjustmentGaussNewton

6.8.2. Pose Estimation from Scratch

"""