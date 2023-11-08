# SimplePlanner

Simple 2D implement of the paper "Planning Dynamically Feasible Trajectories for Quadrotors Using Safe Flight Corridors in 3-D Complex Environments".

![Figure_1](https://github.com/flztiii/SimplePlanner/assets/20518317/2f69f54d-410c-457e-85da-5737f74196cd)

## Info

This repo only aims to help to better understand the paper "Planning Dynamically Feasible Trajectories for Quadrotors Using Safe Flight Corridors in 3-D Complex Environments".

## Modules

This repo include three modules:

jps.py is the implement of the Jump Point Search (JPS) algorithm.

convex_decomp.py is the implement of the convex decompose according to a given path.

trajectory_optmization.py is the implement of the trajectory optimization. The trajectory is a piece-wise bezier curve and the optimization solver is osqp.
