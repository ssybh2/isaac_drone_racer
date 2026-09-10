# Stage 2A Perfect Corner Projection

## Goal

Remove oracle gate position observation.
Replace:

`world gate pose -> target_pos_b`

with:

`gate 3D corners -> camera projection -> 2D corners -> PnP -> gate pose`

## Gate Frame

Standard frame G:

- origin: gate opening center
- X: horizontal right
- Y: vertical up
- Z: gate normal

Corner ordering:

```
0: left-bottom
1: right-bottom
2: right-top
3: left-top
```

## Pipeline

```
Isaac truth pose
      |
3D gate corners
      |
Camera projection
      |
PerfectGateCornerSensor
      |
solvePnP(IPPE)
      |
Gate pose in camera frame
      |
Camera-body extrinsic
      |
Estimated target_pos_b
```

## Next steps

Stage2B will add:

- pixel noise
- latency
- missing corners
- partial visibility

Stage3A will enable IsaacLab tiled camera rendering and automatically generate RGB + keypoint labels.
