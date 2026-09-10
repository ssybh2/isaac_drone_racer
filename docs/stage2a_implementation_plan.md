# Stage2A Perfect Corner Projection + PnP

## Objective
Replace oracle gate relative position with a perception-derived estimate.

## Coordinate convention

Gate frame G:

- Origin: gate opening center
- X: right direction through gate
- Y: upward direction
- Z: gate normal

Corner order:

0 left-bottom
1 right-bottom
2 right-top
3 left-top

## Pipeline

```
Gate USD pose
    |
3D corners in world
    |
Camera projection
    |
Perfect 2D corners
    |
Planar PnP
    |
T_camera_gate
    |
Camera-to-body transform
    |
Estimated target_pos_b
```

## Future work

- Extract exact dimensions from assets/gate/gate.usd
- Add oracle-vs-PnP equivalence tests
- Add Isaac Lab TiledCamera dataset generation
- Add Stage2B corner noise/dropout simulation
