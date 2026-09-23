# 2026-09-23 Progress Summary — Color-Coded Gate Identity, Diversified Vision Retraining, Gate-ID Map Association, and Slow-Coordinated GT Policy Plan

## 1. End goal and today's change in priority

The long-term objective is still a no-GT autonomous drone-racing stack:

    onboard IMU
        +
    learned inertial / IMU propagation
        +
    SC-EKF
        +
    RGB gate perception
        +
    known global gate map
        ↓
    estimated position / velocity / attitude
        ↓
    frozen CTBR racing policy
        ↓
    closed-loop autonomous racing

Today's work significantly advanced the visual landmark side, but it also confirmed that the current Circular-12 GT racing policy is still not a valid trajectory generator for final estimator validation.

The key end-of-day conclusion is:

> The Color20 visual pipeline is now much stronger, but estimator validation must remain paused until the wind-tumble behavior of the GT racing policy is removed.

The next immediate milestone is therefore not another vision experiment.

It is:

> Train a new Circular12-SlowCoordinated-v0 GT policy that produces stable, coordinated circular flight close to the prescribed 20-degree-camera reference experiment.

---

## 2. Starting problem: Circular-12 geometric ambiguity

The Circular-12 track is highly regular:

- radius approximately 12 m;
- 12 evenly spaced gates;
- tangent-aligned gate orientations;
- repeated local geometry.

This created a serious perception problem.

Even with good corner detection, a camera image of one gate could look geometrically very similar to another gate.

That means a detector could produce accurate 2-D corners while the estimator still had ambiguity over which global map landmark was being observed.

The original all-map direct-reprojection association could resolve some cases from the current EKF state, but this becomes weak when inertial drift grows.

The design decision made today was:

> Give every physical gate a stable visual identity using a unique high-contrast texture color.

---

## 3. Twelve real physical gate textures were created and bound

The original source gate texture is:

    assets/gate/textures/bitmap.png

Twelve generated gate textures are now stored under:

    assets/gate/textures/circular12/

The final high-contrast identity map is:

    Gate 01  red
    Gate 02  spring_green
    Gate 03  magenta
    Gate 04  chartreuse
    Gate 05  blue
    Gate 06  orange
    Gate 07  cyan
    Gate 08  rose
    Gate 09  green
    Gate 10  violet
    Gate 11  yellow
    Gate 12  azure

The palette was deliberately permuted around the circular track.

Every adjacent pair, including Gate 12 -> Gate 1, has approximately 150 degrees of HSV hue separation.

This was chosen so consecutive gates are visually very different even under small image size and motion blur.

The recoloring preserves:

- black / white checkerboard regions;
- source texture shading;
- alpha;
- local brightness variation.

Only the original blue-painted region is recolored, with a high saturation floor.

---

## 4. Isaac Sim USD binding issues and final solution

The first texture-binding implementation exposed two USD problems.

### 4.1 First failure: USD/Sdf layer cache

The first generated Gate 01 USD worked at the data level, but subsequent gates failed because the source gate USD remained modified in the process-wide USD layer registry.

This was initially addressed by reloading the source layer before each variant.

### 4.2 Second failure: all gates rendered white

The first full-stage export strategy produced USD files whose texture paths looked correct in inspection but whose material composition did not render correctly in Isaac Sim.

The visible symptom was:

> All 12 gates appeared white in Isaac Sim.

The design was then changed to a reference-wrapper architecture.

Each generated variant now:

- references the authoritative original gate.usd;
- preserves original geometry, collision, UVs, rigid-body properties, and MDL material;
- authors only a stronger texture-path override.

The user then updated the wrapper implementation to use absolute asset paths, which solved Isaac Sim material / texture resolution.

The real physical gates now visibly render with their distinct textures.

Generated runtime USD variants are local build products and remain ignored by Git:

    assets/gate/gate_circular12_*.usd

---

## 5. Color identity became a formal Gate ID contract

A machine-readable source of truth now exists:

    assets/gate/textures/circular12_gate_texture_map.json

A runtime identity helper was added:

    perception/gate_identity.py

The core contract is:

    visual Gate ID 1..12
        <->
    fixed physical gate index
        <->
    known global map pose T_wg

This gives each gate both:

- geometric identity from the global track map;
- visual identity from its unique texture.

---

## 6. Camera / trajectory reference was standardized

The earlier prescribed steady-turn experiment was retained as the visual reference.

Important properties:

    camera pitch up        20 deg
    body height            2.07 m
    track radius           12 m
    prescribed circular trajectory
    coordinated bank
    stable body-fixed FPV

Important limitation:

> This experiment is not a controller and not evidence of closed-loop policy stability.

The script directly writes the desired pose and velocity every frame.

It is therefore only a reference for the flight appearance and camera geometry that the learned GT policy should eventually approximate.

The prescribed experiment remains useful for:

- FPV quality reference;
- camera-angle reference;
- clean vision dataset generation;
- validating Gate ID / corner labels.

---

## 7. Formal diversified vision dataset was collected

A dedicated formal collector was added:

    scripts/perception/collect_circular12_diversified_vision_dataset.py

Dataset root:

    artifacts/racing_vision/circular12_color20_h207_diversified_v1

Capture contract:

    camera pitch       20 deg
    body height        2.07 m
    image size         256 x 256
    capture rate       25 Hz

Diversity:

    speeds:
        14.0
        15.5
        17.0
        18.5
        20.0
        21.5 m/s

    path radii:
        11.6
        12.0
        12.4 m

    phase offsets:
        0.0
        7.5
        15.0 deg

Total runs:

    6 x 3 x 3 = 54

The train / validation / test split is grouped by whole speed-radius conditions so phase-shifted versions of the same condition do not leak across splits.

Final recovered manifest:

    train
        runs      36
        samples   3960

    val
        runs       9
        samples    942

    test
        runs       9
        samples   1038

    total
        runs      54
        samples 5940

The manifest confirms:

    camera_pitch_up_deg = 20.0
    body_height_m       = 2.07

A shutdown-order bug originally prevented manifest writing even though all RGB images and labels had been collected.

A recovery script was added:

    scripts/perception/finalize_circular12_diversified_vision_dataset.py

It reconstructs manifests from the already collected labels without rerunning Isaac Sim.

The collector itself was also fixed so manifests are written before SimulationApp closes.

---

## 8. New visual network now predicts Gate ID and corners together

The multi-gate Keypoint R-CNN was changed from a class-agnostic gate detector to a 13-class detector:

    class 0   background
    class 1   Gate 01
    ...
    class 12  Gate 12

Training target:

    RGB
        ↓
    gate instance
        +
    Gate ID 1..12
        +
    4 semantic gate corners

CornerObservation now carries:

    gate_id
    gate_id_confidence

The runtime detector checkpoint metadata records:

    gate_identity_mode = gate_id_class
    num_classes        = 13

The old class-agnostic checkpoints remain backward compatible because they simply leave gate_id unset.

---

## 9. Visual retraining completed

Training output:

    artifacts/racing_vision/
    circular12_color20_h207_gateid_kprcnn_v1/

Best checkpoint:

    torchvision_keypointrcnn_multigate_best.pt

Best epoch:

    40

Validation:

    gt instances                        2548
    predicted usable instances          4070
    matched instances                   2385
    instance precision                  0.5860
    instance recall                     0.9360
    positive-frame hit rate             1.0000
    negative-frame false positive rate  0.0000
    corner coordinate RMSE              1.299 px
    gate-ID accuracy                    0.6851

Independent test:

    gt instances                        2779
    predicted usable instances          4166
    matched instances                   2447
    instance precision                  0.5874
    instance recall                     0.8805
    positive-frame hit rate             1.0000
    negative-frame false positive rate  0.0000
    corner coordinate RMSE              1.215 px
    corner radial RMSE                  1.718 px
    gate-ID accuracy                    0.6412

Interpretation:

- gate detection recall is strong;
- every positive test frame has at least one successful gate match;
- corner localization is very strong at approximately 1.2 px;
- Gate ID classification is only moderate at approximately 64 percent;
- precision is still modest because of extra / duplicate usable detections.

The strongest part of the new visual model is therefore the geometry, not the identity classifier.

---

## 10. Gate-ID confusion matrix exposed the identity failure structure

A dedicated evaluator was added:

    scripts/perception/evaluate_gate_id_confusion.py

It matches predictions to GT using only corner geometry first, then evaluates Gate ID.

This separates:

    "did the network find the correct physical gate geometry?"

from:

    "what Gate ID did the network call it?"

Test-set per-gate identity accuracy:

    G01 red            82.8%
    G02 spring_green   45.0%
    G03 magenta        60.7%
    G04 chartreuse     57.4%
    G05 blue           73.5%
    G06 orange         63.0%
    G07 cyan           66.7%
    G08 rose           56.7%
    G09 green          53.0%
    G10 violet         63.2%
    G11 yellow         87.4%
    G12 azure          59.7%

Major confusion pairs include:

    G02 spring_green -> G09 green
    G10 violet       -> G12 azure
    G12 azure        -> G10 violet
    G08 rose         -> G07 cyan
    G09 green        -> G11 yellow
    G03 magenta      -> G10 violet
    G04 chartreuse   -> G09 green

This proves that the ROI classifier is not acting as a clean hue reader.

It is also learning correlations from:

- pose;
- screen position;
- gate scale;
- background;
- checkerboard appearance;
- motion blur.

Therefore Gate ID must not be used as a hard global landmark assignment.

---

## 11. Gate ID is now a bounded prior, not the geometric authority

The direct-reprojection association was changed accordingly.

Old Color-ID behavior:

    high-confidence Gate ID
        ↓
    only test that one global Gate
        ↓
    reject the visual observation if it fails

This would throw away many observations because Gate-ID accuracy is only about 64 percent.

The new behavior is:

    detector corners + Gate ID
        ↓
    evaluate Gate-ID candidate
        +
    evaluate all 12 map-gate reprojection candidates
        ↓
    use Gate ID only if:
        confidence >= 0.50
        identity candidate RMSE <= 15 px
        identity candidate is within 4 px of global best
        ↓
    otherwise:
        all-map geometric fallback

Current parameters:

    gate_identity_min_confidence             = 0.50
    gate_identity_preferred_max_rmse_px      = 15.0
    gate_identity_preference_margin_px       = 4.0

Diagnostic association modes:

    gate_id_preferred
    all_map_fallback
    all_map

The principle is now:

> Color identity is a prior that helps break symmetry. Pixel geometry remains the final consistency check.

This matches the actual strengths of the trained model:

    Gate ID       moderate
    corners       strong

---

## 12. GT-shadow Color20 integration task exists, but is intentionally deferred

A dedicated task was added:

    Isaac-Drone-Racer-Learned-Inertial-Swift-CTBR-
    Circular12-KnownStart-GTShadow-Color20VisionV1-v0

It uses:

    camera                       20 deg
    new Color20 Gate-ID model
    direct reprojection
    checkpoint pixel sigma
    bounded Gate-ID prior
    all-map fallback
    learned-motion fusion OFF
    GT actor control ON

However, after reviewing the current state of the GT policy, we decided:

> Do not use this task yet as the main visual qualification test.

Reason:

The original 12GT policy still produces wind-tumble behavior.

Even in GT-shadow mode, that policy generates the camera trajectory.

So a poor visual result under that policy would mix together:

- detector performance;
- extreme body rotation;
- motion blur;
- camera horizon rotation;
- policy pathology.

The visual model should first be evaluated under a physically reasonable free-flight policy.

---

## 13. The central unresolved control problem: wind-tumble is still present

The original high-speed Circular-12 GT policy races very well, but it learned a pathological rolling / tumbling behavior.

Original characteristics:

    mean speed              ~17.8 m/s
    p95 speed               ~19.5 m/s
    mean body-rate norm      ~6.27 rad/s
    p95 body-rate norm       ~7.39 rad/s

The original CTBR authority was:

    p / q / r max = 10 / 10 / 6 rad/s

The original angular-velocity penalty was tiny:

    weight = -0.0001

Stable-v1 was already trained to reduce this:

    p / q / r max = 6 / 6 / 3 rad/s
    ang_vel_l2     = -0.02
    stronger look-at
    rate-command penalty
    command-smoothness penalty

Stable-v1 successfully reduced body-rate magnitude:

    old mean body-rate      ~6.27 rad/s
    Stable-v1 mean          ~3.42 rad/s

but manual FPV still shows:

> Long-period wind-tumble behavior remains.

This is no longer just an "angular speed too large" problem.

It is an orientation / roll-phase problem.

---

## 14. Why the previous anti-spin reward could not fully solve it

The current reward components mainly constrain:

- body-rate magnitude;
- commanded body-rate magnitude;
- command smoothness;
- body +X direction toward the gate or velocity.

The important structural loophole is:

> A roll about body +X leaves body +X almost unchanged.

Therefore a policy can:

- continue progressing around the track;
- keep its nose approximately forward;
- continue passing gates;
- yet slowly rotate the entire camera horizon through a full roll cycle.

That is exactly the long-period wind-tumble still seen in Stable-v1 FPV.

This means simply increasing:

    ang_vel_l2

again is not the correct long-term solution.

It would also penalize angular velocity that is physically required for circular flight.

---

## 15. Prescribed 20-degree-camera experiment defines the new policy target

The prescribed circular experiment provides a much better qualitative target.

Its flight is:

- circular;
- coordinated;
- banked;
- stable;
- camera horizon predictable;
- no full roll cycles.

The next learned GT policy should approximate this free-flight behavior.

It does not need to exactly track the prescribed pose frame-by-frame.

It does need to satisfy the same physical structure:

    body forward axis
        -> local track tangent

    body thrust axis
        -> gravity + centripetal acceleration direction

    no free roll phase

This is the core idea behind the new SlowCoordinated policy.

---

## 16. Next policy: Circular12-SlowCoordinated-v0

The agreed next task is:

    Circular12-SlowCoordinated-v0

The purpose is:

> Produce a genuinely free-flight GT policy whose trajectory is stable enough to qualify the Color20 visual detector and later the estimator.

The target speed is intentionally reduced.

First target:

    approximately 12-14 m/s

Suggested soft upper limit:

    14 m/s

At radius 12 m, the ideal coordinated bank is approximately:

    18-19 m/s -> about 70 deg
    14 m/s    -> about 59 deg
    12 m/s    -> about 51 deg
    10 m/s    -> about 40 deg

So moving from roughly 19 m/s toward 12-14 m/s gives the camera and visual estimator a much more reasonable motion regime while still remaining a fast racing trajectory.

---

## 17. SlowCoordinated action authority

The proposed CTBR rate limit is:

    original        (10, 10, 6) rad/s
    Stable-v1       ( 6,  6, 3) rad/s
    SlowCoordinated ( 4,  4, 2) rad/s

This still leaves large margin over the approximately required coordinated-turn angular motion.

For example, at:

    speed  = 14 m/s
    radius = 12 m

the track heading rate is approximately:

    omega = v / r
          = 1.17 rad/s

So a 2 rad/s yaw-rate authority remains sufficient for the intended motion while reducing the policy's ability to exploit very aggressive tumbling.

---

## 18. Coordinated-turn attitude reference

This is the most important reward change.

Instead of penalizing generic tilt or requiring the vehicle to stay level, construct an ideal coordinated-turn frame.

Desired forward direction:

    x_des = normalized horizontal velocity / track tangent

Desired inward direction:

    n_in = normalized(track_center_xy - position_xy)

Desired thrust axis:

    z_des = normalize(
        gravity * world_up
        +
        v_horizontal^2 / radius * n_in
    )

Then:

    y_des = normalize(z_des x x_des)

and re-orthogonalize:

    x_des = normalize(y_des x z_des)

The policy can then be penalized using body-axis alignment:

    1 - dot(body_x, x_des)

plus:

    1 - dot(body_z, z_des)

This is preferable to an Euler roll penalty because it:

- allows the physically required coordinated bank;
- removes the free roll degree of freedom;
- directly penalizes wind-tumble orientation;
- remains well-defined through large bank angles.

---

## 19. Coordinated angular-rate reference

A second important change is to avoid penalizing all angular velocity toward zero.

Circular flight requires nonzero angular velocity.

The new reward should penalize:

    omega_body - omega_body_desired

instead of just:

    omega_body

For approximately 14 m/s, radius 12 m coordinated flight, the desired body-rate magnitude is much smaller and more structured than the old policy's multi-rad/s tumbling.

This should suppress unnecessary roll while preserving the yaw / pitch-rate combination needed for the circular turn.

---

## 20. Proposed first SlowCoordinated reward structure

Keep the successful racing terms:

    gate_passed          +400
    progress              +20

Add / strengthen:

    coordinated_attitude
    coordinated_rate
    speed-band error
    radial error
    height error
    body-rate command L2
    command smoothness

Initial design direction:

    coordinated_attitude     strong penalty
    coordinated_rate         moderate penalty
    speed band               target about 12-14 m/s
    radial error             keep path near radius 12 m
    height error             keep flight around 2.07 m
    body_rate_command        discourage unnecessary rate commands
    command_smoothness       discourage violent command changes

The exact numeric weights should be introduced in a separate new task so the successful baseline and Stable-v1 remain frozen.

---

## 21. Two-stage fine-tuning strategy

Do not train SlowCoordinated from scratch.

Warm-start from the Stable-v1 checkpoint:

    logs/skrl/swift_ctbr_gt_circular12_stable/
    2026-09-22_22-20-55_ppo_torch_circular12_r12_4096env_roll24_256x3_gt_stable_antispin/
    checkpoints/best_agent.pt

Recommended curriculum:

### Stage A

    target speed band     about 14-16 m/s
    body-rate max         (4, 4, 2)

Goal:

- preserve racing skill;
- force coordinated orientation;
- eliminate roll-phase freedom.

### Stage B

    target speed band     about 12-14 m/s

Goal:

- reduce speed further;
- reduce required bank;
- improve visual stability;
- retain continuous gate passing.

This staged transition avoids asking a policy trained around 18-20 m/s to instantly jump to a very different low-speed control distribution.

---

## 22. Acceptance criteria for SlowCoordinated

The next policy is not accepted merely because PPO reward improves.

It must pass all of the following.

### Racing

    continuous Circular-12 gate passing
    full-lap completion remains high
    no collapse into hover / slow loiter

### Speed

    mean speed near target band
    p95 speed significantly below old ~20 m/s regime

### Angular motion

    no sustained full roll cycles
    body-rate norm materially below original policy
    inversion fraction approximately zero

### Geometry

    radial path close to 12 m
    altitude close to 2.07 m

### Attitude

    body thrust axis close to coordinated-turn reference
    body forward axis close to tangent / velocity direction

### Camera / FPV

This is mandatory:

> Several successful 20 s body-fixed FPV episodes must show no wind-tumble cycle.

The visual appearance should become qualitatively close to the prescribed 20-degree-camera steady-turn experiment.

Only after this passes should the new Color20 detector be considered qualified for online estimator validation.

---

## 23. Correct order from this point forward

The project order is now:

    1. implement Circular12-SlowCoordinated-v0

    2. fine-tune from Stable-v1

    3. evaluate random-start racing

    4. run coordinated-motion audit

    5. render body-fixed 20-degree-camera FPV

    6. manually verify zero wind-tumble

    7. then run Color20 vision GT-shadow

    8. then IMU + Color20 vision + SC-EKF

    9. then reintroduce V7 learned motion only if justified

    10. finally drive the frozen racing policy from estimated state

The visual model and Gate-ID work are therefore preserved, not discarded.

They are simply waiting for a physically reasonable free-flight trajectory source.

---

## 24. What should not be done next

Do not yet:

    run final GT-shadow conclusions using the old tumbling policy

    judge the Color20 detector from wind-tumble video

    connect estimator state to the racing policy

    retrain the visual model again only because Gate-ID accuracy is 64%

    further increase generic angular-rate penalties without an orientation target

    freeze any conclusion that the detector is inadequate under normal flight

The current priority is upstream flight quality.

---

## 25. Important artifacts / checkpoints to preserve

Original strong but tumbling Circular-12 GT checkpoint:

    logs/skrl/swift_ctbr_gt_circular12/
    2026-09-22_15-45-53_ppo_torch_circular12_r12_4096env_roll24_256x3_gt/
    checkpoints/best_agent.pt

Stable-v1 checkpoint:

    logs/skrl/swift_ctbr_gt_circular12_stable/
    2026-09-22_22-20-55_ppo_torch_circular12_r12_4096env_roll24_256x3_gt_stable_antispin/
    checkpoints/best_agent.pt

Formal diversified Color20 vision dataset:

    artifacts/racing_vision/
    circular12_color20_h207_diversified_v1/

New Gate-ID + corner detector:

    artifacts/racing_vision/
    circular12_color20_h207_gateid_kprcnn_v1/
    torchvision_keypointrcnn_multigate_best.pt

Gate-ID diagnostics:

    artifacts/racing_vision/
    circular12_color20_h207_gateid_kprcnn_v1/
    gate_id_diagnostics/test/

Prescribed visual reference:

    artifacts/swift_ctbr/
    circular12_constant_bank_3lap/

---

## 26. Remote branch state before this summary

Branch:

    feature/racing-vision-retraining

Remote HEAD before adding this daily summary:

    0a91446f476afbb4c9d5376a6aaaee6c95bb4324

The branch already contains:

- real Circular-12 per-gate textures;
- high-contrast Gate-ID palette;
- USD wrapper-based texture binding;
- Gate identity map;
- 20-degree / 2.07 m prescribed capture tools;
- diversified visual dataset collector;
- manifest recovery tool;
- 13-class Gate-ID Keypoint R-CNN training path;
- Gate-ID confusion matrix evaluator;
- detector audit-video renderer;
- bounded Gate-ID prior + all-map reprojection fallback;
- a Color20 GT-shadow task, intentionally deferred until the GT policy is stable.

---

## 27. First operation next session

Synchronize the branch:

    cd ~/isaac_projects/isaac_drone_racer

    git fetch ssybh2 --prune
    git switch feature/racing-vision-retraining
    git pull --ff-only ssybh2 feature/racing-vision-retraining

    git rev-parse --short HEAD

Then implement the new dedicated task:

    Circular12-SlowCoordinated-v0

without modifying the successful original or Stable-v1 tasks in place.

The implementation should add:

- coordinated-turn attitude reward;
- coordinated body-rate reward;
- speed-band penalty;
- radius penalty;
- height target near 2.07 m;
- tighter CTBR rate authority;
- separate PPO experiment name / log directory;
- motion-audit and FPV evaluation support.

Then fine-tune from Stable-v1 rather than training from zero.

---

## 28. End-of-day conclusion

The work on 2026-09-23 produced a meaningful visual landmark stack:

    unique physical gate texture
        ↓
    Gate ID supervision
        ↓
    12-class detector
        +
    approximately 1.2 px corner localization
        ↓
    known-map identity prior
        +
    all-map geometric fallback

The visual system is substantially more mature than it was at the beginning of the day.

However, the limiting factor has moved back upstream.

The decisive blocker is now:

> The learned GT racing policy must stop wind-tumbling and produce stable coordinated circular flight before it can be used to qualify the vision / estimator stack.

The next milestone is therefore:

> Circular12-SlowCoordinated-v0 — a free-flight GT policy that races at roughly 12-14 m/s, holds the physically correct bank, maintains about 2.07 m altitude and radius-12 geometry, and shows zero complete wind-tumble cycles in body-fixed FPV.
