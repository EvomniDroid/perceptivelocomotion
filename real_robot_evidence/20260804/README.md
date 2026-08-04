# B2RM real-robot evidence package: 2026-08-04

## Purpose

This package records the evidence behind the next non-visual B2RM velocity
training revision. It is intended to be reviewed together with:

- `source/instinctlab/instinctlab/tasks/parkour/config/b2rm/b2rm_velocity_handoff_cfg.py`
- `B2RM_VELOCITY_HISTORY_BASELINE.md`
- `B2RM_VELOCITY_NO_CONTACT.md`

The central result is that the failed `0.05 m/s` test was a progressive
closed-loop divergence, not an ONNX, joint-map, or first-frame failure. Two
confirmed sim-to-real contract mismatches were found:

1. real foot-contact observations were always `[1, 1, 1, 1]`;
2. training and deployment clipped actions at different stages.

Both are addressed in the new training configuration. A new checkpoint must
be trained from scratch before the corresponding 376-D deployment adapter is
enabled.

## Raw logs

`raw_logs.tar.gz` contains the full unmodified logs for:

```text
b2rm_velocity_history_cpp/logs/20260804_210538_history
b2rm_velocity_history_cpp/logs/20260804_210601_history
b2rm_foot_force_calibration_cpp/logs/20260804_211539_stand
b2rm_foot_force_calibration_cpp/logs/20260804_211617_official_walk
b2rm_foot_force_calibration_cpp/logs/20260804_211708_official_run
b2rm_foot_force_calibration_cpp/logs/20260804_212302_suspended
```

Extract from the repository root with:

```bash
mkdir -p /tmp/b2rm_real_logs
tar -xzf real_robot_evidence/20260804/raw_logs.tar.gz -C /tmp/b2rm_real_logs
```

Archive SHA-256:

```text
99657e90f20912b8a0c736e909a13339a4e225b91fa20b42fa05ccc93065a46c
```

## Systems and contracts tested

### Earlier 432-D actor

Checkpoint:

```text
logs/instinct_rl/b2rm_velocity/20260803_195906
```

Actor terms included eight frames of base linear velocity and foot contacts.
Three zero-command runs established that ONNX inference, term-major history,
joint order, policy rate, and arm exclusion were working, but trajectories
diverged after takeover:

| Run | Policy duration | Maximum tilt | Maximum raw action | Outcome |
|---|---:|---:|---:|---|
| `20260804_155115_history` | 9.63 s | 12.75 deg | 1.82 | remained upright, comparatively stable |
| `20260804_160151_history` | 2.30 s | 26.52 deg | 2.42 | progressive tilt after about 1 s |
| `20260804_160448_history` | 2.03 s | 20.24 deg | 2.00 | progressive tilt after about 1 s |

First-frame action RMS differed by only about `0.04` between these runs. The
differences increased after the blend and reached about `0.57..0.60` by two
seconds. That behavior indicates feedback/history divergence rather than a
random first action or a simple swapped joint.

### Current tested 408-D actor

Checkpoint and exported actor:

```text
logs/instinct_rl/b2rm_velocity/20260804_175932/model_5000.pt
logs/instinct_rl/b2rm_velocity/20260804_175932/exported/actor.onnx
```

Contract:

```text
actor:  [1,408] -> [1,12]
critic: 432
history: 8 frames, term-major, oldest to newest
policy rate: 50 Hz
low-level rate: 500 Hz
PD: 250/5
action scale: 0.4
```

Actor per-frame terms were:

```text
q_rel[12], dq[12], base_ang_vel[3], projected_gravity[3],
command[3], raw_last_action[12], command_gated_gait_phase[2],
foot_contacts[4]
```

Base linear velocity was already removed from this actor and was diagnostic
only in deployment. Therefore the inaccurate stance-foot velocity estimator
did not directly cause the 408-D policy output.

## Latest zero-command experiment

Log:

```text
20260804_210538_history
command=(0,0,0)
```

Summary:

- 571 policy frames, about 11.43 s under policy control;
- gait clock stayed frozen at `[0,1]`;
- reported contacts remained `[1,1,1,1]`;
- tilt settled near 3.33 degrees;
- no raw action exceeded the deployment clip;
- the robot remained standing and the front-leg convergence was smaller than
  in earlier checkpoints.

This result shows that the target2 pose, 250/5 gains, joint mapping, IMU at the
nominal pose, ONNX execution, and eight-frame packing are at least sufficient
for a repeatable zero-command run. It does not prove dynamic gait feedback.

## Latest 0.05 m/s experiment

Log:

```text
20260804_210601_history
command=(0.05,0,0)
```

The user stopped the test after 70 policy frames, about 1.388 s. The gait clock
activated because the implementation uses `command_norm >= 0.05`.

Progressive tilt:

| Time after policy start | Approximate tilt |
|---:|---:|
| 0.00 s | 1.76 deg |
| 0.20 s | 1.15 deg |
| 0.50 s | 7.44 deg |
| 1.01 s | 11.08 deg |
| 1.39 s | 36.92 deg |

At the end, projected gravity was approximately:

```text
(-0.266, -0.539, -0.799)
```

and body angular velocity included approximately:

```text
gyro_x=1.22 rad/s, gyro_y=1.51 rad/s
```

This confirms a real right-side roll/pitch divergence. It did not occur on the
first policy frame and was not an instantaneous gain-switch kick.

Important action evidence:

- 9 of 70 policy frames contained a raw action outside `+/-1.5`;
- FR calf reached about `+1.793` at 1.208 s;
- the deployment target for that joint differed by about `0.117 rad` from the
  target produced by the old training clip semantics;
- approximate PD effort reached 246 Nm on RR calf and 175 Nm on RL calf during
  the divergence, while the other joint groups were substantially lower.

The rear-leg effort is likely a consequence and amplifier of the falling
closed loop, not proof that higher or lower Kp alone is the root cause.

## Offline inference evidence

An earlier recorded real frame was replayed through its ONNX actor with:

```text
maximum replay-versus-recorded action error = 5.66e-7
```

The tested actor input/output shape and logged output matched. This rules out
ONNX numerical disagreement as the main cause. A fully nominal observation
still produced asymmetric non-zero actions, showing that asymmetry can be a
learned policy behavior rather than a logging artifact.

## Foot-force calibration

Four passive `rt/lowstate` captures were recorded at about 166 Hz. The recorder
did not publish commands. `foot_force_est[0..3]` was identically zero in every
capture.

Raw `foot_force` medians:

| Capture | FR | FL | RR | RL |
|---|---:|---:|---:|---:|
| standing, 33.31 s | 514 | 485 | 496 | 456 |
| official walk, 28.72 s | 514 | 499 | 495 | 506 |
| official run, 19.90 s | 512 | 512 | 507 | 502 |
| safely suspended, 32.31 s | 520 | 495 | 521 | 496 |

Observed minima:

| Capture | FR | FL | RR | RL |
|---|---:|---:|---:|---:|
| standing | 469 | 434 | 457 | 415 |
| official walk | 380 | 409 | 322 | 351 |
| official run | 380 | 392 | 328 | 354 |
| suspended | 463 | 444 | 473 | 443 |

Conclusions:

1. The deployment threshold `foot_force >= 20` always returns contact.
2. Raw force contains a large leg-dependent offset around 500.
3. Suspended values do not approach zero and overlap standing/walking values.
4. The direction and separation are not consistent across all four legs; FR
   in particular does not admit a robust loaded/unloaded scalar threshold.
5. Replacing 20 with a common value such as 400 or 500 is unsupported.
6. These force fields should remain diagnostic-only unless a separate verified
   estimator is developed.

During the failed gait run, the 408-D actor therefore saw a changing gait
clock while all eight frames of contact history said that every foot remained
on the ground. This is the strongest confirmed observation mismatch.

## Confirmed code changes

### 1. Remove contacts from the deployable actor

The actor now contains only real-robot observables:

| Term | Per frame | Eight frames |
|---|---:|---:|
| leg q relative to default | 12 | 96 |
| leg dq | 12 | 96 |
| base angular velocity | 3 | 24 |
| projected gravity | 3 | 24 |
| velocity command | 3 | 24 |
| previous raw action | 12 | 96 |
| command-gated gait phase | 2 | 16 |
| **Actor total** | **47** | **376** |

The critic retains eight frames of privileged simulator base linear velocity
and four simulator contact bits, so it remains 432-D. Contact sensors also
remain in gait rewards. Only the deployed actor input was changed.

Expected new ONNX contract:

```text
[1,376] -> [1,12]
```

### 2. Align action clipping with deployment

IsaacLab's `JointPositionAction` computes:

```text
processed_target = default + scale * raw_action
processed_target = clip(processed_target, configured_joint_target_limits)
```

The previous training config set every processed target to `[-1.5,1.5]`, while
the C++ adapter clips raw action to `[-1.5,1.5]` before multiplying by 0.4.
These operations are not equivalent.

The new training target limits exactly represent target2 plus the deployed raw
action range:

```text
hip:   [-0.60,  0.60]
thigh: [ 0.07,  1.27]
calf:  [-1.90, -0.70]
```

Thus both training and deployment implement:

```text
q_des = target2 + 0.4 * clip(raw_action, -1.5, 1.5)
```

## Deliberately unchanged for the next training run

To keep the experiment attributable, the next run should initially retain:

- target2: hip 0, thigh 0.67, calf -1.30;
- leg PD 250/5;
- action scale 0.4 and raw clip +/-1.5;
- eight-frame term-major history;
- 0.7 s command-gated gait clock;
- 40% standing-command environments;
- existing contact-based simulation rewards;
- friction, mass, COM, actuator-gain, observation-noise and 0..2-step actuator
  delay randomization;
- independent fixed folded-arm control.

Do not resume the 408-D checkpoint: the actor input changed. Train the 376-D
actor from scratch.

## Remaining high-priority checks

These are not yet proven root causes and should be evaluated separately.

### IMU frame and sign validation

Static agreement is insufficient. Under safe support, deliberately tilt the
body forward/back and left/right and verify Unitree gyro/quaternion/projected
gravity signs against IsaacLab body-frame conventions. A sign error can remain
invisible at zero pose and create positive feedback during gait.

### Earlier safety damping

The current trigger `projected_gravity_z > -0.50` corresponds to about 60
degrees. The failed run reached 36.9 degrees without triggering. Before the
next free-standing gait test, use a validated 20..25 degree threshold with a
short persistence window and also guard:

- excessive roll/pitch rate;
- stale/non-finite low state;
- persistent action clipping;
- excessive joint speed;
- excessive `|q_des-q|` or estimated PD effort.

Validate all safety behavior while supported first.

### Low-speed command onset and gait clock

`0.05 m/s` is exactly the current gait activation boundary. Training command
resampling can transition from standing to moving while the clock is at an
arbitrary global episode phase. Consider a stateful gait clock that resets at
command onset, and explicitly sample commands in the `0.04..0.10 m/s` range.
Deployment must implement exactly the same onset/reset rule. A command ramp is
useful only if the gait phase is reset when the threshold is crossed.

### Episode-constant sensor bias

Current per-frame uniform observation noise does not reproduce constant real
encoder and IMU offsets. Consider per-episode random bias for:

- joint position zero;
- gyroscope;
- projected gravity/orientation;
- left/right motor gain asymmetry.

Add these only after checking their measured real ranges.

### Arm payload and lateral COM

Verify the real folded arm angles, arm/tool payload, cable forces, total mass,
and lateral COM against the B2RM URDF. Existing base mass and COM randomization
may not cover a mismatched arm payload or mounting offset.

## Deployment work after retraining

Do not switch the current C++ adapter before the new ONNX exists. Then update
training and deployment together:

1. require `[1,376] -> [1,12]`;
2. use 47 values per frame and remove contact history packing;
3. retain raw foot-force logging as diagnostics only;
4. keep raw previous action semantics aligned with IsaacLab;
5. verify PyTorch versus ONNX on multiple observations;
6. verify term-major oldest-to-newest packing;
7. add the earlier safety guards;
8. test zero command before any moving command.

## Proposed test sequence and acceptance criteria

1. Isaac play: zero, 0.05, 0.10 and lateral/yaw commands.
2. Offline ONNX comparison against PyTorch.
3. Supported real test at zero command, including safety-trigger tests.
4. Free-standing zero command for 30..60 seconds across repeated runs.
5. Supported or securely restrained `0.05 m/s` test.
6. Free `0.05 m/s` only after logs show bounded tilt/action/error.

Acceptance criteria before increasing speed:

- repeated zero-command runs remain upright for at least 30..60 seconds;
- gait tests do not show progressive roll/pitch divergence;
- raw-action clipping is rare and not persistent;
- projected gravity, gyro and joint directions are verified;
- rear-calf target error and estimated effort remain bounded;
- no stale state, NaN, excessive speed or safety-damping event;
- repeated runs remain similar beyond two seconds.

## Questions for the reviewing AI

1. Is the 376-D actor / 432-D privileged critic split implemented correctly?
2. Are the per-joint processed target limits mathematically equivalent to the
   deployment raw-action clip for all 12 joints?
3. Should command-onset gait phase be reset, randomized, or learned without an
   explicit clock for low-speed locomotion?
4. Which episode-constant bias ranges are justified by the real logs?
5. Does the B2RM URDF mass/COM/arm model cover the physical robot?
6. Which safety limits can be added without masking policy quality during
   supported tests?
7. After the two confirmed fixes, what is the smallest next experiment that
   can distinguish IMU-frame error from insufficient policy robustness?
