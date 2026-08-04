# B2RM Velocity-History Deployment Baseline

The single non-visual deployment baseline is:

```text
Instinct-B2RM-LegOnly-Velocity-History-v0
Instinct-B2RM-LegOnly-Velocity-History-Play-v0
```

## Reset

Every episode starts directly from the deployed target2 pose at `Kp/Kd=250/5`:

```text
root_z: 0.58 m
hips:   0.0 rad
thighs: 0.67 rad
calves: -1.30 rad
```

The policy controls from frame zero. Scripted prone stand-up and the
`1000/10 -> 250/5` handoff are intentionally outside the learned policy.

## Actor contract

The actor receives eight-frame, term-major history and does not receive
simulator ground-truth base linear velocity:

```text
joint_pos             8 x 12
joint_vel             8 x 12
base_ang_vel          8 x 3
projected_gravity     8 x 3
velocity_command      8 x 3
previous_action       8 x 12
command-gated clock   8 x 2
foot_contacts         8 x 4
--------------------------------
actor total              408
```

The critic additionally receives eight frames of simulator base linear
velocity, giving a total of 432 inputs. For a command norm below `0.05`, the
gait clock is frozen at `[0, 1]`.

## First-stage command distribution

```text
standing samples: 40%
vx: -0.20 .. 0.20 m/s
vy: -0.10 .. 0.10 m/s
wz: -0.20 .. 0.20 rad/s
```

The first acceptance target is robust zero-command standing and low-speed
tracking. Expand these ranges only after real logs pass that target.

## Deployment objectives

- low action magnitude at zero command
- low body linear/angular motion at zero command
- four-foot support at zero command
- penalty above an action soft limit of 1.0
- existing action-rate, pose, orientation, torque and gait objectives
- friction, mass, center-of-mass and actuator-gain randomization
- delayed actuator with random `0..2` physics-step latency

The arm remains physically present and is held by its independent fixed action
term, but it is absent from actor input and output.

## Registered B2RM training tasks

1. `Instinct-Parkour-Target-Amp-B2RM-v0`
2. `Instinct-B2RM-LegOnly-Velocity-History-v0`

All superseded non-visual velocity and handoff experiments are unregistered.

## Saved baseline checkpoint

```text
logs/instinct_rl/b2rm_velocity/20260804_175932/model_5000.pt
```

This is the stable target2-start, 408-D actor baseline checkpoint for deployment
testing. Its inference contract matches this task. The subsequent
`velocity_command_deficit` reward update affects only future training runs, not
the checkpoint's inference interface.
