# B2RM Generic Velocity History Policy

Train the standing-start generic velocity task with:

```bash
cd ~/isaac/instinctlab
python scripts/instinct_rl/train.py \
  --task=Instinct-B2RM-LegOnly-Velocity-History-v0 \
  --num_envs=2048 \
  --max_iterations=5000
```

The environment resets directly around the SDK2/deployment `target2` pose
(`hip=0`, `thigh=0.67`, `calf=-1.30`, root height `0.58 m`). Leg joints
retain the basic-locomotion-style `+/-0.2 rad` reset randomization around
that center pose. The
leg controller remains at `Kp=250`, `Kd=5`; there is no scripted stand-up,
gain switch, or action blend.

Each of the eight history frames has 54 values:

```text
12 joint position offsets
12 joint velocities
 3 base linear velocity
 3 base angular velocity
 3 projected gravity
 3 velocity command
12 previous leg actions
 2 gait clock values
 4 foot contacts: FL, FR, RL, RR
--
54 values per frame x 8 frames = 432 actor observations
```

The gait clock and diagonal-trot rewards use a synchronized `0.7 s` period
(`1.43 Hz`), between basic-locomotion B2's `1.4 Hz` clock and the
unitree_rl_lab B2RM's `0.6 s` period.

IsaacLab stores history per observation term. The flattened policy input is
therefore term-major, not frame-major:

```text
joint_pos[8x12], joint_vel[8x12], base_lin_vel[8x3],
base_ang_vel[8x3], projected_gravity[8x3], command[8x3],
previous_action[8x12], gait_phase[8x2], foot_contacts[8x4]
```

The policy outputs only 12 leg position actions with `action_scale=0.4`.
The arm remains physically present and is held at its folded target outside
the policy output.

Command coverage:

```text
vx: -0.50 .. 0.50 m/s
vy: -0.25 .. 0.25 m/s
wz: -0.50 .. 0.50 rad/s
standing episodes:   12.5%
```

The command components are sampled independently with IsaacLab's
`UniformVelocityCommand`; they are not derived from a Parkour target point.
Actions are clipped to `+/-1.5` before applying `action_scale=0.4`.

Task-local domain randomization includes robot friction/restitution, all-link
mass scaling, base mass and center-of-mass offsets, leg PD gains, the existing
reset force, periodic push, and the delayed actuator's `0..2` physics-step lag.

The current 50-input C++ deployment path intentionally rejects this
432-input model. Deployment needs an eight-frame history reader matching
the ordering above.
