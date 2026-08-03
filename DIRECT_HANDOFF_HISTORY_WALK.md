# B2RM Direct-Handoff History Walk

Train this task from scratch with:

```bash
cd ~/b2rmlow/perceptivelocomotion
/home/qx100/IsaacLab-2.0.0/isaaclab.sh -p scripts/instinct_rl/train.py \
  --task Instinct-B2RM-LegOnly-Velocity-Handoff-Walk-v0 --headless
```

The task ID now uses `B2RMHandoffRlEnv`.  Its scripted startup matches the
real deployment contract:

```text
prone -> target1 at 1000/10 -> target2 at 1000/10 -> 0.16 s target2 hold
-> direct policy frame zero at 250/5
```

There is no low-gain default-pose policy frame and no gain/action blend.
The target2 hold fills the history buffer while the robot is still at
1000/10.

Each of the eight history frames has 57 values:

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
 3 handoff controls: normalized Kp, normalized Kd, takeover-ready flag
--
57 values per frame x 8 frames = 456 actor observations
```

The policy remains leg-only: it outputs 12 position actions with
`action_scale=0.4`, and the post-handoff leg PD is `Kp=250`, `Kd=5`.
`35%` of environments are explicit zero-command standing episodes.  New
ONNX exports therefore require a new real deployment reader for 456 inputs;
the current 50-input C++ deployment intentionally will reject them.
