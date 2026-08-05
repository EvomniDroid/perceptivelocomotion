# B2RM velocity actor without foot contacts

## Why this variant exists

Real-robot calibration showed that `foot_force_est` is always zero and raw
`foot_force` has a large, leg-dependent offset. Standing, official walking,
official running, and suspended measurements do not admit a reliable common
or per-foot binary contact threshold. The previous actor therefore received
eight frames of incorrect all-contact observations during commanded gait.

## Observation contract

The actor keeps eight-frame, term-major history:

| Term | Per frame | Eight frames |
|---|---:|---:|
| leg joint position relative to default | 12 | 96 |
| leg joint velocity | 12 | 96 |
| base angular velocity | 3 | 24 |
| projected gravity | 3 | 24 |
| velocity command | 3 | 24 |
| previous raw action | 12 | 96 |
| command-gated gait sine/cosine | 2 | 16 |
| **Actor total** | **47** | **376** |

The critic keeps privileged base linear velocity and four simulated contact
bits, so its input remains 432-D. Contact sensors also remain in reward terms;
only the actor input changed.

Expected exported ONNX contract:

```text
[1, 376] -> [1, 12]
```

Do not deploy the existing 408-D checkpoint with the 376-D adapter. Retrain
from scratch, export the new actor, verify its dimensions, then update the
real-robot deployment contract and default checkpoint together.

## What remains unchanged

- target2 default pose
- 250/5 policy PD gains
- action scale 0.3 and raw action clip +/-1.0
- eight history frames and term-major flattening
- command-gated 0.7 s gait clock
- 30% standing environments
- contact-based gait rewards in simulation
- mass, COM, friction, actuator-gain, observation-noise, and delay randomization

The action clip is expressed in IsaacLab as per-joint processed target limits:

```text
hip:   [-0.30,  0.30]
thigh: [ 0.37,  0.97]
calf:  [-1.60, -1.00]
```

These ranges equal `target2 + 0.3 * clip(raw_action, -1.0, 1.0)`. The previous
generic processed-target clip of `[-1.5, 1.5]` was not equivalent to the real
adapter and allowed the training policy to execute targets that deployment
would modify differently.

## Command and gait sampling

- 30% of environments receive an exact zero command.
- 70% of moving environments receive pure forward commands.
- Forward speed is sampled from 0.05 to 0.50 m/s.
- The remaining moving environments also cover lateral and yaw commands.
- The gait clock resets whenever a new command is sampled.
- Swing-speed and foot-height objectives scale with command magnitude, so a
  low-speed command does not demand a full-amplitude fast swing.
- Linear tracking uses a tighter 0.15 m/s exponential kernel and the velocity
  deficit term requires at least 75% of commanded speed.

## Deployment follow-up

After a new checkpoint is available:

1. Change the C++ frame size from 51 to 47 and total input from 408 to 376.
2. Remove contact values from history construction and flattening.
3. Keep raw force logging for diagnostics only.
4. Validate `[1,376] -> [1,12]` and exact PyTorch/ONNX agreement offline.
5. Set deployment action scale/clip to 0.3 and +/-1.0 before loading this model.
6. Test zero command first, then 0.05 m/s with a 25-degree fall guard.
