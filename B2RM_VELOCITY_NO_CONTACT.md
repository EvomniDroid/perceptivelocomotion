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
- action scale 0.4 and action clip +/-1.5
- eight history frames and term-major flattening
- command-gated 0.7 s gait clock
- 40% standing environments
- contact-based gait rewards in simulation
- mass, COM, friction, actuator-gain, observation-noise, and delay randomization

The action clip is expressed in IsaacLab as per-joint processed target limits:

```text
hip:   [-0.60,  0.60]
thigh: [ 0.07,  1.27]
calf:  [-1.90, -0.70]
```

These ranges equal `target2 + 0.4 * clip(raw_action, -1.5, 1.5)`. The previous
generic processed-target clip of `[-1.5, 1.5]` was not equivalent to the real
adapter and allowed the training policy to execute targets that deployment
would modify differently.

## Deployment follow-up

After a new checkpoint is available:

1. Change the C++ frame size from 51 to 47 and total input from 408 to 376.
2. Remove contact values from history construction and flattening.
3. Keep raw force logging for diagnostics only.
4. Validate `[1,376] -> [1,12]` and exact PyTorch/ONNX agreement offline.
5. Test zero command first, then 0.05 m/s with a 25-degree fall guard.
