from __future__ import annotations

from collections.abc import Callable

import torch


def handoff_blend_state(env) -> torch.Tensor:
    """Expose action and gain blend progress to the handoff policy."""
    action_alpha = getattr(env, "handoff_action_alpha", None)
    gain_alpha = getattr(env, "handoff_gain_alpha", None)
    if action_alpha is None or gain_alpha is None:
        return torch.zeros((env.num_envs, 2), device=env.device)
    return torch.stack((action_alpha, gain_alpha), dim=-1)


def handoff_control_state(env) -> torch.Tensor:
    """Expose the PD regime and the imminent/direct policy takeover.

    The actor receives this term in its observation history.  In particular,
    the final high-gain target2 observation carries ``takeover_ready=1`` so
    policy frame zero can distinguish an imminent 1000/10 -> 250/5 handoff
    from ordinary standing.
    """
    gain_alpha = getattr(env, "handoff_gain_alpha", None)
    step_buf = getattr(env, "handoff_step_buf", None)
    hold_end = getattr(env, "_hold_end", None)
    if gain_alpha is None or step_buf is None or hold_end is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    stand_kp = float(getattr(env.cfg, "handoff_stand_kp", 250.0))
    stand_kd = float(getattr(env.cfg, "handoff_stand_kd", 5.0))
    policy_kp = float(getattr(env.cfg, "handoff_policy_kp", stand_kp))
    policy_kd = float(getattr(env.cfg, "handoff_policy_kd", stand_kd))
    kp = stand_kp * (1.0 - gain_alpha) + policy_kp * gain_alpha
    kd = stand_kd * (1.0 - gain_alpha) + policy_kd * gain_alpha
    takeover_ready = (step_buf >= hold_end).to(dtype=torch.float)
    return torch.stack((kp / 1000.0, kd / 10.0, takeover_ready), dim=-1)


def foot_contacts(env, sensor_cfg, force_threshold: float = 20.0) -> torch.Tensor:
    """Return ordered binary foot contacts for policy observations."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1) >= force_threshold).to(dtype=torch.float)


def handoff_gated_termination(
    env,
    wrapped_func: Callable,
    wrapped_params: dict,
    counter_name: str | None = None,
    minimum_duration_s: float = 0.0,
) -> torch.Tensor:
    """Disable ordinary episode termination until scripted handoff completes."""
    terminated = wrapped_func(env, **wrapped_params)
    active = getattr(env, "handoff_policy_active", None)
    if active is None:
        return terminated
    gated = torch.logical_and(terminated, active)
    if counter_name is None or minimum_duration_s <= 0.0:
        return gated

    counters = getattr(env, "_handoff_termination_counters", None)
    if counters is None:
        counters = {}
        env._handoff_termination_counters = counters
    counter = counters.get(counter_name)
    if counter is None:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        counters[counter_name] = counter

    counter[:] = torch.where(gated, counter + 1, 0)
    required_steps = max(1, int(round(minimum_duration_s / env.step_dt)))
    return counter >= required_steps
