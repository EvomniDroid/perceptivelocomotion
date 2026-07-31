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
