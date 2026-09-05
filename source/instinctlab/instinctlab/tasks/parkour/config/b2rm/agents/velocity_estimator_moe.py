from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from instinct_rl.modules.moe_actor_critic import MoEActorCritic
from instinct_rl.modules.state_estimator import EstimatorMixin
from instinct_rl.utils.utils import get_obs_slice, get_subobs_indexing_by_components


class _EstimatorActorExporter(nn.Module):
    """Export the estimator and actor as one deployment-safe graph."""

    def __init__(self, model: "B2RMEstimatorMoEActorCritic"):
        super().__init__()
        self.actor = model.actor
        self.state_estimator = model.state_estimator
        self.obs_segments = model.obs_segments
        self.target_components = set(model.estimator_target_components)
        input_indices = get_subobs_indexing_by_components(
            model.obs_segments,
            model.estimator_obs_components,
        )
        self.register_buffer("estimator_input_indices", input_indices.long())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        estimator_input = torch.index_select(observations, -1, self.estimator_input_indices)
        estimated_state = self.state_estimator(estimator_input)

        actor_parts = []
        estimate_start = 0
        for component, shape in self.obs_segments.items():
            obs_slice, _ = get_obs_slice(self.obs_segments, component)
            if component in self.target_components:
                width = int(np.prod(shape))
                actor_parts.append(estimated_state[..., estimate_start : estimate_start + width])
                estimate_start += width
            else:
                actor_parts.append(observations[..., obs_slice])
        return self.actor(torch.cat(actor_parts, dim=-1))


class B2RMEstimatorMoEActorCritic(EstimatorMixin, MoEActorCritic):
    """MoE locomotion policy with a supervised concurrent velocity estimator."""

    def export_as_onnx(self, observations: torch.Tensor, filedir: str):
        if self.is_recurrent:
            raise RuntimeError("B2RMEstimatorMoEActorCritic export expects a feed-forward estimator.")

        self.eval()
        os.makedirs(filedir, exist_ok=True)
        combined_model = _EstimatorActorExporter(self).eval()
        with torch.no_grad():
            torch.onnx.export(
                combined_model,
                observations,
                os.path.join(filedir, "actor.onnx"),
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )

            estimator_input = torch.index_select(
                observations,
                -1,
                combined_model.estimator_input_indices,
            )
            torch.onnx.export(
                self.state_estimator,
                estimator_input,
                os.path.join(filedir, "velocity_estimator.onnx"),
                input_names=["estimator_input"],
                output_names=["base_lin_vel_b"],
                opset_version=12,
            )

        print(f"Exported estimator+actor model to {os.path.join(filedir, 'actor.onnx')}")
        print(f"Exported velocity estimator to {os.path.join(filedir, 'velocity_estimator.onnx')}")
