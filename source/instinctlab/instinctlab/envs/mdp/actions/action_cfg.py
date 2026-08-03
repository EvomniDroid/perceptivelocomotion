from dataclasses import MISSING

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from . import joint_actions


@configclass
class ActionOverridenJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the action overridden delayed joint position action term.

    See :class:`ActionOverridenointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.ActionOverridenJointPositionAction

    asset_cfg: SceneEntityCfg = MISSING
    """Whether to override the action with the delayed action. Defaults to False."""

    override_value: float = 0.0
    """Delay in frames before the action is overridden. Defaults to 0."""


@configclass
class DynamicTargetJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action that can be overridden by a per-env target stored on the asset."""

    class_type: type[ActionTerm] = joint_actions.DynamicTargetJointPositionAction

    target_attr_name: str = "_dynamic_joint_position_target"
    """Name of the articulation attribute containing a full joint target tensor."""

    ee_target_pos_attr_name: str | None = None
    """Optional articulation attribute containing an end-effector target position in world frame."""

    ee_body_name: str | None = None
    """End-effector body name used for Jacobian IK when ee_target_pos_attr_name is set."""

    ik_damping: float = 0.05
    """Damping used by the damped-least-squares IK solve."""

    ik_gain: float = 0.7
    """Joint update gain applied to the IK solution each policy step."""

    max_ik_delta: float = 0.12
    """Maximum end-effector position correction per policy step in meters."""


@configclass
class FixedJointPositionActionCfg(ActionTermCfg):
    """Configuration for a fixed joint target with zero policy dimensions."""

    class_type: type[ActionTerm] = joint_actions.FixedJointPositionAction

    joint_names: list[str] = MISSING
    """Joint names to hold."""

    joint_pos: dict[str, float] = MISSING
    """Fixed position target for every selected joint."""

    preserve_order: bool = False
    """Whether to preserve the configured joint-name order."""
