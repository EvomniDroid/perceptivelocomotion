"""查看机器人的简单脚本 - 支持 b2 / g1 / b2rm，WASD 给推力，数字键调机械臂角度"""
# python source/instinctlab/instinctlab/tasks/parkour/scripts/view_robot.py b2rm fix_base
# python source/instinctlab/instinctlab/tasks/parkour/scripts/view_robot.py b2rm
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import sys
import math
import torch
import carb
import omni
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.assets import AssetBaseCfg

robot_name = sys.argv[1] if len(sys.argv) > 1 else "b2"
mode = sys.argv[2] if len(sys.argv) > 2 else "free"

if robot_name == "b2":
    from instinctlab.assets.unitree_b2 import B2_CFG as ROBOT_CFG
    title = "B2 机器人"
elif robot_name == "b2rm":
    from instinctlab.assets.unitree_b2rm import B2RM_CFG as ROBOT_CFG
    title = "B2RM 机器人"
elif robot_name == "g1":
    from instinctlab.assets.unitree_g1 import G1_CFG as ROBOT_CFG
    title = "G1 机器人"
else:
    print(f"未知机器人: {robot_name}, 可选: b2, b2rm, g1")
    simulation_app.close()
    exit(1)

has_arm = robot_name == "b2rm"
arm_joint_names = [f"arm_joint_{i}" for i in range(1, 7)]

robot_cfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

if mode == "fix_base":
    robot_cfg.spawn.fix_base = True
    title += " (base fixed)"
elif mode == "collision":
    robot_cfg.spawn.fix_base = True
    title += " (collision view)"


@configclass
class ViewSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 2.5

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = robot_cfg


force = torch.zeros(3, device="cuda:0")
torque = torch.zeros(3, device="cuda:0")
want_reset = False
FORCE_SCALE = 500.0

selected_arm_joint = 0
arm_joint_angles = [math.radians(0), math.radians(90), math.radians(-174), math.radians(0), math.radians(-9), math.radians(1)]


def on_keyboard_event(event, *args, **kwargs):
    global force, torque, want_reset, selected_arm_joint, arm_joint_angles
    if isinstance(event.input, str):
        key = event.input
    else:
        key = event.input.name
    if key.startswith("KEY_") and len(key) == 5 and key[4].isdigit():
        key = key[4]
    elif key.upper() in ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"]:
        key = {"ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6"}[key.upper()]
    is_number_key = key in "123456"
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if key == "W":
            force[0] = FORCE_SCALE
        elif key == "S":
            force[0] = -FORCE_SCALE
        elif key == "A":
            force[1] = FORCE_SCALE
        elif key == "D":
            force[1] = -FORCE_SCALE
        elif key == "Q":
            torque[2] = 200.0
        elif key == "E":
            torque[2] = -200.0
        elif key == "R":
            force[:] = 0.0
            torque[:] = 0.0
        elif key == "SPACE":
            want_reset = True
        if has_arm:
            arm_key = False
            if key == "U":
                arm_joint_angles[selected_arm_joint] += math.radians(5)
                arm_key = True
            elif key == "J":
                arm_joint_angles[selected_arm_joint] -= math.radians(5)
                arm_key = True
            elif key == "I":
                arm_joint_angles[selected_arm_joint] += math.radians(1)
                arm_key = True
            elif key == "K":
                arm_joint_angles[selected_arm_joint] -= math.radians(1)
                arm_key = True
            elif is_number_key:
                key_val = key.upper()
                if key_val in "123456":
                    idx = int(key_val) - 1
                elif "NUMPAD" in key_val or "NUM_" in key_val:
                    idx = int(key_val[-1]) - 1
                else:
                    idx = 0
                selected_arm_joint = idx
                arm_key = True
            if arm_key:
                deg = math.degrees(arm_joint_angles[selected_arm_joint])
                print(f"  关节 {selected_arm_joint+1}: {deg:.1f}° | 全部: "
                      f"[{', '.join(f'{math.degrees(a):.0f}°' for a in arm_joint_angles)}]")
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        if key == "W" and force[0] > 0:
            force[0] = 0.0
        elif key == "S" and force[0] < 0:
            force[0] = 0.0
        elif key == "A" and force[1] > 0:
            force[1] = 0.0
        elif key == "D" and force[1] < 0:
            force[1] = 0.0
        elif key == "Q" and torque[2] > 0:
            torque[2] = 0.0
        elif key == "E" and torque[2] < 0:
            torque[2] = 0.0
    return True


def main():
    global force, torque, want_reset, arm_joint_angles

    app_window = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()
    keyboard_sub = input_iface.subscribe_to_keyboard_events(
        keyboard,
        on_keyboard_event,
    )

    sim_cfg = SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 1.5, 1.2], [0.0, 0.0, 0.0])

    scene_cfg = ViewSceneCfg()
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    sim.set_camera_view([2.0, 1.5, 1.2], [0.0, 0.0, 0.0])

    robot = scene["robot"]

    arm_joint_indices = []
    if has_arm:
        for name in arm_joint_names:
            arm_joint_indices.append(robot.find_joints(name)[0][0])

    print("=" * 60)
    print(f"{title} 已加载！")
    print("用法: python view_robot.py <robot> <mode>")
    print("  robot: b2, b2rm, g1")
    print("  mode : free(默认), fix_base, collision")
    print()
    print("键盘控制:")
    print("  W/S  - 向前/向后推力")
    print("  A/D  - 向左/向右推力")
    print("  Q/E  - 绕Z轴旋转力矩")
    print("  R    - 清除所有力和力矩")
    print("  Space - 重置机器人姿态")
    if has_arm:
        print()
        print("机械臂调姿（fix_base模式下使用）:")
        print("  1~6  - 选择关节1~6")
        print("  U/J  - 选中关节 +/-5°")
        print("  I/K  - 选中关节 +/-1° (微调)")
        print("  Ctrl+C 复制当前角度打印输出")
    print("  鼠标中键拖动旋转视角，滚轮缩放")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    while simulation_app.is_running():
        sim.step()
        if not sim.is_playing():
            sim.reset()

        if want_reset:
            want_reset = False
            root_state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(root_state[:, :7])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            force[:] = 0.0
            torque[:] = 0.0

        if has_arm:
            joint_pos = robot.data.joint_pos.clone()
            for i, idx in enumerate(arm_joint_indices):
                joint_pos[0, idx] = arm_joint_angles[i]
            robot.write_joint_state_to_sim(joint_pos, robot.data.joint_vel.clone())

        body_force = torch.zeros((1, robot.num_bodies, 3), device="cuda:0")
        body_torque = torch.zeros((1, robot.num_bodies, 3), device="cuda:0")
        body_force[0, 0, :] = force
        body_torque[0, 0, :] = torque
        robot.root_physx_view.apply_forces_and_torques_at_position(
            force_data=body_force,
            torque_data=body_torque,
            position_data=None,
            indices=torch.tensor([0], dtype=torch.long, device="cuda:0"),
            is_global=False,
        )

    input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    simulation_app.close()


if __name__ == "__main__":
    main()
