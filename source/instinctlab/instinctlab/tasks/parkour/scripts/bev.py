"""
BEV摔倒率图生成脚本

将深度图转换为鸟瞰图(BEV)，并叠加预设摔倒率地图
"""

import numpy as np
import cv2
import os
from pathlib import Path

CAMERA_INTRINSICS = {
    'fx': 720.6,
    'fy': 820.1,
    'cx': 320.0,
    'cy': 160.0,
    'width': 640,
    'height': 320
}

CAMERA_HEIGHT = 0.5

ROBOT_POS = (0.0, 0.0)
ROBOT_YAW = 0.0

BEV_X_RANGE = (-2.0, 2.0)
BEV_Y_RANGE = (0.0, 4.0)
BEV_RESOLUTION = 0.01

MAP_SIZE = 12.0
MAP_CELL_SIZE = 0.01

def get_preset_fall_rate(world_x, world_y):
    """获取预设摔倒率"""
    in_danger_zone = (3.5 <= world_x <= 4.5 and 1.5 <= world_y <= 2.5)
    return 0.8 if in_danger_zone else 0.1

def world_to_map_index(world_x, world_y):
    """世界坐标转地图索引"""
    col = int(world_x / MAP_CELL_SIZE)
    row = int((MAP_SIZE - world_y) / MAP_CELL_SIZE)
    row = np.clip(row, 0, int(MAP_SIZE / MAP_CELL_SIZE) - 1)
    col = np.clip(col, 0, int(MAP_SIZE / MAP_CELL_SIZE) - 1)
    return row, col

def depth_to_bev_with_fallback(depth_path):
    """
    将深度图转换为BEV深度图和BEV摔倒率图

    Returns:
        bev_depth: BEV深度图 (米)
        bev_fall_rate: BEV摔倒率图
    """
    depth_img = cv2.imread(depth_path)
    if depth_img is None:
        print(f"[ERROR] 无法读取: {depth_path}")
        return None, None

    if len(depth_img.shape) == 3:
        depth_gray = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        depth_gray = depth_img.astype(np.float32)

    depth_m = depth_gray / 255.0 * 10.0

    fx = CAMERA_INTRINSICS['fx']
    fy = CAMERA_INTRINSICS['fy']
    cx = CAMERA_INTRINSICS['cx']
    cy = CAMERA_INTRINSICS['cy']
    h, w = depth_m.shape

    bev_width = int((BEV_X_RANGE[1] - BEV_X_RANGE[0]) / BEV_RESOLUTION)
    bev_height = int((BEV_Y_RANGE[1] - BEV_Y_RANGE[0]) / BEV_RESOLUTION)

    bev_depth = np.zeros((bev_height, bev_width), dtype=np.float32)
    bev_fall_rate = np.zeros((bev_height, bev_width), dtype=np.float32)
    bev_count = np.zeros((bev_height, bev_width), dtype=np.float32)

    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

    depth_valid = depth_m > 0.1

    u_flat = u_coords[depth_valid]
    v_flat = v_coords[depth_valid]
    d_flat = depth_m[depth_valid]

    x_cam = (cx - u_flat) * d_flat / fx
    y_cam = (v_flat - cy) * d_flat / fy
    z_cam = d_flat

    x_flat = x_cam
    y_flat = z_cam
    z_flat = np.abs(y_cam)

    z_world = CAMERA_HEIGHT - z_flat

    cos_yaw = np.cos(ROBOT_YAW)
    sin_yaw = np.sin(ROBOT_YAW)

    world_x_robot = x_flat * cos_yaw - y_flat * sin_yaw + ROBOT_POS[0]
    world_y_robot = x_flat * sin_yaw + y_flat * cos_yaw + ROBOT_POS[1]

    valid = (z_world > 0) & (z_world < 3.0) & (world_x_robot >= ROBOT_POS[0] - 2) & (world_x_robot < ROBOT_POS[0] + 2) & (world_y_robot >= ROBOT_POS[1]) & (world_y_robot < ROBOT_POS[1] + 4)

    bev_u = ((world_x_robot[valid] - (ROBOT_POS[0] - 2)) / BEV_RESOLUTION).astype(int)
    bev_v = ((world_y_robot[valid] - ROBOT_POS[1]) / BEV_RESOLUTION).astype(int)

    valid_bev = (bev_u >= 0) & (bev_u < bev_width) & (bev_v >= 0) & (bev_v < bev_height)

    for i in range(len(bev_u)):
        if valid_bev[i]:
            depth_val = z_world[valid][i]
            v_idx = bev_height - 1 - bev_v[i]
            u_idx = bev_u[i]
            if bev_count[v_idx, u_idx] == 0 or depth_val < bev_depth[v_idx, u_idx]:
                bev_depth[v_idx, u_idx] = depth_val
                bev_fall_rate[v_idx, u_idx] = get_preset_fall_rate(world_x_robot[valid][i], world_y_robot[valid][i])
            bev_count[v_idx, u_idx] += 1

    return bev_depth, bev_fall_rate

def process_folder(depth_folder, output_folder):
    """处理一个文件夹中的所有深度图"""
    depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png') and 'color' not in f])

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'fall_rate'), exist_ok=True)

    print(f"[INFO] 处理文件夹: {depth_folder}")
    print(f"[INFO] 找到 {len(depth_files)} 张深度图")
    print(f"[INFO] 机器人位置: {ROBOT_POS}, 朝向: {ROBOT_YAW}")

    for i, filename in enumerate(depth_files):
        depth_path = os.path.join(depth_folder, filename)
        bev_depth, bev_fall_rate = depth_to_bev_with_fallback(depth_path)

        if bev_depth is not None:
            bev_depth_vis = (bev_depth / bev_depth.max() * 255).astype(np.uint8) if bev_depth.max() > 0 else np.zeros_like(bev_depth, dtype=np.uint8)
            cv2.imwrite(os.path.join(output_folder, 'depth', filename), bev_depth_vis)

            fall_rate_vis = (bev_fall_rate * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, 'fall_rate', filename), fall_rate_vis)

            if i % 10 == 0:
                print(f"[PROGRESS] {i+1}/{len(depth_files)} 完成")

    print(f"[DONE] 结果保存到: {output_folder}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成BEV摔倒率图')
    parser.add_argument('--depth_folder', type=str, default='/home/zh/isaac/instinctlab/logs/instinct_rl/g1_parkour/20260326_142216/depth_images/run_20260423_201925_1280', help='深度图文件夹')
    parser.add_argument('--output_folder', type=str, default='/home/zh/isaac/liveratemodel/data/bev', help='输出文件夹')
    parser.add_argument('--robot_x', type=float, default=3.0, help='机器人X坐标')
    parser.add_argument('--robot_y', type=float, default=1.5, help='机器人Y坐标')
    parser.add_argument('--robot_yaw', type=float, default=0.0, help='机器人朝向(弧度)')
    args = parser.parse_args()

    global ROBOT_POS, ROBOT_YAW
    ROBOT_POS = (args.robot_x, args.robot_y)
    ROBOT_YAW = args.robot_yaw

    process_folder(args.depth_folder, args.output_folder)

if __name__ == '__main__':
    main()
