import sys
import os

# ========= 手动注入 Isaac Sim 路径 (请根据你实际的 isaac-sim 版本号修改，比如 4.0.0 或 4.2.0) =========
ISAAC_SIM_PATH = os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.2.0") 

if os.path.exists(ISAAC_SIM_PATH):
    sys.path.append(os.path.join(ISAAC_SIM_PATH, "exts/omni.isaac.kit"))
    sys.path.append(os.path.join(ISAAC_SIM_PATH, "exts/omni.isaac.sim.python"))
    # 这一步是让 Python 能找到 carb 的核心
    sys.path.append(os.path.join(ISAAC_SIM_PATH, "kit/python/lib/python3.10/site-packages")) 
else:
    print(f"[警告] 未在默认路径找到 Isaac Sim: {ISAAC_SIM_PATH}，请检查版本号是否匹配！")
# =======================================================================================

# 后面才是你原本的 import
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
# 1. 必须先创建仿真上下文（这会启动 Omniverse 视窗/后台）
sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim_context = SimulationContext(sim_cfg)

# 2. 配置并 Spawn 资产
__file_dir__ = '/home/zh/isaac/instinctlab/source/instinctlab/instinctlab/assets'
B2RM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(__file_dir__, 'resources/unitree_b2/urdf/b2rm.urdf'),
        fix_base=False,
    ),
)

sim_utils.create_ground_plane('/World/ground', sim_cfg)

# 这一步会在 Stage 中真正创建 Prim
robot = Articulation(cfg=B2RM_CFG, prim_path='/World/b2rm')

# 3. 必须让仿真上下文 Play，把资产真正加载进内存，这时才能获取到关节名字
sim_context.reset()

print(f'Joint names: {robot.joint_names}')