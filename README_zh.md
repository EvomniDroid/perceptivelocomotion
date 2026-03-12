# 🤖 InstinctLab 训练与使用指南

本项目是基于 **Project Instinct**（清华大学 IIIS & 上海期智研究院）系列顶会论文（如 CoRL 2025 *Embrace Collisions* 等）复现的人形机器人强化学习训练框架。
包含从基础运动、跑酷、全身动作表达到基于视觉的师生蒸馏（Vision-based Distillation）全套流程。

---

## 🎯 核心任务列表 (Tasks)

环境任务通过 `gym.register` 注册，每个任务都配套有用于训练的 `v0` 版本和用于推理验证的 `Play-v0` 版本。

### 1. 基础移动 (Locomotion)
让机器人学习最基础的平地行走与姿态控制。
- **训练任务**: `Instinct-Locomotion-Flat-G1-v0`
- **测试任务**: `Instinct-Locomotion-Flat-G1-Play-v0`

### 2. 跑酷与避障 (Parkour)
基于 Target-AMP 机制的指令驱动跑酷，能够根据参考运动（AmassMotion）跨越地形。
- **训练任务**: `Instinct-Parkour-Target-Amp-G1-v0`
- **测试任务**: `Instinct-Parkour-Target-Amp-G1-Play-v0`

### 3. 全身精准跟随 (Whole Body Shadowing)
不仅控制下肢，还包含上肢、手臂、躯干的精确动捕数据动作模仿。
- **训练任务**: `Instinct-Shadowing-WholeBody-Plane-G1-v0` / `Instinct-BeyondMimic-Plane-G1-v0`
- **测试任务**: `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0` / `Instinct-BeyondMimic-Plane-G1-Play-v0`

### 4. 视觉强化感知系统 (Perceptive) 🌟旗舰流水线🌟
实现从“上帝视角（特权信息）”到“纯深度图视觉（真实传感器部署）”的师生蒸馏跨越。
- **Teacher（教师）训练任务**: `Instinct-Perceptive-Shadowing-G1-v0`
- **Student（学生 VAE）训练任务**: `Instinct-Perceptive-Vae-G1-v0`
- **测试任务**: `Instinct-Perceptive-Vae-G1-Play-v0`

---

## 🚀 基础命令指南

所有命令必须在 `instinctlab` 或 workspace 根目录下执行。

### 1. 启动训练 (Train)
```bash
python scripts/instinct_rl/train.py --task=<Task-ID> --num_envs 64
```
*常用附加参数：*
* `--headless`: 无头模式（不启动图形界面，只在后台训练，极大节省显存，强推）
* `--num_envs`: 并行环境数量（建议根据机器显存动态调整大小，如 64, 128, 256）

### 2. 运行验证 (Play)
训练完成后，加载生成的 `.pt` 权重文件进行可视化验证：
```bash
python source/instinctlab/instinctlab/tasks/相应的目录/scripts/play.py \
    --task=<Task-ID-Play-v0> \
    --load_run=<对应权重的绝对路径或 logs 下的运行目录名>
```
*附加参数：*
* `--keyboard_control`: 运行期间启用键盘控制机器人的走向。

### 3. 模型导出 (Export for Deployment)
将训练好的 PyTorch 策略模型导出为可供真机部署的 ONNX 格式：
```bash
python source/instinctlab/instinctlab/tasks/相应的目录/scripts/play.py \
    --task=<Task-ID> \
    --load_run=<run_name> \
    --exportonnx --useonnx
```

---

## 🎓 重点：师生视觉蒸馏 (Perceptive) 完整实操流程

此流程用于训练出可直接在真实 Unitree G1 等双足机器人上部署的纯物理+视觉决策大脑（Vision-based RL）。

### Step 1: 训练 Teacher 模型 (盲人宗师)
Teacher 拥有“特权信息”（如环境中精准的实时地形高度和物理摩擦力），它在复杂的约束下学习出最鲁棒的全身动作跟随。
```bash
python scripts/instinct_rl/train.py --task=Instinct-Perceptive-Shadowing-G1-v0 --num_envs 64
```
**输出**：通常在达到 1 ~ 3 万个迭代后会收敛，权重自动保存在 `/logs/instinct_rl/g1_perceptive_shadowing/` 目录下。

### Step 2: 挂载 Teacher 给 Student
复制上一步生成的最好的 Teacher 权重文件夹绝对路径，将其填入对应 Student 的算法配置文件 `agents/instinct_rl_vae_cfg.py` 中的 `teacher_logdir` 字段。

### Step 3: 开始蒸馏 Student 模型 (VAE)
启动 VAE 任务，迫使只能观测自身传感器（关节）和深度图影像（Depth Image）的 Student 模型去模仿 Teacher。
该过程依赖最小化 `distillation_loss`（动作复制损失）与 `kl_loss`（潜空间分布差异）。
```bash
python scripts/instinct_rl/train.py --task=Instinct-Perceptive-Vae-G1-v0 --num_envs 64
```

### Step 4: 仿真验证与真机导出
等 Student 的 Loss 收敛不再下降后，提取其生成的 `.pt` 文件进入验证环节，验证无误导出 ONNX 文件，即可进入下游的 `deploy` 上真机实测（请注意系好安全绳消除 Sim2Real Gap）。
