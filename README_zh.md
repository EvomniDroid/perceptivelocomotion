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

---

## 🧠 算法思想深度解析：为什么需要“师生蒸馏”？视觉是怎么工作的？

在具身智能领域，**直接让机器人在仿真里“看图（像素）学走路”是极难收敛的**。因为强化学习去试错时不仅要学“如何控制电机不摔倒”，还要学“如何从成千上万个像素点里理解这是个台阶”。
因此，这项工作采用了目前顶会（如 CoRL 2025）最核心的设计哲学——**不对称 Actor-Critic + 师生策略蒸馏（Teacher-Student Distillation）**。

### 1.第一阶段：Teacher 怎么学？（上帝视角的盲人宗师）
* **原理**: 在 `Perceptive-Shadowing`（Teacher 环境）中，我们给机器人开启了“外挂” (Privileged Information)。
* **输入**: 它**不看**深度图！系统直接将地形的**精确三维高度点阵 (Height Scan)**、完美的机身速度、物理摩擦力等以一维数组的形式直接喂给神经网络。
* **为什么？**: 对多层感知机（MLP）来说，这些 1D 数据极好理解。借助上帝视角，强化学习（通过跟踪给定的动捕数据与奖励函数，如 `link_pos_imitation` 贴合惩罚、`joint_limit` 关节极限惩罚）能极快收敛，练就一套不仅动作逼真，且不论地形多复杂都不摔倒的“无敌策略”。

### 2.第二阶段：Student 怎么学？（长出眼睛的现实量产机）
* **原理**: 现实中 G1 机器人的脑门上只有 RealSense 深度相机，没有绝对精准的上帝视角地形高度扫描器。
* **输入**: Student 被强制要求**只看**历史关节状态和**深度图影像 (Depth Image)**。
* **怎样更新**: Student 在此阶段**不再主要依靠在环境里试错、拿 Reward 分数来更新**，而是变为一种**监督学习**！
  1. Teacher 在同一套环境里走，它的输出被奉为完美的“标准答案”。
  2. Student 通过其自带的 `Conv2dHeadModel` 卷积视觉网络模块，拍下一张黑白景深照片。
  3. 照片进入 **VAE (变分自编码器)**，将 2D 像素压缩提取成 1D 的“隐变量 (Latent Vector)”（理解为地形特征），然后交出动作。
  4. 算法强制计算 Student 给出的动作与 Teacher 标准答案之间的差值（主要观测日志中的 `distillation_loss` 以及潜空间分布散度 `kl_loss`）。通过缩小差值，将 Teacher 肌肉记忆的分布“蒸馏（刻印）”给 Student。

> **终极目的**：Teacher 负责开挂学环境动力学；Student 负责学“怎么把眼睛拍到的深度图，翻译成对应地形的特征并触发 Teacher 教我的动作”。练成下山后，Student 就可以靠着一双“眼睛”在真机上稳稳当当度过复杂的现实路面了。
