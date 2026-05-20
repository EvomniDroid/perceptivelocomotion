# Residual RL 在线适应方案技术报告

**项目**: 视觉蒸馏策略 + 真实环境在线适应
**日期**: 2026-05-15
**状态**: 方案设计阶段

---

## 1. 背景与目标

### 1.1 当前状态
- 已在仿真中完成视觉蒸馏策略训练
- 主策略 (π_base) 包含CNN视觉编码器，能识别地形但无法感知"软硬"
- 部署到真实机器人时，沙地、泥地等软质地形的视觉特征与普通地面相似

### 1.2 问题
```
仿真中训练的地形 vs 真实部署地形:
- 视觉特征相似（都是"土色"）
- 物理特性不同（沙泥软、阻力大、会陷进去）
- 主策略在仿真中无法学到这些物理差异
```

### 1.3 目标
```
在真实机器人上实现:
1. 主策略冻结，保证机器人不会乱动
2. 在线训练一个小型残差模块 (π_residual)
3. 残差模块通过触觉反馈学习适应软硬地形
4. 最终动作 = 主策略动作 + α * 残差动作
```

---

## 2. 方案选择

### 2.1 方案对比

| 方案 | 复杂度 | 安全性 | 实施难度 |
|------|--------|--------|----------|
| A: Residual RL | 低 | 高 | 中 |
| B: RMA风格Adaptation | 高 | 中 | 高 |
| C: 触觉提示 | 低 | 高 | 低 |

**选定方案**: **方案A - Residual RL**

### 2.2 为什么选 Residual RL

1. **安全性高**: 主策略冻结，机器人不会因在线学习而失控
2. **样本效率高**: 小模块需要的数据量少（估计1000-5000步可学会基础）
3. **实现简单**: 不需要改主网络结构，只需要加一个小模块
4. **可插拔**: 残差模块可以单独保存，随时加载/卸载

---

## 3. 技术方案详细设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        真实机器人部署                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   视觉观察 ──→ ┌─────────────┐                                  │
│                │  主策略     │ → action_base (冻结，不更新)     │
│   本体感觉 ──→ │ π_base     │                                  │
│                │ (视觉CNN    │                                  │
│                │  + 控制器)  │                                  │
│                └─────────────┘                                  │
│                       ↓                                          │
│                                                                 │
│   ┌─────────────────────────────────────────────┐                │
│   │           残差模块 π_residual               │                │
│   │           (在线更新，小网络)                 │                │
│   ├─────────────────────────────────────────────┤                │
│   │ 输入:                                       │                │
│   │   - 关节角度 (12维)                        │                │
│   │   - 关节速度 (12维)                        │                │
│   │   - 足端接触信号 (4维，二值)                │                │
│   │   - 关节力矩 (12维，关键！沙泥阻力大)       │                │
│   │   - 上一步动作 (12维)                      │                │
│   │ 输出: 动作残差 Δa (12维，Tanh限制[-1,1])   │                │
│   └─────────────────────────────────────────────┘                │
│                       ↓                                          │
│              action = action_base + α * Δa                      │
│                       ↓                                          │
│                  机器人执行                                       │
│                       ↓                                          │
│   ┌─────────────────────────────────────────────┐                │
│   │              在线更新逻辑                     │                │
│   │   - 检测"异常": 力矩大但速度小 = 陷进去了    │                │
│   │   - 如果这步 reward 好 → 强化这个残差       │                │
│   │   - 使用简单策略: REINFORCE 或 PG           │                │
│   └─────────────────────────────────────────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 残差模块网络结构

```python
class ResidualAdapter(nn.Module):
    """
    小型残差适应模块，用于在线学习软硬地形
    输入维度: 12(joint_pos) + 12(joint_vel) + 4(contact) + 12(torque) + 12(last_action) = 52
    输出维度: 12 (关节动作残差)
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(52, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),  # 正则化，防止过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 12),  # 输出12维动作残差
            nn.Tanh()  # 限制输出范围 [-1, 1]
        )
        # α: 残差系数，限制残差模块的影响力
        self.alpha = 0.1  # 可学习或固定

    def forward(self, x):
        residual = self.net(x)
        return self.alpha * residual  # 限制幅度
```

### 3.3 在线更新策略

```python
def online_update(residual_adapter, batch, optimizer):
    """
    使用策略梯度更新残差模块
    简化的REINFORCE算法
    """
    states, actions_base, rewards, next_states = batch

    # 1. 计算当前残差
    residual = residual_adapter(states)

    # 2. 实际动作 = base + residual
    actual_actions = actions_base + residual

    # 3. 简化的策略梯度损失
    # 目标: 最大化累积奖励
    # 梯度 = E[∇log π(a|s) * R]
    loss = -residual.mean() * rewards.mean()  # 简化版

    # 4. 更新
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(residual_adapter.parameters(), 1.0)
    optimizer.step()

    return loss
```

### 3.4 触觉感知关键信号

```python
class TerrainPerception:
    """
    从关节力矩和接触信号感知地形软硬
    """

    @staticmethod
    def detect_sinking(joint_torques, joint_velocities, foot_contacts):
        """
        检测是否陷入软地形
        启发式规则:
        - 力矩大 + 速度小 + 脚着地 = 可能陷进去了
        """
        mean_torque = joint_torques.mean()
        mean_velocity = joint_velocities.abs().mean()

        # 如果力矩超过阈值但速度很低，认为在软地上打滑/陷进去
        if mean_torque > 5.0 and mean_velocity < 0.1:
            return True, "sinking"  # 陷进去了
        return False, "normal"

    @staticmethod
    def compute_terrain_reward(joint_torques, foot_contacts, base_action):
        """
        计算地形适应奖励
        用于指导残差模块学习
        """
        # 正向奖励: 力矩正常 + 脚接触稳定
        torque_penalty = (joint_torques ** 2).mean() * 0.01
        contact_reward = foot_contacts.float().mean() * 0.1

        return contact_reward - torque_penalty
```

### 3.5 Reset 检测与处理

```python
class ResetHandler:
    """
    处理真实机器人训练中的reset情况
    """

    def __init__(self, tilt_threshold=30):
        self.tilt_threshold = tilt_threshold  # 倾斜角度阈值(度)
        self.no_contact_duration = 0  # 无脚着地持续时间

    def check_reset(self, roll, pitch, foot_contacts):
        """
        判断是否需要重置
        """
        # 情况1: 彻底摔倒 (roll/pitch超过阈值)
        if abs(roll) > self.tilt_threshold or abs(pitch) > self.tilt_threshold:
            return "fallen", "tilt_exceeded"

        # 情况2: 被人抱起来 (没有脚着地且持续一段时间)
        if foot_contacts.sum() == 0:
            self.no_contact_duration += 1
            if self.no_contact_duration > 50:  # 约2.5秒
                return "carried", "no_contact"
        else:
            self.no_contact_duration = 0

        return "normal", ""

    def get_reset_action(self, reset_type):
        """
        返回重置后的初始动作
        """
        if reset_type == "fallen":
            # 触发站起来策略 (如果有)
            return "stand_up"
        else:
            # 被抱起时给0动作，等着被放下
            return np.zeros(12)
```

---

## 4. 实施计划

### 阶段1: 仿真预训练 (1-2周)

**目标**: 在仿真中预训练残差模块，使其具备基本的地形适应能力

```
1. 创建软硬地形仿真环境
   - 沙地: 高阻力、会陷进去
   - 泥地: 粘性大、速度慢

2. 主策略冻结，只训练残差模块
   - 在多种软硬程度的地形上训练
   - 残差模块学习"这种情况要这样抬腿"

3. 保存预训练好的残差模块权重
```

### 阶段2: 真实部署 + 在线微调 (持续)

**目标**: 在真实机器人上部署，并根据实际反馈微调

```
1. 加载预训练的残差模块
2. 主策略冻结，残差模块解冻
3. 实时收集触觉数据
4. 在线更新残差模块
5. 每隔N步评估一次 reward
6. 如果 reward 持续提升，说明在学到东西
```

### 阶段3: 安全监控

**目标**: 保证在线学习不会让机器人失控

```
1. 残差幅度限制: α ≤ 0.2
2. 动作幅度限制: 最终动作不超过关节安全范围
3. 异常检测: 如果残差输出异常大，自动回退
4. 随时可切换: 一键切换到"只使用主策略"模式
```

---

## 5. 关键参数配置

| 参数 | 建议值 | 说明 |
|------|--------|------|
| α (残差系数) | 0.1 | 限制残差模块影响力 |
| hidden_dim | 128 | 残差网络隐藏层维度 |
| update_freq | 1 | 每步都更新 |
| batch_size | 32 | 离线更新时的batch |
| tilt_threshold | 30° | 摔倒检测阈值 |
| torque_threshold | 5.0 | 软地检测阈值 |
| max_residual_norm | 0.2 | 残差幅度上限 |

---

## 6. 参考资料

### 6.1 论文

| 论文 | 关键点 |
|------|--------|
| RMA (2021) | Adaptation Module 从状态历史估计地形特征 |
| A Walk in the Park (2023) | 真实机器人在线学习，20分钟学会行走 |
| Residual RL | 主策略固定，在线学残差补偿 |
| DroQ | SAC + Dropout + LayerNorm，高效在线更新 |

### 6.2 代码参考

| 项目 | 路径 | 用途 |
|------|------|------|
| raisimLib/RMA | /home/zh/isaac/raisimLib/raisimGymTorch/ | Adaptation Module参考 |
| walk_in_the_park | /home/zh/isaac/walk_in_the_park/ | 在线学习训练循环参考 |
| instinctlab | /home/zh/isaac/instinctlab/ | 主策略训练框架 |

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 在线学习导致机器人失控 | 高 | 残差幅度严格限制，主策略冻结 |
| 样本效率低，学不会 | 中 | 仿真预训练 + 残差模块足够小 |
| 触觉信号噪声大 | 低 | 使用滑动窗口平滑 + 阈值过滤 |
| 真实地形与仿真不符 | 中 | 在线微调 + 残差模块泛化能力 |

---

## 8. 后续修改记录

| 日期 | 修改内容 | 修改原因 |
|------|----------|----------|
| 2026-05-15 | 初始方案设计 | - |

---

## 9. 快速参考

### 9.1 核心公式

```
最终动作 = 主策略动作 + α * 残差模块动作
a_final = π_base(obs) + α * π_residual(state_history)

其中 state_history = [关节角度, 关节速度, 足端接触, 关节力矩, 上一步动作]
```

### 9.2 关键文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 主策略 | /home/zh/isaac/instinctlab/scripts/instinct_rl/train.py | 视觉蒸馏训练 |
| 残差模块 | 待创建 | 在线适应模块 |
| 在线更新 | 待创建 | 部署时的更新逻辑 |

### 9.3 部署命令模板

```bash
# 仿真中预训练残差模块
python train_residual.py --terrain=soft --num_envs=100 --epochs=1000

# 真实机器人部署 + 在线学习
python deploy_with_residual.py \
    --base_policy=/path/to/base_policy.pt \
    --residual_module=/path/to/residual_module.pt \
    --real_robot=True \
    --enable_online_update=True
```
