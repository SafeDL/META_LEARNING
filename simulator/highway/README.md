# HighwayEnv 场景执行

本目录提供注册环境 `response-highway-v2`，按照 `highway-env` 官方 `AbstractEnv` 接口执行直道交通交互，记录原生碰撞事件、最小车身净距、TTC 和 20 Hz 车辆轨迹。

## 文件职责

| 文件 | 职责 |
|---|---|
| [collision_env.py](collision_env.py) | `CollisionWorld(AbstractEnv)`，复用官方 `step()`、`_simulate()`、`Road.act()`、`Road.step()` 与车辆碰撞判定；支持跟车制动、受限切入、切出露障碍、侧向侵入及驾驶能力课程 |
| [interaction_env.py](interaction_env.py) | 配对切入场景的类型校验封装，底层与其余场景共用 `CollisionWorld` |
| [observation.py](observation.py) | 生成控制器可见状态 `VisibleWorld`，执行射线几何遮挡判定并过滤未来脚本信息 |
| [timing.py](timing.py) | 训练、能力验收、数据采集与规划预测共用的 20 Hz 物理／5 Hz 决策时钟契约 |
| [training_env.py](training_env.py) | 离散元动作强化学习训练环境 `DrivingEnv`（供 `PPO`／`DQN` 训练） |
| [continuous_env.py](continuous_env.py) | 连续加速度与转向强化学习训练环境 `ContinuousDrivingEnv`（供 `Continuous SAC` 在隔离命名空间 `policy_train` 训练并排除全部保留坐标） |
| [repair_env.py](repair_env.py) | 同谱系策略修复训练环境 `RepairEnv`，按登记比例混合普通驾驶与已付费碰撞重置点 |

---

## 一、物理时钟、场景语义与碰撞判定

1. **四类正式场景**：跟车制动 `car_following_braking_onset`（单车道，前车从 0 s 恒减速至停车，时长 14 s）、受限切入 `cut_in_onset`（三车道，时长 10 s）、切出露障碍 `cut_out_rear_gap`（双车道，时长 10 s）和侧向侵入 `side_encroachment`（双车道，时长 10 s）。参数域定义与整数不放回采样见 [scenarios.py](../../experiments/scenarios.py) 与 [formal_scenarios.py](../../experiments/formal_scenarios.py)。
2. **原生动力学与时钟**：物理更新固定 20 Hz、控制决策固定 5 Hz。所有车辆通过原生 `Vehicle.step()` 推进；自车指令由被测控制器给出，背景车辆由意图控制器输出受相同运动学限幅约束的加速度与转向。
3. **碰撞标签与可见性边界**：碰撞标签直接取原生 `ego.crashed`，碰撞对象取自原生冲击回调记录；几何间距、TTC 和 RSS 裕度作为独立轨迹测量派生，不修改原生碰撞判定。控制器仅接收当前时刻不可变的 `VisibleWorld`。
4. **离线轨迹回放**：[Safe 回放](../../results/scene_replays/safe/README.md)与[危险回放](../../results/scene_replays/collision/README.md)直接读取已封存的 20 Hz 原生轨迹，使用 Pillow 重建动画帧并校验像素与帧时长，不重新调用仿真器。

---

## 二、环境安装

```powershell
conda create -n metadrive -c pytorch -c conda-forge python=3.10 pytorch=2.5.0 numpy=2.2.6 gymnasium=0.29.1 scipy=1.15.3 shapely=2.1.2 scikit-learn=1.7.2 pyyaml matplotlib pillow threadpoolctl pytest psutil pip
conda activate metadrive
python -m pip install highway-env==1.9.1 stable-baselines3==2.3.0 optuna==5.0.0 torch-geometric==2.7.0 scikit-fuzzy==0.5.0
```

MetaDrive 平台的额外依赖与入口见 [MetaDrive 说明](../metadrive/README.md)。

---

## 三、策略修复课程与坐标隔离

- [repair_env.py](repair_env.py) 复用 `DrivingEnv` 的可见观测、五类离散驾驶意图与奖励函数。仅将当前版本链已付费发现的碰撞场景开放为修复重置点，策略通过与原生环境重新闭环交互采集当前 rollout；独立评价点、固定套件与未查询候选按物理坐标从训练重置池中严格排除。
- [continuous_env.py](continuous_env.py) 在 `policy_train` 命名空间生成训练场景，并显式校验排除全部 8,010 个已登记目标坐标、正式能力课程与开发选模课程坐标。
