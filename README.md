# 历史响应迁移与联合反馈测试

本项目研究自动驾驶系统（ADS）在有限候选集中的失效发现：利用其他驾驶系统的历史测试响应，结合新系统的少量执行反馈，在给定预算内优先发现碰撞场景。方法英文名为 *Historical Response Transfer with Joint Feedback*。

不同 ADS 在相同场景中的行为存在差异。方法保留各历史来源的风险预测和不确定性，再用目标系统的碰撞标签与 RSS 安全距离裕度修正预测和来源权重。真实失效由仿真碰撞事件判定。

## 方法设计

| 环节 | 实现思路 |
|---|---|
| RSS 辅助的历史风险先验构建 | 联合建模历史碰撞与 RSS，学习共享表征，分别构建各历史 ADS 的响应后验 |
| 少样本目标风险适配 | 冻结共享表征，用全部已付费反馈适配目标残差；按查询前碰撞似然更新来源权重 |
| 预算约束下的失效导向测试 | 从共同初始化开始，逐次执行预测碰撞概率最高的未查询候选，反馈后继续适配 |
| 任务完成后的知识积累 | 归档已执行记录，为后续任务构建来源，并在任务之间更新共享模型 |

各功能场景独立训练、测试和积累知识，迁移发生在同类场景的不同 ADS 之间。当前联合模型使用碰撞 Bernoulli 观测与 RSS 高斯观测，论文选定共享／独立潜过程结构。模型构建与在线推断见[方法说明](testing_methods/history_transfer/README.md)。

## 当前实现

HighwayEnv 主实验包含匀速跟车、前车制动、前车切出暴露障碍、自车换道与后车冲突，使用规则、优化、搜索及学习控制器。每类包含 2,000 个候选，每任务共同初始化 20 次、总预算 200 次，算法种子为 11。MetaDrive 提供独立的场景执行和搜索实现。

评价包括唯一碰撞候选数、发现曲线及 AUC、完整候选真值覆盖率，以及归一化参数空间的两两距离和最近邻距离。当前选例按碰撞概率排序，多样性由评估端衡量。设置与证据见[主结果](artifacts/runs/joint_response/paper/reports/results.md)及[冻结协议](artifacts/runs/joint_response/paper/reports/protocol.json)，各组件的收益按相应比较解释。

## 工程结构

| 目录 | 职责 |
|---|---|
| [testing_methods/](testing_methods/README.md) | 本文方法、基线及共享概率模型 |
| [controllers/](controllers/README.md) | 被测驾驶控制器和策略权重 |
| [simulator/](simulator/README.md) | 闭环仿真、可见观测与轨迹记录 |
| [experiments/](experiments/README.md) | 数据准备、预算查询、实验调度和报告导出 |
| [artifacts/](artifacts/README.md) | 执行档案、模型、结果和图片索引 |
| [tests/](tests/README.md) | 方法、反馈权限、仿真和指标的回归检查 |
| [docs/](docs/style.md) | 编码、目录组织与研究写作规范 |
| [ref/](ref/README.md) | 学术图片的样式参考 |

## 构建与运行

从历史响应构建模型，先训练基础联合核，再以历史前缀的下一条碰撞响应训练编码器。共享／独立核通过 `training.train(..., kernel_type=SharedPrivateKernel)` 构建，完整示例见[方法入口](testing_methods/history_transfer/README.md#构建与使用)。

已有论文数据和模型通过以下入口复用；数据采集、训练及其他操作见[实验说明](experiments/README.md)。

~~~powershell
python -B -m experiments.paper_validation
python -B -m pytest tests -q
~~~

<details>
<summary>HighwayEnv 环境安装</summary>

~~~powershell
conda create -n metadrive -c pytorch -c conda-forge python=3.10 pytorch=2.5.0 numpy=2.2.6 gymnasium=0.29.1 scipy=1.15.3 shapely=2.1.2 scikit-learn=1.7.2 pyyaml matplotlib pillow threadpoolctl pytest psutil pip
conda activate metadrive
python -m pip install highway-env==1.9.1 stable-baselines3==2.3.0 optuna==5.0.0 torch-geometric==2.7.0 scikit-fuzzy==0.5.0
~~~

MetaDrive 的依赖与运行入口见[平台说明](simulator/metadrive/README.md)。

</details>
