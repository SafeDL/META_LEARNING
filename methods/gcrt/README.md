# 基于历史测试结果的碰撞场景选择

本方法利用历史驾驶系统的测试结果，估计新系统在哪些场景中容易碰撞，并逐次选择下一次测试。研究问题、系统名单、场景和指标见 [项目说明](../../README.md)。`gcrt` 保留为代码目录名，当前实现采用固定查询预算。

训练时，轮流把一个历史系统当作“暂时未知的新系统”，用少量已知结果模拟适配，再检查剩余场景的预测。这个模拟目标自己的结果不能混入历史上下文。碰撞标签是主要监督，间距和 TTC 用于辅助学习场景关系；独立开发系统用于选择模型。

正式测试时，场景编码器保持不变。每获得一次真实反馈，就更新碰撞概率及各历史来源的参考权重；另有一个只依靠目标反馈和物理参数的分量。选例时同时考虑当前碰撞概率，以及测完这个场景对后续选择的帮助，窗口最多为 5 次查询。

该选例规则属于 [ENS 主动搜索](https://proceedings.mlr.press/v70/jiang17d.html) 的条件收益近似结构，使用本项目的历史后验和有限窗口；它与历史表示、目标适配的作用需要分别比较。RGPE碰撞分类基线的80项、ScaML及同推断目标控制的160项独立开发比较均已完成并审计；ENS的独立基线尚未实现。

| 文件 | 作用 |
|---|---|
| [train.py](train.py) | 从历史和开发档案训练场景表示与碰撞模型 |
| [prior.py](prior.py) | 组合历史来源预测及目标自身的预测分量 |
| [posterior.py](posterior.py) | 根据新的碰撞或未碰撞标签更新概率与来源权重 |
| [selection.py](selection.py) | 比较下一次测试的当前收益和后续预期收益 |
| [discovery.py](discovery.py) | 独立开发实验的来源独立核、局部可靠性及查询前证据记录 |
| [consistency.py](consistency.py) | 相同网络的匹配元训练与支持集规模诊断，独立保存开发模型 |
| [response_adaptation.py](response_adaptation.py) | 后续开发的来源响应坐标与目标核适配，比较可适配和固定来源参数 |
| [safety_response.py](safety_response.py)、[response_geometry.py](response_geometry.py) | 历史连续安全响应，以及只由初始场景参数计算的相对运动预测坐标 |
| [adaptive_target.py](adaptive_target.py) | 强目标适配与历史来源的统一查询前可靠性；包含不使用历史来源的融合对照 |
| [joint_response.py](joint_response.py)、[joint_prediction.py](joint_prediction.py) | 碰撞标记与连续间距的联合GP推断，以及与完整后验一致的批量边缘预测 |
| [joint_training.py](joint_training.py) | 从角色隔离的历史档案训练联合模型，开发NLL选择检查点 |
| [joint_search.py](joint_search.py) | 历史联合后验迁移、目标核适配与查询前联合似然更新；280项开发验证已完成并独立审计，主方法未通过门槛 |
| [physical_mixture.py](physical_mixture.py) | 历史后验与独立物理GP的选择原型；全局配置仅使用在线碰撞，320项开发验证已完成并独立审计，主方法未通过门槛 |
| [conditional_adaptation.py](conditional_adaptation.py) | 固定来源条件下的碰撞GP适配及分类边缘概率，用于相同付费前缀的机制诊断 |
| [phase_response.py](phase_response.py) | 三阶段条件标签、早期终止后的未观察项，以及历史阶段GP后验；目前为离线预测原型 |
| [phase_calibration.py](phase_calibration.py) | 保持初始核不变，使独立目标分量的初始边际概率匹配历史风险集频率；不保证适配后的校准 |
| [residual_scale.py](residual_scale.py) | 分离跨系统残差幅度与绝对风险核；提供缓存核重建、来源条件先验和查询前可靠性更新，参数只由历史系统间响应选择 |
| [residual_search.py](residual_search.py) | 预先固定历史尺度的碰撞搜索，含局部/全局、固定尺度、单教师和双教师目标控制；通过隔离查询获取实际反馈 |
| [response_locality.py](response_locality.py) | 以固定历史来源碰撞概率构建可靠性传播距离；仅用于历史预测诊断，未启动新的在线搜索 |
| [counterfactual_adaptation.py](counterfactual_adaptation.py) | 分离虚拟反馈的后验条件化、目标核重拟合和物理核重拟合；目前用于历史前瞻近似诊断 |
| [converged_response.py](converged_response.py)、[converged_search.py](converged_search.py) | 独立收敛的来源条件分类推断，以及局部/全局和同推断目标控制；160项完整矩阵及480次独立重建通过，指定主方法未通过发现收益门槛 |
| [feature_reliability.py](feature_reliability.py) | 保留查询前对数似然，按当前目标特征重新计算其局部影响；只用于历史距离诊断，不改变已登记搜索 |
| [continuous.py](continuous.py) | 向候选库外提出新场景，并在实际查询后更新模型 |
| [config.py](config.py)、[build.py](build.py) | 本方法的训练设置，以及全来源和留类别模型的单独构建入口 |

完整实验矩阵由 [benchmark.build_collision_models](../../benchmark/build_collision_models.py) 构建，覆盖本方法及迁移基线。输入为 `data/collision/history/` 和 `data/collision/development/`，本方法的模型输出至 `results/gcrt/collision/models/`。

```powershell
python -B -m benchmark.build_collision_models core
python -B -m benchmark.build_collision_models complete
python -B -m pytest tests/gcrt -q -p no:cacheprovider
```

数据应先完成采集及资格检查。已有模型只有在代码、配置和输入档案一致时才会复用。正式有限候选与连续比较均有实际结果，报告见 [结果索引](../../results/gcrt/README.md)，执行步骤见 [实验运行说明](../../benchmark/README.md)。

局部历史经验的独立开发实验按[分阶段方案](../../docs/collision_discovery_plan.md)执行，主指标为碰撞数量及对全库碰撞的召回率。首轮冻结同一个历史编码器，仅改变核与来源权重，实际状态见[执行记录](../../docs/collision_discovery_execution.md)。该实验结果与原冻结矩阵分别保存。

[阶段响应方案](../../docs/collision_discovery_phase_response.md)只使用既有查询前缀做预测诊断。历史来源后验由角色隔离的历史档案和对应阶段标签构建；目标适配只用已经付费的阶段标签。该信息比二值碰撞反馈丰富，必须比较同信息目标模型。阶段原型尚未通过碰撞发现门槛，新的隔离在线接口、正式独立目标和留家族矩阵尚未执行。
