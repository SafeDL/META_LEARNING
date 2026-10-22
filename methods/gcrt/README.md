# 基于历史测试结果的碰撞场景选择

本方法利用历史驾驶系统的测试结果，估计新系统在哪些场景中容易碰撞，并逐次选择下一次测试。研究问题、系统名单、场景和指标见 [项目说明](../../README.md)。`gcrt` 保留为代码目录名，当前实现采用固定查询预算。

训练时，轮流把一个历史系统当作“暂时未知的新系统”，用少量已知结果模拟适配，再检查剩余场景的预测。这个模拟目标自己的结果不能混入历史上下文。碰撞标签是主要监督，间距和 TTC 用于辅助学习场景关系；独立开发系统用于选择模型。

正式测试时，场景编码器保持不变。每获得一次真实反馈，就更新碰撞概率及各历史来源的参考权重；另有一个只依靠目标反馈和物理参数的分量。选例时同时考虑当前碰撞概率，以及测完这个场景对后续选择的帮助，窗口最多为 5 次查询。

该选例规则属于 [ENS 主动搜索](https://proceedings.mlr.press/v70/jiang17d.html) 的条件收益近似结构，使用本项目的历史后验和有限窗口；它与历史表示、目标适配的作用需要分别比较。目前 ENS 与 RGPE 的独立基线尚未实现。

| 文件 | 作用 |
|---|---|
| [train.py](train.py) | 从历史和开发档案训练场景表示与碰撞模型 |
| [prior.py](prior.py) | 组合历史来源预测及目标自身的预测分量 |
| [posterior.py](posterior.py) | 根据新的碰撞或未碰撞标签更新概率与来源权重 |
| [selection.py](selection.py) | 比较下一次测试的当前收益和后续预期收益 |
| [continuous.py](continuous.py) | 向候选库外提出新场景，并在实际查询后更新模型 |
| [config.py](config.py)、[build.py](build.py) | 本方法的训练设置，以及全来源和留类别模型的单独构建入口 |

完整实验矩阵由 [benchmark.build_collision_models](../../benchmark/build_collision_models.py) 构建，覆盖本方法及迁移基线。输入为 `data/collision/history/` 和 `data/collision/development/`，本方法的模型输出至 `results/gcrt/collision/models/`。

```powershell
python -B -m benchmark.build_collision_models core
python -B -m benchmark.build_collision_models complete
python -B -m pytest tests/gcrt -q -p no:cacheprovider
```

数据应先完成采集及资格检查。已有模型只有在代码、配置和输入档案一致时才会复用。正式有限候选与连续比较均有实际结果，报告见 [结果索引](../../results/gcrt/README.md)，执行步骤见 [实验运行说明](../../benchmark/README.md)。
