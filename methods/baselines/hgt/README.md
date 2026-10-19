# 历史引导高斯过程风险测试

历史引导高斯过程风险测试（HGT，History-Guided Testing）用历史近邻风险建立均值，用历史任务训练风险核，再根据当前控制器的实测反馈更新高斯过程并选择场景。

HGT 的基础核训练、历史数据读取和风险工具也供 GCRT、CG-BRT、DART、LRT 复用。

| 文件 | 用途 |
|---|---|
| [config.py](config.py)、[history.py](history.py) | 六个历史来源、数据读取与训练划分 |
| [kernel.py](kernel.py)、[train.py](train.py) | 风险核训练与模型读取 |
| [gp.py](gp.py)、[search.py](search.py) | 高斯过程更新与测试循环 |
| [baseline.py](baseline.py)、[archive.py](archive.py) | GP-EI、GP-UCB、RF-BO、随机测试等标准基线及场景档案 |
| [scenarios.py](scenarios.py)、[risk.py](risk.py)、[measurements.py](measurements.py) | 场景参数、风险定义与物理执行 |
| [oracle.py](oracle.py)、[io.py](io.py) | 查询接口和文件读写 |

历史记录位于 [data/history](../../../data/history/)，模型位于 [results/baselines/hgt](../../../results/baselines/hgt/README.md)。已有比较中的记录 ID 为 `frozen_original`，展示名称统一为 HGT。按类比较见 [LRT 按类实验](../../../results/baselines/lrt/by_family/README.md)。
