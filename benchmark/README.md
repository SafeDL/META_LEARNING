# 实验设置与评价工具

本目录统一管理场景测试集、共同初始化、风险反馈和耗时统计。当前 GCRT 使用种子 11、每类最多 200 次测试，前 10 次共同初始化计费。

| 文件 | 用途 |
|---|---|
| [config.py](config.py)、[pools.py](pools.py) | 驾驶控制器、场景参数、采样点与共用设置 |
| [data.py](data.py) | 定位历史数据、场景响应和逐场景执行耗时 |
| [initialization.py](initialization.py) | 生成不依赖风险标签的共同初始化 |
| [risk_channel.py](risk_channel.py) | 向测试方法逐次返回已查询场景的风险 |
| [timing.py](timing.py) | 记录初始化、场景选择、反馈更新与最终停止判断的耗时 |
| [evaluation.py](evaluation.py)、[baselines.py](baselines.py) | 计算发现数量和覆盖率，运行标准搜索基线 |
| [prepare.py](prepare.py)、[interface_audit.py](interface_audit.py)、[validate_banks.py](validate_banks.py) | 采集实测响应，核验控制器接口与数据 |

实测数据见 [data](../data/README.md)，共用核验与标准基线结果见 [results/benchmark](../results/benchmark/README.md)。GCRT 的构建和评价入口见[方法说明](../methods/gcrt/README.md)，其专用参数在 [methods/gcrt/config.py](../methods/gcrt/config.py)。
