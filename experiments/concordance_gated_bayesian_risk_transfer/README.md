# CG-BRT 实验

固定设置：六个被测系统、两个 2048 点场景池、五个种子（11、23、37、53、71）。每条轨迹预算 200，含 10 次初始化，共 60 条主方法轨迹。

| 文件 | 作用 |
|---|---|
| [evaluate.py](evaluate.py) | 运行主方法；已有轨迹复用，缺失时在锁定响应池上选例，逐次返回风险 |
| [compare.py](compare.py) | 重算已有方法的高风险与碰撞指标，写入比较表 |
| [verify.py](verify.py) | 核对反馈、来源权重、门控更新与冻结源码 |
| [config.py](config.py) | 结果路径、种子和指标设置 |

在项目根目录运行：

```powershell
conda activate metadrive
python -B -m experiments.concordance_gated_bayesian_risk_transfer.compare
python -B -m experiments.concordance_gated_bayesian_risk_transfer.verify
```

这两条命令只处理已有记录。当前比较包括 16 个方法的五种子结果和 9 个方法的种子 11 补充结果，共 1056 个唯一轨迹文件。

方法见[实现说明](../../methods/concordance_gated_bayesian_risk_transfer/README.md)，指标与结果见[结果目录](../../results/concordance_gated_bayesian_risk_transfer/README.md)。
