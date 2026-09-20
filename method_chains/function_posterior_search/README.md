# 功能后验搜索（Function Posterior Search）

本目录只保留通过独立物理确认的自适应贪心版本。该方法按功能维护去重后的源响应
假设，并仅用已经消耗预算获得的目标响应更新后验。每次查询优先选择后验临界事件
概率最高的场景；前 10 次查询保证六种功能均被覆盖。

噪声尺度从 `0.05/0.15/0.30` 中选择。每个场景种子只在六个源系统之间进行内部
留一验证，18 个目标系统的标签和响应均不参与调参。失效判据、预算、反馈权限及
比较方法与确认基准完全一致。

## 独立确认

确认实验包含五个新场景种子、每种子 600 个场景、六种功能，以及 18 个按
“源覆盖程度 × 功能异质性”组成的目标系统，共执行 41,000 个物理 episode。
预先声明的主指标是相对 Function-Conditioned DIVA 的 Recall@50。

- Posterior Search：42.60%
- Function-Conditioned DIVA：41.83%
- 配对差异：+0.78 个百分点
- 分层 bootstrap 95% 区间：+0.46 至 +1.16 个百分点

增益并不均匀：该方法在边界和范围外压力场景更强，在核心场景及部分功能上更弱。
因此证据支持“压力区域优先搜索”，不支持所有功能或运行区域上的普遍优越性。

## 文件与运行

- `benchmark.py`：确认实验的 SUT、场景因子和物理响应库；
- `search.py`：目标隐藏的后验更新和自适应查询；
- `confirmation.py`：唯一正式实验入口、基线、统计和报告；
- `tests/`：信息边界、场景设计、查询轨迹和结果重算检查。

```powershell
conda run -n metadrive python -B -m method_chains.function_posterior_search.confirmation
conda run -n metadrive python -B -m pytest method_chains/function_posterior_search/tests -q -p no:cacheprovider
```

正式结果位于
[`results/method_chains/function_posterior_search/confirmation/`](../../results/method_chains/function_posterior_search/confirmation/)。
开发阶段的向前规划、全局假设和重复源假设分支未获得正面结论，已连同开发结果
删除，不再作为兼容入口保留。
