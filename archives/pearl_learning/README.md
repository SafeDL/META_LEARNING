# PEARL 逻辑场景元强化学习（Legacy Baseline）

> 此包已冻结为 merge-only legacy baseline。新的 active MVR 方法位于 [`mvr/`](../../mvr/)；不要在此包追加新方法功能。

本模块实现 PEARL-SAC 的逻辑汇入场景适应。当前代码已按原论文的概率上下文变量表述重新校验；历史 checkpoint 和历史性能结果已退役，必须使用当前 `pearl_checkpoint` 与 `transition_product_recent_context` 方法契约重新训练。

## 方法契约

- 任务潜变量为对角高斯 `q(z|c)`；空 context 严格返回单位高斯先验。
- 默认 `context_aggregation: transition_product`：每条 `(s,a,r,s')` 转移独立产生一个高斯因子，再按精度相加形成 Product-of-Gaussians 后验。episode 分组不改变后验。
- `episode_product` 仅是显式消融，不是默认 PEARL 实现。
- encoder context 只从每任务的 recent context buffer 采样；该窗口最多保留 16 个完整 episode，并每 1,000 次优化更新清空刷新。
- SAC 的 RL batch 从长期 task replay 采样，并排除本次 context 使用的全部 episode，防止同批证据泄漏。
- 任务采集和优化 meta-batch 均无放回采样，同一 meta-batch 不重复任务。
- 新任务适应只更新后验，不更新网络参数；support 与 query case 完全分离。

## 评估契约

默认 K 为 `0/1/2/4/8`。每个 K 必须成对报告：

- `posterior_mean_deterministic`：使用后验均值和确定性条件策略，作为低方差主评估及 checkpoint 选择口径；
- `posterior_sampled`：每个 query case 从后验按可复现独立种子采样一次 latent，整集固定 latent，并使用确定性条件策略，用于显示后验不确定性对行为的影响。

测试域严格拆分：

- `meta_test_template` → `id_known_logical_type`：逻辑类型已见、模板/几何留出；
- `meta_test_logical` → `ood_unseen_logical_type`：逻辑类型未见；
- `meta_test_all` 一次生成两个 regime 和两种 query mode 的统一 suite，禁止把 ID/OOD 合并成一个平均数。

无效事件奖励采用可验证下界。当前 horizon=180、每步最大正奖励 1.1、目标奖励 200、margin=1，因此非目标碰撞、出界和错误路线惩罚均须至少为 `200 + 180×1.1 + 1 = 399`；配置取 400，环境创建时会 fail fast。

## 运行

环境：

```powershell
conda run -n metadrive python -m pytest archives/pearl_learning/tests -q
```

构建冻结输入：

```powershell
conda run -n metadrive python -m archives.pearl_learning.scripts.build_taskbook --config archives/pearl_learning/configs/posterior_adaptation_protocol.yaml --output results/pearl_learning/posterior_adaptation/taskbooks
```

该命令同时在 taskbook 的父目录生成 `casebooks/`。

训练入口为 `archives/pearl_learning/scripts/train_pearl.py`。正式训练必须完成配置规定的多个训练种子和 validation freeze；smoke 只能验证流程，不能替代正式训练。

配对 ID/OOD 评估示例：

```powershell
conda run -n metadrive python -m archives.pearl_learning.scripts.evaluate_fewshot --config archives/pearl_learning/configs/posterior_adaptation_protocol.yaml --checkpoint <checkpoint> --taskbook results/pearl_learning/posterior_adaptation/taskbooks --casebook-root results/pearl_learning/posterior_adaptation --split meta_test_all --run-name <run-name> --validation-freeze-manifest <freeze.json>
```

当前结果状态见 [`../../results/pearl_learning/README.md`](../../results/pearl_learning/README.md)。
