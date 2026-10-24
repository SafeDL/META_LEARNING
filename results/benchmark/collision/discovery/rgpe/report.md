# RGPE碰撞分类开发比较

完整80项、10,240次等效查询通过独立审计，新增物理调用为零。最强既有参考为`physical_local`。

| 方法 | 平均去重碰撞数 | 平均召回率 | 发现曲线AUC |
|---|---:|---:|---:|
| rgpe_classifier | 36.375000 | 95.171525% | 22.797070 |
| physical_rgpe_classifier | 36.725000 | 96.436841% | 22.801172 |
| physical_local | 37.875000 | 98.585721% | 24.098828 |
| residual_local | 37.700000 | 98.304101% | 24.227344 |

完整预算点、曲线、配对分层区间及分项成本见`report.json`。该比较不替换已失败的指定主方法，不自动通过阶段B或启动C/D/E。

原论文为连续响应GP与期望改进；这里采用Bernoulli/Laplace、不同标签对、共享FSBO表示及碰撞概率贪心，详细差异见`methods/baselines/rgpe/README.md`。
