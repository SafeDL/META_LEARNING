# Hierarchical Map-Aware Meta-Testing

`meta_testing` 是当前 active MVR 研究包：地图作为显式条件，完整测试轨迹作为 one-episode context token，潜变量仅描述未见 SUT 的 vulnerability，Outer scene policy 与 Inner adversarial SAC 共用地图/latent 表征。

当前只支持 IDM 与 rule-based controller profiles，用于 controller-profile POC；外部/学习型 ADS adapter 需要显式提供 controller 和 I/O contract 后才可注册。

```powershell
conda run -n metadrive python -m pytest meta_testing/tests -q
conda run -n metadrive python -m pytest pearl_learning/tests -q
```

训练必须按 `inner_pretrain → posterior → outer → joint` 顺序推进。评估使用总预算 20 个 simulator episodes，K=0/1/2/4 的 probe 成本全部计入预算；未通过 G1–G8 前不得输出方法性能结论。
