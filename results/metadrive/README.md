# MetaDrive 实验结果

本目录保存独立的 MetaDrive Risk Mining 和 Formal Teacher 实验记录，不作为当前 Highway-env FBRT 回放输入。

| 目录 | 内容 |
| --- | --- |
| `mining/cutin_g01/` | Risk Mining 的源场景观测、先验模型、LOSO 评价、门槛结果和可视化；其中源观测与 LOSO 文件由 `metadrive_sim_env/configs/formal_teacher.yaml` 读取 |
| `formal_teacher/cutin_g01/formal_teacher/` | Formal Teacher 的校准、诊断、门槛评价和运行清单 |

当前 Formal Teacher 结果在第一阶段停止，`stage1_gate.json` 的决策为 `stop_before_transformer`，教师门槛未通过。这是该研究链的阴性结果记录，不代表 FBRT 方法效果。若继续复核 MetaDrive 研究，需保留 `mining/` 的源数据和这些评价记录。
