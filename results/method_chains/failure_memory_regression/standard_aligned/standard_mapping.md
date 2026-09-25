# 场景与规范来源

本实验是规范启发的功能对齐研究，不是标准认证。物理参数范围为研究设置。

| template_id | functional_name | reference_standard | verified_parent_section | specific_clause_status | source_url | borrowed_behavior_skeleton | research_parameter_ranges | deviations |
|---|---|---|---|---|---|---|---|---|
| fbrt_cutin | 前方车辆切入 | GB/T 41798-2022；GB/T 47025-2026（仿真参考） | 6.4 周边车辆响应／6.5 自动紧急避险的功能框架 | 未核对具体子条款 | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=C3FD7FF23C6D06A9F7459DCD73E68905 | 相邻前车切入本车道；中汽研解读 https://www.castc.net/news/9807.cshtml | ((8.0, 50.0), (0.6, 3.0)) | highway-env 研究简化，非标准规定数值 |
| fbrt_lead_emergency_brake | 单车道前车紧急制动 | GB/T 41798-2022；GB/T 47025-2026（仿真参考） | 6.4 周边车辆响应／6.5 自动紧急避险的功能框架 | 未核对具体子条款 | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=C3FD7FF23C6D06A9F7459DCD73E68905 | 前车刹停；中汽研解读 https://www.castc.net/news/9807.cshtml | ((5.0, 120.0), (3.0, 8.0)) | highway-env 研究简化，非标准规定数值 |
| fbrt_stop_hold_go | 前车停保持起步 | GB/T 41798-2022；GB/T 47025-2026（仿真参考） | 6.4 周边车辆响应／6.5 自动紧急避险的功能框架 | 未核对具体子条款 | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=C3FD7FF23C6D06A9F7459DCD73E68905 | 前车停车、保持、起步；中汽研解读 https://www.castc.net/news/9807.cshtml | ((4.0, 35.0), (2.0, 5.0)) | highway-env 研究简化，非标准规定数值 |
| fbrt_cutout_static | 切出后静止车辆 | GB/T 41798-2022；GB/T 47025-2026（仿真参考） | 6.4 周边车辆响应／6.5 自动紧急避险的功能框架 | 未核对具体子条款 | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=C3FD7FF23C6D06A9F7459DCD73E68905 | 前车切出、露出静止车辆；中汽研解读 https://www.castc.net/news/9807.cshtml | ((8.0, 35.0), (1.5, 4.0)) | highway-env 研究简化，非标准规定数值 |
