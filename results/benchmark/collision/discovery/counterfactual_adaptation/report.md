# 虚拟反馈中的目标核适配：历史诊断

16个历史任务，64点相同反馈，各比较两个预定候选的两个虚拟标签。
192组全512点预测完成独立核对；没有新增开发搜索或物理调用。

| 场景 | 候选 | 窗口 | 更新 | 预测批量收益 | 实际分支批量碰撞潜力 | 剩余NLL |
|---|---|---:|---|---:|---:|---:|
| cut_in | greedy | 5 | fixed | 2.545 | 3.000 | 0.186161 |
| cut_in | greedy | 5 | refit_target | 2.586 | 3.000 | 0.186121 |
| cut_in | greedy | 5 | refit_both | 2.586 | 3.000 | 0.186121 |
| cut_in | greedy | 64 | fixed | 26.680 | 28.625 | 0.186161 |
| cut_in | greedy | 64 | refit_target | 26.884 | 28.750 | 0.186121 |
| cut_in | greedy | 64 | refit_both | 26.884 | 28.750 | 0.186121 |
| cut_in | entropy_excluding_greedy | 5 | fixed | 2.389 | 2.875 | 0.183515 |
| cut_in | entropy_excluding_greedy | 5 | refit_target | 2.414 | 2.875 | 0.182676 |
| cut_in | entropy_excluding_greedy | 5 | refit_both | 2.414 | 2.875 | 0.182677 |
| cut_in | entropy_excluding_greedy | 64 | fixed | 26.636 | 28.500 | 0.183515 |
| cut_in | entropy_excluding_greedy | 64 | refit_target | 26.672 | 29.000 | 0.182676 |
| cut_in | entropy_excluding_greedy | 64 | refit_both | 26.672 | 29.000 | 0.182677 |
| cut_out | greedy | 5 | fixed | 4.323 | 4.750 | 0.143557 |
| cut_out | greedy | 5 | refit_target | 4.350 | 4.750 | 0.143476 |
| cut_out | greedy | 5 | refit_both | 4.350 | 4.750 | 0.143475 |
| cut_out | greedy | 64 | fixed | 36.904 | 37.250 | 0.143557 |
| cut_out | greedy | 64 | refit_target | 37.111 | 37.250 | 0.143476 |
| cut_out | greedy | 64 | refit_both | 37.111 | 37.250 | 0.143475 |
| cut_out | entropy_excluding_greedy | 5 | fixed | 3.920 | 4.625 | 0.140878 |
| cut_out | entropy_excluding_greedy | 5 | refit_target | 3.994 | 4.500 | 0.139740 |
| cut_out | entropy_excluding_greedy | 5 | refit_both | 3.994 | 4.500 | 0.139740 |
| cut_out | entropy_excluding_greedy | 64 | fixed | 35.760 | 37.500 | 0.140878 |
| cut_out | entropy_excluding_greedy | 64 | refit_target | 37.057 | 37.375 | 0.139740 |
| cut_out | entropy_excluding_greedy | 64 | refit_both | 37.057 | 37.375 | 0.139740 |

完整重拟合相对固定参数，在32项两个候选/窗口比较中改变4项偏好。
实际分支批量潜力使用历史评估标签，一次查询后固定批量，不是128次自适应搜索的发现数。
本诊断只比较两个预定候选，没有对448个可查询候选执行完整前瞻选例。
基础教师见过全部历史对象，自身仅从来源后验排除；结果不作为严格留系统验证。
总耗时2409.17秒，与基线回放共享机器资源；重拟合及独立核对分项见report.json。
正式阶段、原失败结论与当前开发矩阵保持原状；没有从本项自动启动新的选例实验。
