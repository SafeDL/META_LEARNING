# META_LEARNING 最新训练化与代码清理计划

## 1. 当前版本判断

当前版本已经完成主要架构：

-   HPTR-style map encoder
-   episode-level trajectory context
-   PEARL-inspired latent posterior
-   Outer Hybrid PPO
-   Inner SAC
-   online few-shot adaptation
-   IDM behavioral profile based SUT simulation

下一阶段不再调整核心算法，而进入：

1.  可直接训练；
2.  可稳定评估；
3.  安全清理旧代码。

------------------------------------------------------------------------

## 2. 默认 GPU 配置

当前训练配置应默认使用 CUDA：

``` yaml
training:
  device: cuda
```

同时增加自动 fallback：

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

需要统一：

-   CUDA seed
-   checkpoint device compatibility
-   tensor device migration

------------------------------------------------------------------------

## 3. 正式训练脚本

### train_inner.py

目标：训练 adversarial SAC。

流程：

    MetaDrive rollout
          ↓
    InnerRiskReward
          ↓
    Replay Buffer
          ↓
    SAC update

训练：

-   map encoder
-   shared feature encoder
-   option embedding
-   inner SAC

------------------------------------------------------------------------

### train_posterior.py

目标：训练：

    q(z|C_K)

训练：

-   trajectory encoder
-   episode token builder
-   posterior
-   outcome decoder

------------------------------------------------------------------------

### train_outer.py

目标：训练：

    π_scene(x|g,z)

流程：

    OnlineMetaTest
          ↓
    OuterRolloutBuffer
          ↓
    GAE
          ↓
    PPO update

------------------------------------------------------------------------

### evaluate_meta_test.py

执行：

    z0
     ↓
    test 1
     ↓
    posterior update
     ↓
    z1
     ↓
    test 2

评价：

-   cumulative unique failures
-   failure discovery AUC
-   tests-to-first-failure
-   tests-to-5 failures

------------------------------------------------------------------------

## 4. 训练顺序

固定：

    Inner SAC pretraining
            ↓
    Posterior training
            ↓
    Outer PPO training
            ↓
    Optional light joint calibration

------------------------------------------------------------------------

## 5. 当前必须修复的问题

### State extractor

需要增加：

    raw simulator observation
            ↓
    canonical state extractor
            ↓
    fixed state vector

保证 Inner 输入契约稳定。

### CUDA pipeline

统一迁移：

-   map embedding
-   latent
-   trajectory token
-   outcome
-   mask

### Optimizer ownership

Inner stage：

    map_encoder
    shared_feature_encoder
    option_embedding
    inner_sac

Posterior stage：

    episode_token_builder
    posterior
    outcome_decoder

Outer stage：

    scene_policy

------------------------------------------------------------------------

## 6. 可以安全清理的旧代码

满足以下条件后删除：

-   无 active import
-   无 config 引用
-   有新模块替代
-   legacy 可通过 git history 获取

------------------------------------------------------------------------

## 7. 建议归档/删除

### 旧 PEARL 主实现

归档：

    pearl_learning/src/pearl_agent.py
    pearl_learning/src/pearl_trainer.py
    pearl_learning/src/context_encoder.py
    pearl_learning/src/scenario_encoder.py

原因：

已被 meta_testing 替代。

------------------------------------------------------------------------

### 旧 merge-only task 分支

删除：

-   merge-only task spec
-   merge-only scenario encoder
-   transition-level context sampler

------------------------------------------------------------------------

### 历史探索代码

归档：

    moe.py
    FiLM experiments
    logical_order experiments
    old causal audits
    old posterior diagnostics

这些不是最终方法。

------------------------------------------------------------------------

## 8. 必须保留

保留：

    scenario/
    map/
    route/
    critical/
    checkpoint/
    provenance/
    meta_testing/

这些属于当前方法基础设施。

------------------------------------------------------------------------

## 9. 最终冻结方法

保持：

    HPTR Map Encoder
            +
    q(z|C) latent inference
            +
    Outer Hybrid PPO
            +
    Inner SAC

其中：

-   Meta-learning 来自 latent inference；
-   PPO/SAC 是不同时间尺度优化器；
-   IDM profiles 用于当前 SUT heterogeneity 验证；
-   最终目标是在固定 simulator budget 下发现更多 unique failures。

下一阶段重点：

    正确架构
        ↓
    真实训练闭环
        ↓
    可靠实验
        ↓
    安全清理旧代码
