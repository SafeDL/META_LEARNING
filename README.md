# 历史响应驱动的跨系统碰撞场景发现

复用历史驾驶系统的测试经验，结合新系统的少量反馈，在有限预算内发现碰撞场景。

## 环境安装

```powershell
conda env create -f environment.yml
conda activate metadrive
```

以下命令均在项目根目录执行。

## 代码检查

```powershell
python -B -m pytest tests simulator/metadrive/tests -q -p no:cacheprovider
uv tool run --from ruff ruff check .
uv tool run --from ruff ruff format . --check
```

## 使用入口

- [方法训练与运行](testing_methods/history_transfer/README.md)
- [实验运行指导](experiments/README.md)
- [已有实验产物](experiment_outputs/README.md)
- [代码与写作规范](docs/style.md)
