from ..training.stages import TrainingStage
from .training_cli import run


if __name__ == "__main__":
    run(TrainingStage.INNER_PRETRAIN)
