"""The only default experiment pipeline."""
from methods.ras_frt_uq.unified import train as train_ras
from .evaluate import evaluate
from .experiment import compare
from .prepare import prepare
from .train import train


def main():
    prepare()
    train()
    train_ras()
    compare()
    evaluate()


if __name__ == "__main__":
    main()
