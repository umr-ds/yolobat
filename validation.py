"""Re-run validation on a finished run, without retraining it.

Rebuilds the trainer from the run's own args.yaml, so the model is evaluated
exactly as it was during training, and writes the metrics, the plots and
confusion_matrix.npy into the run directory.

    python validation.py runs/detect/bats_experiments/yolobat/<run>
"""
import argparse
from pathlib import Path

from trainer.Trainer import CustomTrainer


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path,
                        help="run directory holding args.yaml and weights/best.pt")
    parser.add_argument("--weights", default="best.pt",
                        help="checkpoint inside <run>/weights. Default: best.pt")
    args = parser.parse_args()

    cfg = args.run / "args.yaml"
    weights = args.run / "weights" / args.weights
    for path in (cfg, weights):
        if not path.exists():
            parser.error(f"{path} does not exist")

    trainer = CustomTrainer(cfg=None, overrides=str(cfg))
    trainer._setup_train()

    # Call the validator directly rather than final_eval(), which strips the
    # optimizer state out of the checkpoint it is given.
    validator = trainer.validator
    validator.args.plots = trainer.args.plots
    metrics = validator(model=weights)
    metrics.pop("fitness", None)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
