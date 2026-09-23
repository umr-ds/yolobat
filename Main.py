import argparse

import yaml
from ultralytics.cfg import smart_value

from trainer.Trainer import CustomTrainer


DEFAULT_CFG = '/yolobat-train/files/cfg.yaml'


def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg', default=DEFAULT_CFG,
                        help='training config to load')
    parser.add_argument('--set', dest='kv_pairs', nargs='*', default=[],
                        help='Key-value pairs in the format key=value')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    overrides = load_cfg(args.cfg)
    for arg in args.kv_pairs:
        k, v = arg.split('=')
        overrides[k] = smart_value(v)
    trainer = CustomTrainer(cfg=None, overrides=overrides)

    trainer.train()
