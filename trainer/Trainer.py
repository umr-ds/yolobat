import json

from ultralytics.models import YOLO
import torch
import yaml
from pathlib import Path
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import (
    LOCAL_RANK,
    LOGGER,
    RANK,
    callbacks,
)
from torch.utils.data import distributed
from torch import distributed as dist
from Datasets import YoloDataset
from ultralytics.cfg import get_cfg
import torch.nn as nn
import math
from copy import copy

from util.taxonomy import load_taxonomy, species_to_group
from validator.Validator import CustomValidator

from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_imgsz

from ultralytics.utils.torch_utils import (
    init_seeds,
    strip_optimizer,
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
)
import warnings
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    module='librosa'
)
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module='audiomentations',
)


class CustomTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides='files/cfg.yaml'):
        """
        Custom trainer, for yolo.
        Creates yaml file with classes, is needed for yolo to work.
        :param cfg: needed for multi gpu training with ddp
        :param overrides: yolo trainer settings
        """
        # get configuration
        overrides = get_cfg(overrides)

        dataset_name = overrides.dataset_name

        x_size = math.floor(math.ceil((overrides.sr / overrides.hop_length) * overrides.analyze_length) / 32) * 32
        low_cutoff_bins = 0
        if overrides.cut_low_freq:
            low_cutoff_bins = math.floor(overrides.cut_low_freq / ((overrides.sr / 2) / (overrides.n_fft / 2 + 1)))
        y_size = math.floor(((overrides.n_fft / 2 + 1) - low_cutoff_bins) / 32) * 32
        overrides.imgsz = [y_size, x_size]

        self.data_path = f'var/data_exports/{dataset_name}_dataset.json'
        self.training_files = f'var/data_exports/{dataset_name}_train.txt'
        self.validation_files = f'var/data_exports/{dataset_name}_val.txt'
        init_seeds(overrides.seed + 1 + RANK, deterministic=overrides.deterministic)

        for path in (self.data_path, self.training_files, self.validation_files):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found. Put the annotation file and the two split "
                    f"lists under var/data_exports/, named after dataset_name in "
                    f"files/cfg.yaml; see the Data section of README.md."
                )

        with open(self.data_path, 'r') as file:
            data = json.load(file)

        # which classes belong together: the loss smooths within a group, the
        # plots give each group its own color
        taxonomy = load_taxonomy()
        overrides.smoothing_groups = taxonomy["smoothing_groups"]
        class_group = species_to_group(taxonomy["report_groups"])
        palette = ["red", "blue", "green", "purple", "orange", "brown", "cyan",
                   "magenta", "olive", "teal"]
        group_colors = {group: palette[i % len(palette)]
                        for i, group in enumerate(taxonomy["report_groups"])}

        classes = {}
        class_map = {}
        class_linestyles = ["-", "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1)), (0, (3, 10, 1, 10, 1, 10))] #["-", "--", ":", "-", "-", "--", "-", "-", "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1)), (0, (3, 10, 1, 10, 1, 10)), "-", "--", ":", "-.", (0, (3, 5, 1, 5)), "-", "--"]

        self.plot_settings = []

        if overrides.single_cls:
            LOGGER.info("Overriding class names with single class.")
            classes = {0: "bat"}
            class_map = {0: 0}
            self.data = {}
            self.data["names"] = classes
            self.data["nc"] = 1
        else:
            ls = 0  # index for cycling through linestyles
            current_group = None
            for i, x in enumerate(data['categories']):
                group = class_group.get(x['name'])
                if group == current_group:
                    ls += 1
                else:
                    current_group = group
                    ls = 0
                classes[i] = x['name']
                class_map[x['id']] = i
                self.plot_settings.append({
                    'name': x['name'],
                    'color': group_colors.get(group, 'black'),
                    'line_style': class_linestyles[ls % len(class_linestyles)],
                })
            self.data = {
                "nc": len(classes),
                "names": classes,
            }

        if not overrides.data:
            with open('classes.yaml', 'w') as file:
                yaml.dump({'path': '/data/bats/'}, file, default_flow_style=False)
                yaml.dump({'train': '.'}, file, default_flow_style=False)
                yaml.dump({'val': '.'}, file, default_flow_style=False)
                yaml.dump({'test': '.'}, file, default_flow_style=False)

                yaml.dump({'names': classes}, file, default_flow_style=False)

            overrides.data = 'classes.yaml'

        if overrides.get('workers', None):
            self.workers = overrides.get('workers')
        else:
            self.workers = 0

        super().__init__(overrides=overrides.__dict__)

    def preprocess_batch(self, batch):
        """Moves the spectrograms to the device and scales them to [0, 1]."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def get_dataset(self) -> tuple[YoloDataset, YoloDataset]:
        train_set = YoloDataset(
            annotation_file=self.data_path,
            filter_file=self.training_files,
            augment=True,
            hyp=self.args,
            audio_path="/data/bats/"
        )

        val_set = YoloDataset(
            annotation_file=self.data_path,  # file path to annotation file
            filter_file=self.validation_files,
            # file path to a file, which has the names of audio files, which should be used in the dataset
            augment=False,
            hyp=self.args,
            audio_path="/data/bats/"
        )
        return train_set, val_set

    def get_dataloader(self, rank=0, mode="train"):  # type: ignore
        return self.create_dataloader(rank, mode)

    def create_dataloader(self, rank=None, mode='train') -> InfiniteDataLoader:
        if mode == 'val':
            dataset: YoloDataset = self.data["val"]  # type: ignore
            shuffle = False
        else:
            dataset = self.data["train"]
            shuffle = True
        shuffle = shuffle and not self.args.oversampling
        sampler = None
        if rank is not None and rank != -1:
            sampler = distributed.DistributedSampler(dataset, rank=rank, shuffle=shuffle)  # num_replicas=1

        def collate_fn_wrapper(batch, batch_index=None):
            if batch_index is None:
                batch_index = [0]
            result = custom_collate_fn(batch)
            batch_index[0] += 1
            return result

        return InfiniteDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=self.workers,
            sampler=sampler,
            collate_fn=collate_fn_wrapper,
            pin_memory=True
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build DetectionModel. The spectrogram always has three channels: the dB
        spectrogram plus the two positional encodings, or the spectrogram replicated
        three times when pos_enc is off."""
        from ultralytics.nn.tasks import DetectionModel
        n_channels = 3
        model = DetectionModel(cfg or self.args.model, nc=self.data["nc"], ch=n_channels, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self, mode='val'):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CustomValidator(
            self.val_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks,
            plot_settings=self.plot_settings
        )

    def export(self):
        best = YOLO(self.best)
        best.overrides = self.args.__dict__
        best.export(format="onnx", imgsz=self.args.imgsz, batch=1, simplify=True, int8=False)
        best.export(format="ncnn", imgsz=self.args.imgsz, batch=1, simplify=True, int8=False)
        best.export(format="openvino", imgsz=self.args.imgsz, batch=1, simplify=True, int8=True, dataset=self.trainset)
        best.export(format="openvino", imgsz=self.args.imgsz, batch=1, simplify=True, int8=False, dataset=None)

    def _setup_train(self):
        """Builds dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        # type: ignore
        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=2)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(
                model=self.model,
                imgsz=self.args.imgsz,  # type: ignore
                amp=self.amp,
                batch=self.batch_size,
            )

        # Dataloaders
        self.train_loader = self.get_dataloader(rank=LOCAL_RANK, mode="train")
        LOGGER.info(f"{len(self.train_loader.dataset)} training samples")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.val_loader = self.get_dataloader(rank=-1, mode="val")
            self.validator = self.get_validator('val')
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")  # type: ignore
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots and not self.args.snippets:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(
            len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs  # type: ignore
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # type: ignore # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)  # type: ignore
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)


def custom_collate_fn(batch):
    im_files = [item['im_file'] for item in batch]
    ori_shapes = [item['ori_shape'] for item in batch]
    resized_shapes = [item['resized_shape'] for item in batch]
    ratio_pads = [item['ratio_pad'] for item in batch]

    imgs = torch.stack([item['img'] for item in batch])

    classes_list = []
    bboxes_list = []
    batch_idx_list = []
    for i, item in enumerate(batch):
        n_bboxes = len(item['bboxes'])
        classes_list.append(item['cls'])
        bboxes_list.append(item['bboxes'])
        batch_idx_list.append(torch.full((n_bboxes,), i, dtype=torch.float32))

    try:
        classes = torch.cat(classes_list, dim=0)
        bboxes = torch.cat(bboxes_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)
    except Exception:
        classes_list = [c.squeeze(-1) if c.ndim == 2 else c for c in classes_list]
        classes = torch.cat(classes_list, dim=0)
        bboxes = torch.cat(bboxes_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)

    return {
        'im_file': im_files,
        'ori_shape': ori_shapes,
        'resized_shape': resized_shapes,
        'ratio_pad': ratio_pads,
        'img': imgs,
        'cls': classes,
        'bboxes': bboxes,
        'batch_idx': batch_idx
    }