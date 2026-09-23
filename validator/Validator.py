import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from matplotlib.figure import Figure
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, RANK, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import (
    select_device,
    smart_inference_mode,
    unwrap_model,
)

from validator.coco_map import compute_coco_map


class CustomValidator(DetectionValidator):
    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        plot_settings={},
    ):
        super().__init__(dataloader, save_dir, args, _callbacks, plot_settings)
        self.class_metrics_csv = self.save_dir.joinpath("class_metrics.csv")
        self.epoch_class_metrics: list[tuple[str, float]] = []
        self.final_figures: list[Figure] = []
        self.coco_records: list[dict[str, Any]] = []

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (
                trainer.epoch == trainer.epochs - 1
            )
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml"):
                LOGGER.warning(
                    "WARNING ⚠️ validating an untrained model YAML will result in 0 mAP."
                )
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get(
                    "batch", 1
                )  # export.py models default to batch-size 1
                LOGGER.info(
                    f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})"
                )

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(
                    emojis(
                        f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"
                    )
                )

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = (
                    0  # faster CPU val as time dominated by inference, not dataloading
                )
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(
                self.data.get(self.args.split), self.args.batch
            )
            imgsz = self.dataloader.dataset.shape

            model.eval()
            model.warmup(
                imgsz=(1 if pt else self.args.batch, 3, imgsz[0], imgsz[1])
            )  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(batch_i, preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results(trainer)
            self.run_callbacks("on_val_end")

        if self.training:
            model.float()
            results = {
                **stats,
                **trainer.label_loss_items(
                    self.loss.cpu() / len(self.dataloader), prefix="val"
                ),
            }
            if getattr(self.args, 'coco_map', False):
                if not self.coco_records:
                    LOGGER.warning("coco_map requested but no records were "
                                   "collected - skipping pycocotools mAP.")
                else:
                    try:
                        coco = compute_coco_map(self.coco_records, self.data["names"])
                        results.update({
                            "coco/map50":    round(coco["map50"],    5),
                            "coco/map50_95": round(coco["map50_95"], 5),
                        })
                    except ImportError as e:
                        LOGGER.warning(f"coco_map requested but pycocotools is not "
                                       f"installed ({e}) - no COCO metrics reported. "
                                       f"It is listed in requirements.txt.")
                    except Exception as e:
                        LOGGER.warning(f"pycocotools mAP failed: {e}")
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            if getattr(self.args, 'coco_map', False):
                if not self.coco_records:
                    LOGGER.warning("coco_map requested but no records were "
                                   "collected - skipping pycocotools mAP.")
                else:
                    try:
                        coco = compute_coco_map(self.coco_records, self.data["names"])
                        LOGGER.info(
                            f"pycocotools  mAP50: {coco['map50']:.4f}  mAP50-95: {coco['map50_95']:.4f}"
                        )
                    except ImportError as e:
                        LOGGER.warning(f"coco_map requested but pycocotools is not "
                                       f"installed ({e}) - no COCO metrics reported. "
                                       f"It is listed in requirements.txt.")
                    except Exception as e:
                        LOGGER.warning(f"pycocotools mAP failed: {e}")
            return stats

    def print_results(self, trainer):  # type: ignore
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(
            pf
            % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results())
        )
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(
                f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels"
            )

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i)[:-1],
                    )  # type: ignore
                )
        elif self.args.verbose and self.training and self.nc > 1 and len(self.metrics.stats):
            # Saves training metrics to a CSV file
            keys: list[str] = ["Precision", "Recall", "mAP50", "mAP50:95", "F1"]
            n: int = len(keys) * self.nc + 1  # number of cols

            s: str = (
                ""
                if self.class_metrics_csv.exists()
                else (
                    (
                        "%s,"
                        * n
                        % tuple(
                            ["epoch"]
                            + [
                                f"{self.names[i]}_{metric}"
                                for i in self.names.keys()
                                for metric in keys
                            ]
                        )
                    ).rstrip(",")
                    + "\n"
                )
            )  # header
            with open(self.class_metrics_csv, "a", encoding="utf-8") as f:
                self.epoch_class_metrics = [
                    (f"{self.names[i]}_{metric_name}", class_metric)
                    for i in self.names.keys()
                    for metric_name, class_metric in zip(
                        keys, self.metrics.class_result(i)
                    )
                ]
                f.write(
                    s
                    + (
                        "%.6g,"
                        * n
                        % tuple(
                            [trainer.epoch + 1]
                            + [
                                class_metric
                                for _, class_metric in self.epoch_class_metrics
                            ]
                        )
                    ).rstrip(",")
                    + "\n"
                )  # type: ignore

        if self.args.plots:
            # Persist the raw matrix, not just the rendered figure, so matrices
            # from several seeds can be averaged afterwards.
            np.save(self.save_dir / "confusion_matrix.npy", self.confusion_matrix.matrix)
            for normalize in True, False:
                self.final_figures.append(
                    self.confusion_matrix.plot(
                        save_dir=self.save_dir,
                        normalize=normalize,
                        on_plot=self.on_plot,
                    )
                )

    def update_metrics(self, batch_i: int, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth.

        Args:
            batch_i (int): Batch index.
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            batch (Dict[str, Any]): Batch data containing ground truth.
        """
        if batch_i == 0:
            self.coco_records = []
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            stat = dict(
                conf=np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                pred_cls=np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                target_img=np.unique(cls),
                target_cls=cls,
                **self._process_batch(predn, pbatch)
            )
            self.metrics.update_stats(stat)
            self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)

            # BD2 record — reuse already-computed pbatch/predn/stat from this iteration
            gt_cls_bd2   = cls.astype(int)
            gt_boxes_bd2 = pbatch["bboxes"].cpu().numpy() if len(gt_cls_bd2) else np.zeros((0, 4), np.float32)
            if no_pred:
                self.coco_records.append({
                    "im_file":    batch["im_file"][si],
                    "gt_cls":     gt_cls_bd2,
                    "gt_boxes":   gt_boxes_bd2,
                    "pred_conf":  np.zeros(0, np.float32),
                    "pred_cls":   np.zeros(0, int),
                    "pred_boxes": np.zeros((0, 4), np.float32),
                    "tp_iou50":   np.zeros(0, bool),
                })
                continue
            self.coco_records.append({
                "im_file":    batch["im_file"][si],
                "gt_cls":     gt_cls_bd2,
                "gt_boxes":   gt_boxes_bd2,
                "pred_conf":  stat["conf"],
                "pred_cls":   stat["pred_cls"].astype(int),
                "pred_boxes": predn["bboxes"].cpu().numpy(),
                "tp_iou50":   stat["tp"][:, 0].astype(bool),
            })

            # Save
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (
            batch["img"].half() if self.args.half else batch["img"].float()
        ) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        return batch
