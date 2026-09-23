"""
Evaluate a yolobat checkpoint with the four metrics from Mac Aodha et al. (2022).

  AP Det    — detection AP, TP = predicted start time within 10 ms of GT start time
  mAP Class — mean per-class AP, sorted by class confidence
  Top Class — AP where TP requires time match AND predicted class == GT class
  mAP File  — file-level species-presence AP, score = max confidence per class per file

AP uses PASCAL VOC all-points interpolation (identical to original BD2 code).
Matching uses 10 ms temporal start-time proximity (as specified in the paper),
NOT bounding-box IoU.

--coco-map adds mAP50 and mAP50-95 over the bounding boxes, through pycocotools;
these are the overlap-based numbers the paper reports. Ultralytics' own mAP is
printed alongside them and scores about two points higher.

Usage:
    python evaluate.py --config runs/detect/bats_experiments/yolobat/<run>/args.yaml
    python evaluate.py --config <run>/args.yaml --coco-map
"""

import argparse
import json
import sys
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trainer.Trainer import CustomTrainer
from validator.Validator import CustomValidator
from validator.coco_map import compute_coco_map

TIME_MATCH_S = 0.010  # 10 ms


# ─────────────────────────────────────────────────────────────────────────────
# AP — identical to batdetect2.evaluate.metrics.common
# ─────────────────────────────────────────────────────────────────────────────

def _average_precision(y_true: np.ndarray, y_score: np.ndarray, n_gt: int) -> float:
    if n_gt == 0 or len(y_true) == 0:
        return float("nan")
    sort_idx = np.argsort(-y_score)
    y_true = y_true[sort_idx].astype(float)
    tp_cum = np.cumsum(y_true)
    fp_cum = np.cumsum(1 - y_true)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)
    mprec = np.hstack((0, precision, 0))
    mrec = np.hstack((0, recall, 1))
    for i in range(mprec.shape[0] - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    inds = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(((mrec[inds] - mrec[inds - 1]) * mprec[inds]).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Time-proximity matching (greedy, highest-confidence first)
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_time_match(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    pred_cls: np.ndarray | None = None,
    gt_cls: np.ndarray | None = None,
) -> np.ndarray:
    """TP mask for predictions already sorted by confidence descending."""
    tp = np.zeros(len(pred_times), dtype=bool)
    if len(gt_times) == 0:
        return tp
    matched: set[int] = set()
    for i, pt in enumerate(pred_times):
        diffs = np.abs(gt_times - pt)
        mask = diffs <= TIME_MATCH_S
        if pred_cls is not None and gt_cls is not None:
            mask &= gt_cls == pred_cls[i]
        cands = np.where(mask)[0]
        free = [c for c in cands if c not in matched]
        if free:
            best = free[int(np.argmin(diffs[free]))]
            tp[i] = True
            matched.add(best)
    return tp


# ─────────────────────────────────────────────────────────────────────────────
# Validator — stores start times (seconds) instead of full boxes
# ─────────────────────────────────────────────────────────────────────────────

class PaperMetricsValidator(CustomValidator):
    """Extends CustomValidator to collect per-image data for paper metrics."""

    def __init__(self, *args, clip_duration_s: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_duration_s = clip_duration_s
        self.records: list[dict[str, Any]] = []

    def init_metrics(self, model):
        super().init_metrics(model)
        # AutoBackend when validating standalone, DetectionModel during training
        inner = getattr(model, "model", model)   # AutoBackend → DetectionModel
        if hasattr(inner, "model"):
            inner = inner.model                  # DetectionModel → nn.Sequential
        self._head = inner[-1]

    def postprocess(self, preds):
        """Return per-image dicts with NMS survivors AND all pre-NMS anchors above threshold."""
        from ultralytics.utils import nms as _nms
        if self.end2end:
            # _inference recovers the pre-NMS (B, 4+nc, num_anchors) tensor
            y   = preds[0]                                               # (B, max_det, 6)
            pre = self._head._inference(preds[1]["one2one"])             # (B, 4+nc, num_anchors)
            pre = pre.permute(0, 2, 1)                                   # (B, num_anchors, 4+nc)
            result = []
            for bi in range(y.shape[0]):
                surv  = y[bi]
                surv  = surv[surv[:, 4] > self.args.conf][: self.args.max_det]
                pre_i = pre[bi]                                          # (num_anchors, 4+nc)
                filt  = pre_i[:, 4:].amax(1) > self.args.conf
                pre_f = pre_i[filt]
                result.append({
                    "bboxes":         surv[:, :4],
                    "conf":           surv[:, 4],
                    "cls":            surv[:, 5],
                    "extra":          surv[:, 6:],
                    "all_boxes":      pre_f[:, :4],      # decoded xyxy, all anchors ≥ conf
                    "all_cls_scores": pre_f[:, 4:],      # sigmoid class scores, (n_pre, nc)
                })
            return result
        else:
            # non_max_suppression converts the box columns in place, so the second
            # call below would see xyxy where it expects xywh: clone first.
            raw_xywh = (preds[0] if isinstance(preds, (list, tuple)) else preds).clone()

            outputs = _nms.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                nc=0 if self.args.task == "detect" else self.nc,
                multi_label=True,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det,
                end2end=False,
                rotated=self.args.task == "obb",
            )

            # Pool for mAP Class / mAP File, thinned by the same class-aware NMS:
            # a one2many head fires many anchors per call, and BD2 allows one match
            # per ground truth, so the rest would count as false positives.
            suppressed = _nms.non_max_suppression(
                raw_xywh,
                self.args.conf,
                self.args.iou,
                nc=0 if self.args.task == "detect" else self.nc,
                multi_label=True,
                # pinned to the checkpoint's setting: --agnostic-nms here would
                # change the metric rather than the model
                agnostic=getattr(self.args, "pool_agnostic",
                                 self.args.single_cls or self.args.agnostic_nms),
                max_det=30000,
                end2end=False,
                rotated=self.args.task == "obb",
                return_cls_scores=True,
            )

            result = []
            for bi, x in enumerate(outputs):
                sup_i = suppressed[bi]
                result.append({
                    "bboxes":         x[:, :4],
                    "conf":           x[:, 4],
                    "cls":            x[:, 5],
                    "extra":          x[:, 6:],
                    "all_boxes":      sup_i[:, :4],
                    "all_cls_scores": sup_i[:, 6:6 + self.nc],
                })
            return result

    def update_metrics(self, batch_i, preds, batch):
        # confusion_matrix.process_batch iterates ALL keys and applies the same conf
        # mask to every value — but all_boxes/all_cls_scores have a different length
        # than the NMS survivors. Pop them before calling super, use the stash directly.
        pre_nms_stash = [
            {"all_boxes": pred.pop("all_boxes"), "all_cls_scores": pred.pop("all_cls_scores")}
            for pred in preds
        ]

        super().update_metrics(batch_i, preds, batch)

        for si, pred in enumerate(preds):
            pbatch  = self._prepare_batch(si, batch)
            predn   = self._prepare_pred(pred)
            pre_nms = pre_nms_stash[si]

            gt_cls = pbatch["cls"].cpu().numpy().astype(int)

            img_w = float(pbatch["ori_shape"][1]) if "ori_shape" in pbatch \
                    else float(batch["img"].shape[-1])

            if len(gt_cls) > 0:
                gt_boxes = pbatch["bboxes"].cpu().numpy()  # xyxy pixel space
                gt_x1_s = gt_boxes[:, 0] / img_w * self.clip_duration_s
            else:
                gt_x1_s = np.zeros(0, dtype=np.float32)

            # NMS survivors — used for AP Det and Top Class
            nc = self.nc
            if len(predn["cls"]) > 0:
                pred_conf  = predn["conf"].cpu().numpy()
                pred_cls   = predn["cls"].cpu().numpy().astype(int)
                pred_boxes = predn["bboxes"].cpu().numpy()
                pred_x1_s  = pred_boxes[:, 0] / img_w * self.clip_duration_s
            else:
                pred_conf = np.zeros(0, dtype=np.float32)
                pred_cls  = np.zeros(0, dtype=int)
                pred_x1_s = np.zeros(0, dtype=np.float32)

            # All pre-NMS anchors above threshold — used for mAP Class and mAP File
            all_boxes_t = pre_nms["all_boxes"]
            if len(all_boxes_t) > 0:
                all_x1_s      = all_boxes_t[:, 0].cpu().numpy() / img_w * self.clip_duration_s
                all_cls_scores = pre_nms["all_cls_scores"].cpu().numpy()   # (n_pre, nc)
            else:
                all_x1_s      = np.zeros(0, dtype=np.float32)
                all_cls_scores = np.zeros((0, nc), dtype=np.float32)

            self.records.append({
                "im_file":        batch["im_file"][si],
                "gt_cls":         gt_cls,
                "gt_x1_s":        gt_x1_s,
                "pred_conf":      pred_conf,
                "pred_cls":       pred_cls,
                "pred_x1_s":      pred_x1_s,
                "all_x1_s":       all_x1_s,
                "all_cls_scores": all_cls_scores,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Four paper metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_paper_metrics(
    records: list[dict[str, Any]],
    class_names: dict[int, str],
    yolo_stats: dict,
    clip_duration_s: float = 1.0,
    ignore_start_end: float = 0.01,
) -> dict:
    nc = len(class_names)
    names_list = [class_names[i] for i in range(nc)]

    # ── apply ignore_start_end filter per clip, then flatten ──────────────────
    lo = ignore_start_end
    hi = clip_duration_s - ignore_start_end

    def _cat_filtered(key, mask_key, dtype=None):
        arrs = []
        for r in records:
            mask = r[mask_key]
            vals = r[key][mask]
            if len(vals) > 0:
                arrs.append(vals)
        out = np.concatenate(arrs) if arrs else np.zeros(0)
        return out.astype(dtype) if dtype else out

    for r in records:
        r["_pred_mask"] = (r["pred_x1_s"] >= lo) & (r["pred_x1_s"] <= hi)
        r["_gt_mask"]   = (r["gt_x1_s"]   >= lo) & (r["gt_x1_s"]   <= hi)

    # match within each clip: every clip shares the same [0, clip_duration] axis,
    # so flattening first would match across clips

    det_conf_list:  list[np.ndarray] = []
    tp_det_list:    list[np.ndarray] = []
    tp_top_list:    list[np.ndarray] = []
    cls_conf_lists: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in range(nc)}

    all_gt_cls_list: list[np.ndarray] = []
    all_gt_t_list:   list[np.ndarray] = []

    for r in records:
        pm = r["_pred_mask"]
        gm = r["_gt_mask"]

        conf    = r["pred_conf"][pm]
        pred_t  = r["pred_x1_s"][pm]
        pred_c  = r["pred_cls"][pm]
        gt_t    = r["gt_x1_s"][gm]
        gt_c    = r["gt_cls"][gm]

        all_gt_cls_list.append(gt_c)
        all_gt_t_list.append(gt_t)

        sort_idx = np.argsort(-conf)
        conf_s   = conf[sort_idx]
        pred_t_s = pred_t[sort_idx]
        pred_c_s = pred_c[sort_idx]

        # AP Det: class-agnostic
        tp_det = _greedy_time_match(pred_t_s, gt_t)
        det_conf_list.append(conf_s)
        tp_det_list.append(tp_det)

        # Top Class: class-aware
        tp_top = _greedy_time_match(pred_t_s, gt_t, pred_cls=pred_c_s, gt_cls=gt_c)
        tp_top_list.append(tp_top)

        # mAP Class: all pre-NMS anchors ranked by class-c score (BD2-exact)
        all_t  = r["all_x1_s"]
        all_sc = r["all_cls_scores"]                 # (n_pre, nc)
        am     = (all_t >= lo) & (all_t <= hi)
        all_t_m  = all_t[am]
        all_sc_m = all_sc[am]
        for c in range(nc):
            gt_c_t = gt_t[gt_c == c]
            if len(gt_c_t) == 0 or not am.any():
                continue
            score_c = all_sc_m[:, c]
            sort_c  = np.argsort(-score_c)
            tp_c = _greedy_time_match(all_t_m[sort_c], gt_c_t)
            cls_conf_lists[c].append((score_c[sort_c], tp_c))

    # ── flatten and sort globally for AP ──────────────────────────────────────
    all_gt_cls   = np.concatenate(all_gt_cls_list) if all_gt_cls_list else np.zeros(0, int)
    n_gt_total   = int(all_gt_cls.size)
    gt_per_class = np.bincount(all_gt_cls.astype(int), minlength=nc) if n_gt_total else np.zeros(nc, int)

    def _global_ap(conf_list, tp_list, n_gt):
        if not conf_list or n_gt == 0:
            return float("nan")
        conf_all = np.concatenate(conf_list)
        tp_all   = np.concatenate(tp_list)
        gsort    = np.argsort(-conf_all)
        return _average_precision(tp_all[gsort], conf_all[gsort], n_gt)

    # ── 1. AP Det ─────────────────────────────────────────────────────────────
    ap_det = _global_ap(det_conf_list, tp_det_list, n_gt_total)

    # ── 2. mAP Class ──────────────────────────────────────────────────────────
    per_class_ap: dict[int, float] = {}
    for c in range(nc):
        n_gt_c = int(gt_per_class[c])
        if n_gt_c == 0:
            continue
        pairs = cls_conf_lists[c]
        if not pairs:
            per_class_ap[c] = 0.0
            continue
        conf_c = np.concatenate([p[0] for p in pairs])
        tp_c   = np.concatenate([p[1] for p in pairs])
        gsort  = np.argsort(-conf_c)
        per_class_ap[c] = _average_precision(tp_c[gsort], conf_c[gsort], n_gt_c)
    map_class = float(np.nanmean(list(per_class_ap.values()))) if per_class_ap else 0.0

    # ── 3. Top Class AP ───────────────────────────────────────────────────────
    top_cls_ap = _global_ap(det_conf_list, tp_top_list, n_gt_total)

    # ── 4. mAP File ───────────────────────────────────────────────────────────
    # For each class c and file f: score = max(cls_scores[:, c]) across all detections.
    file_dets: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    file_gt: dict[str, set] = defaultdict(set)

    for r in records:
        f = r["im_file"]
        file_gt[f].update(r["gt_cls"][r["_gt_mask"]].tolist())
        am_f = (r["all_x1_s"] >= lo) & (r["all_x1_s"] <= hi)
        if am_f.any():
            cs = r["all_cls_scores"][am_f]
            for ci in range(nc):
                v = float(cs[:, ci].max())
                if v > file_dets[f][ci]:
                    file_dets[f][ci] = v

    all_files = sorted(set(list(file_gt.keys()) + list(file_dets.keys())))
    file_ap_per_class: dict[int, float] = {}
    for c in range(nc):
        if gt_per_class[c] == 0:
            continue
        scores_f = np.array([
            file_dets[f][c] for f in all_files
        ])
        labels_f = np.array([1 if c in file_gt.get(f, set()) else 0 for f in all_files])
        sort_f = np.argsort(-scores_f)
        file_ap_per_class[c] = _average_precision(
            labels_f[sort_f], scores_f[sort_f], int(labels_f.sum())
        )

    map_file = float(np.nanmean(list(file_ap_per_class.values()))) if file_ap_per_class else 0.0

    return {
        "map": {
            "yolo": {
                "map50":     yolo_stats.get("metrics/mAP50(B)",     0.0),
                "map50_95":  yolo_stats.get("metrics/mAP50-95(B)",  0.0),
                "precision": yolo_stats.get("metrics/precision(B)", 0.0),
                "recall":    yolo_stats.get("metrics/recall(B)",    0.0),
            },
        },
        "ap_det":            ap_det,
        "map_class":         map_class,
        "top_cls_ap":        top_cls_ap,
        "map_file":          map_file,
        "per_class_ap":      {names_list[c]: v for c, v in per_class_ap.items()},
        "file_ap_per_class": {names_list[c]: v for c, v in file_ap_per_class.items()},
        "n_gt":              n_gt_total,
        "n_det":             int(sum(len(c) for c in det_conf_list)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(m: dict) -> None:
    W = 72
    SEP = "─" * W

    def row(label, value):
        print(f"  {label:<44s} {value}")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "   nan"

    print()
    print("=" * W)
    print("  BatDetect2 evaluation  (Mac Aodha et al. 2022 metrics)  [YOLO model]")
    print("=" * W)

    maps = m["map"]
    y = maps["yolo"]
    print("\n  YOLO/COCO metrics (supplementary):")
    print(SEP)
    row("mAP50    (Ultralytics)",   f"{y['map50']:.4f}")
    row("mAP50-95 (Ultralytics)",   f"{y['map50_95']:.4f}")
    if "coco_map" in maps:
        c = maps["coco_map"]
        row("mAP50    (pycocotools)", f"{c['map50']:.4f}")
        row("mAP50-95 (pycocotools)", f"{c['map50_95']:.4f}")

    print(f"\n  Paper metrics  (GT calls: {m['n_gt']}  |  Predictions: {m['n_det']}):")
    print(SEP)
    row("AP Det    (detection, 10 ms temporal match)", fmt(m["ap_det"]))
    row("mAP Class (mean per-class AP)",               fmt(m["map_class"]))
    row("Top Class (top-cls confidence AP)",            fmt(m["top_cls_ap"]))
    row("mAP File  (file-level species-presence AP)",  fmt(m["map_file"]))

    print("\n  Per-class breakdown:")
    print(SEP)
    header = f"  {'Species':<16s} {'AP-class':>9s} {'AP-file':>9s}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name in sorted(m["per_class_ap"]):
        ap_c = m["per_class_ap"][name]
        ap_f = m["file_ap_per_class"].get(name, float("nan"))
        print(f"  {name:<16s} {fmt(ap_c):>9s} {fmt(ap_f):>9s}")

    print("=" * W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to args.yaml of the trained run")
    parser.add_argument("--clip-duration", type=float, default=1.0, metavar="S",
                        help="Clip duration in seconds (for pixel→time conversion). Default: 1.0")
    parser.add_argument("--out", default=None,
                        help="Path for JSON output.")
    parser.add_argument("--plots", action="store_true",
                        help="Save YOLO confusion matrix and PR curve plots.")
    parser.add_argument("--coco-map", action="store_true",
                        help="Also compute mAP50/mAP50-95 via pycocotools COCOeval "
                             "and store under 'coco_map' in the JSON output.")
    parser.add_argument("--end2end", default="auto", choices=["auto", "on", "off"],
                        help="Which YOLO26 head to score: 'auto' keeps whatever the "
                             "checkpoint declares (one2one for YOLO26), 'on' forces the "
                             "NMS-free one2one head, 'off' forces the one2many head plus "
                             "NMS. Default: auto")
    parser.add_argument("--agnostic-nms", default="auto", choices=["auto", "on", "off"],
                        help="Class-agnostic NMS. 'auto' keeps the checkpoint's own setting. 'off' makes NMS "
                             "class-wise, so a higher-scoring wrong-class box can no longer suppress the "
                             "correct-class box on the same object. Only affects --end2end off.")
    parser.add_argument("--max-det", type=int, default=None, metavar="N",
                        help="Max detections per clip. For the one2one head this is a "
                             "topk that always saturates; for one2many it is a post-NMS "
                             "cap. Default: value from args.yaml")
    parser.add_argument("--iou", type=float, default=None, metavar="T",
                        help="NMS IoU threshold. Only has an effect with --end2end off "
                             "(the one2one head ignores it). Default: value from args.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    run_dir = config_path.parent

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["device"] = 0  # always evaluate on GPU 0, regardless of args.yaml

    trainer = CustomTrainer(cfg=None, overrides=cfg_dict)
    # args.yaml stores the bare run name, so log output would not say which run
    # this is; name it after the directory the checkpoint came from instead.
    trainer.args.name = "/".join(run_dir.parts[-2:])
    trainer._setup_train()
    best_ckpt = run_dir / "weights" / "best.pt"
    if not best_ckpt.exists():
        best_ckpt = run_dir / "weights" / "last.pt"
    if not best_ckpt.exists():
        sys.exit(f"No checkpoint found in {run_dir / 'weights'}")

    print(f"\nCheckpoint    : {best_ckpt}")
    print(f"Clip duration : {args.clip_duration}s")

    # the dataset has a two-way split: the held-out set is used both to monitor
    # training and to report results, so "test" and "val" name the same recordings
    dataloader = trainer.val_loader

    save_dir = run_dir / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    validator_args = copy(trainer.args)
    validator_args.plots = args.plots
    validator_args.save_json = False
    validator_args.save_txt = False
    validator_args.conf = 0.001
    # NOTE: BaseValidator.__init__ does self.args = get_cfg(overrides=args), which
    # builds a *new* namespace -- anything set on validator_args after the validator
    # is constructed is silently ignored. Apply head/NMS overrides here.
    if args.end2end != "auto":
        validator_args.end2end = args.end2end == "on"
    if args.iou is not None:
        validator_args.iou = args.iou
    if args.max_det is not None:
        validator_args.max_det = args.max_det
    validator_args.pool_agnostic = validator_args.single_cls or validator_args.agnostic_nms
    if args.agnostic_nms != "auto":
        validator_args.agnostic_nms = args.agnostic_nms == "on"

    validator = PaperMetricsValidator(
        dataloader=dataloader,
        save_dir=save_dir,
        args=validator_args,
        _callbacks=trainer.callbacks,
        plot_settings=trainer.plot_settings,
        clip_duration_s=args.clip_duration,
    )

    if args.end2end == "auto":
        model_arg = best_ckpt
    else:
        # Passing a path would make --end2end a silent no-op: the override is
        # guarded by hasattr(model, "end2end"), and loading a path fuses the model,
        # which nulls the one2many head. Load unfused and set the flag here.
        from ultralytics.nn.tasks import load_checkpoint
        want_e2e = args.end2end == "on"
        model_arg, _ = load_checkpoint(best_ckpt, device="cpu", fuse=False)
        model_arg.end2end = want_e2e

    head = "one2one (NMS-free)" if validator.args.end2end in (None, True) and args.end2end != "off" \
        else f"one2many (NMS, iou={validator.args.iou})"
    print(f"Head          : {head}  [validator.args.end2end={validator.args.end2end}, "
          f"iou={validator.args.iou}, conf={validator.args.conf}, "
          f"max_det={validator.args.max_det}]")

    yolo_stats = validator(model=model_arg)

    coco_map = None
    if args.coco_map:
        if not validator.coco_records:
            sys.exit("ERROR: --coco-map requested but the validator collected no "
                     "records; cannot compute pycocotools mAP.")
        try:
            coco_map = compute_coco_map(validator.coco_records, trainer.data["names"])
        except ImportError as e:
            sys.exit(f"ERROR: --coco-map requires pycocotools, which is not installed "
                     f"({e}). Install it with 'pip install pycocotools' or rebuild the "
                     f"image (it is listed in requirements.txt).")
        except Exception as e:
            sys.exit(f"ERROR: pycocotools mAP failed: {e}")

    metrics = compute_paper_metrics(
        records=validator.records,
        class_names=trainer.data["names"],
        yolo_stats=yolo_stats or {},
        clip_duration_s=args.clip_duration,
        ignore_start_end=0.01,
    )

    if coco_map is not None:
        metrics["map"]["coco_map"] = coco_map

    print_report(metrics)

    # ── summary row, tab separated ────────────────────────────────────────────
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "nan"
    print("── mAP50 | mAP50-95 | AP Det | mAP Class | Top Class | mAP File ──")
    yolo = metrics["map"]["yolo"]
    vals = [yolo["map50"], yolo["map50_95"], metrics["ap_det"], metrics["map_class"], metrics["top_cls_ap"], metrics["map_file"]]
    print("\t".join(_fmt(v) for v in vals))
    print()

    out_path = Path(args.out) if args.out else save_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()