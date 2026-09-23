"""
Compute mAP50 and mAP50-95 via pycocotools COCOeval.

Input: the per-image records CustomValidator collects in coco_records.

Each record must contain:
    im_file   : str   — path used as a unique image identifier
    gt_boxes  : (N,4) float32 ndarray — ground-truth boxes in xyxy pixel coords
    gt_cls    : (N,)  int ndarray     — ground-truth class ids (0-indexed)
    pred_boxes: (M,4) float32 ndarray — predicted boxes in xyxy pixel coords
    pred_conf : (M,)  float32 ndarray — prediction confidence scores
    pred_cls  : (M,)  int ndarray     — predicted class ids (0-indexed)
"""

from __future__ import annotations

import contextlib
import io
from typing import Any


def _image_key(record: dict[str, Any]) -> str:
    """Unique COCO image identifier for one record.

    `im_file` is not always one-per-image: callers that also compute the
    file-level metric coarsen it to a recording-level key, which would merge
    every window of a recording into a single COCO image. Those callers pass
    the raw per-image path as `coco_key`.
    """
    return record.get("coco_key", record["im_file"])


def compute_coco_map(
    records: list[dict[str, Any]],
    class_names: dict[int, str],
) -> dict[str, float]:
    """Return mAP50 and mAP50-95 computed by pycocotools COCOeval.

    Boxes must be in a consistent xyxy pixel coordinate space across GT and
    predictions (e.g. the letterboxed imgsz space used by CustomValidator).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    file_to_id: dict[str, int] = {}
    for r in records:
        file_to_id.setdefault(_image_key(r), len(file_to_id))
    nc = len(class_names)

    images     = [{"id": i} for i in range(len(file_to_id))]
    categories = [{"id": c, "name": class_names[c]} for c in range(nc)]
    annotations: list[dict] = []
    ann_id = 0
    for r in records:
        img_id = file_to_id[_image_key(r)]
        for box, cat in zip(r["gt_boxes"], r["gt_cls"]):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w, h = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(cat),
                "bbox": [x1, y1, w, h],
                "area": w * h, "iscrowd": 0,
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    preds: list[dict] = []
    for r in records:
        img_id = file_to_id[_image_key(r)]
        for box, conf, cat in zip(r["pred_boxes"], r["pred_conf"], r["pred_cls"]):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w, h = x2 - x1, y2 - y1
            preds.append({
                "image_id": img_id,
                "category_id": int(cat),
                "bbox": [x1, y1, w, h],
                "score": float(conf),
            })

    if not preds:
        return {"map50": 0.0, "map50_95": 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_pred = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # precision: [T(iou), R(recall), K(class), A(area), M(maxDet)]
    prec = coco_eval.eval["precision"]
    per_class: dict[str, dict[str, float]] = {}
    for k in range(nc):
        p_all = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        if (p_all > -1).sum() == 0:
            continue
        per_class[class_names[k]] = {
            "map50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else 0.0,
            "map50_95": float(p_all[p_all > -1].mean()),
        }

    # stats[0] = mAP@0.50:0.95,  stats[1] = mAP@0.50
    return {
        "map50": float(coco_eval.stats[1]),
        "map50_95": float(coco_eval.stats[0]),
        "per_class": per_class,
    }