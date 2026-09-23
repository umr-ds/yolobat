from Datasets import YoloDataset
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

import argparse
import os

DEFAULT_OUTPUT_DIR = '/yolobat-train/var/model_export/yolobat'


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO model exporter")
    parser.add_argument('--config', type=str, required=True, help='Path to YOLO experiment folder')
    parser.add_argument('--version', type=str, required=True, help='Model name, used as the export directory (e.g., yolobat)')
    parser.add_argument('--release', type=str, required=True, help='Release identifier (e.g., 2026.1)')
    parser.add_argument('--format', type=str, default='openvino', choices=['onnx', 'openvino', 'ncnn'], help='Export format')
    parser.add_argument('--int8', action='store_true', help='Quantize model with INT8. Only available with OpenVINO format')
    parser.add_argument('--local', action='store_true', help='Save model beside the checkpoint instead of in the output directory')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Where exports are written, as <output-dir>/<version>/<version>_<release>. '
                             f'Pass an empty string to write beside the checkpoint instead. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument('--end2end', type=str, default='auto', choices=['auto', 'on', 'off'],
                        help="Which YOLO26 head to export. 'auto' keeps what the checkpoint declares "
                             "(one2one, NMS-free). 'off' exports the one2many head instead, which then "
                             "requires NMS -- combine with --nms to fuse it into the graph. Lets you time "
                             "the NMS cost with the backbone held constant.")
    parser.add_argument('--nms', action='store_true',
                        help='Fuse NMS into the exported graph (Ultralytics `nms=True`). Only meaningful '
                             'with --end2end off; an NMS-free head needs no NMS.')
    return parser.parse_args()

def export_model(args):
    """Export YOLO model to specified format."""
    precision = "INT8" if args.int8 else "FP32"
    print(f"Exporting {args.version} model from {args.config} to {args.format}-{precision} format...")
    
    cfg = get_cfg(f'{args.config}/args.yaml')

    cfg.version = args.version
    cfg.release = args.release
    cfg.output_dir = args.output_dir
    
    ckpt = f'{args.config}/weights/best.pt'
    if args.end2end == 'auto':
        best = YOLO(ckpt)
    else:
        # Same trap as the eval scripts: a path never satisfies the hasattr(model, "end2end")
        # guard, and loading via a path fuses the model -- Detect.fuse() sets cv2 = cv3 = None
        # while still in end2end mode, destroying the one2many head before it can be selected.
        # Load unfused, set the flag, then wrap.
        from ultralytics.nn.tasks import load_checkpoint
        _m, _ = load_checkpoint(ckpt, device='cpu', fuse=False)
        _m.end2end = (args.end2end == 'on')
        best = YOLO(ckpt)
        best.model = _m
        print(f"Head override: end2end={_m.end2end} "
              f"({'one2one, NMS-free' if _m.end2end else 'one2many + NMS'})")
    best.overrides = cfg.__dict__
    match args.format:
        case "onnx":
            best.export(format="onnx", imgsz=cfg.imgsz, dynamic=True, batch=1, simplify=True, int8=False, nms=args.nms, local=args.local)
        case "ncnn":
            # ncnn takes neither dynamic axes nor NMS; the onnx below keeps both
            best.export(format="ncnn", imgsz=cfg.imgsz, batch=1, local=args.local)
            best.export(format="onnx", imgsz=cfg.imgsz, dynamic=True, batch=1, simplify=True, int8=False, nms=args.nms, local=args.local)
        case "openvino":
            if args.int8:
                trainset = YoloDataset(
                    annotation_file=f'var/data_exports/{cfg.dataset_name}_dataset.json', # file path to annotation file
                    filter_file=f'var/data_exports/{cfg.dataset_name}_val.txt',    # file path to a file, which has the names of audio files, which should be used in the dataset
                    hyp=cfg,
                    augment=True,
                    audio_path="/data/bats/"
                )
            best.export(format="openvino", imgsz=cfg.imgsz, dynamic=True, batch=1, simplify=True, int8=args.int8, nms=args.nms, dataset=trainset if args.int8 else None, local=args.local) # type: ignore
            best.export(format="onnx", imgsz=cfg.imgsz, dynamic=True, batch=1, simplify=True, int8=False, nms=args.nms, local=args.local)


def main():
    args = parse_args()
    assert os.path.exists(args.config), f"Config not found {args.config}"
    assert not args.int8 or args.format == 'openvino', "INT8 quantization is only available with OpenVINO format"
    if args.output_dir and not args.local:
        target = os.path.join(args.output_dir, args.version, f'{args.version}_{args.release}')
        assert not os.path.exists(target), f"Release already exists: {target}"
    export_model(args)
    print("Export complete!")

if __name__ == "__main__":
    main()