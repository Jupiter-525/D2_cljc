"""标准AP评估，以及与predict.py完全相同处理路径的固定阈值评估。"""

import argparse
import json
from pathlib import Path
from project_runtime import (
    EXT,
    config,
    dataset,
    load_model,
    output_dir,
    write_json,
    snapshot,
    run_prediction,
    operating_metrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    cfg = config()
    data, spec = dataset(cfg)
    model = load_model(cfg)
    output = output_dir("evaluation", f"模型评估_{args.split}_")
    write_json(
        output / "settings.json",
        dict(snapshot(cfg), split=args.split, standard_ap_conf=0.001),
    )
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=cfg["imgsz"],
        batch=1,
        device=cfg["device"],
        workers=0,
        conf=0.001,
        iou=cfg["iou"],
        rect=False,
        plots=True,
        verbose=True,
        project=str(output),
        name="standard",
        exist_ok=True,
    )
    write_json(
        output / "standard_metrics.json",
        dict(
            metrics=metrics.results_dict,
            per_class=metrics.summary(),
            speed=metrics.speed,
            note="AP扫描置信度，表中P/R不一定对应实际检测conf；实际工作点见operating_metrics.json。",
        ),
    )
    base = Path(spec.get("path", data.parent))
    if not base.is_absolute():
        base = data.parent / base
    image_dir = base / spec[args.split]
    # 此项目由prepare_dataset.py生成images/split及labels/split结构。
    label_dir = base / "labels" / args.split
    paths = sorted(
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXT
    )
    rows = run_prediction(cfg, model, paths, output / "operating", image_dir)
    report = operating_metrics(cfg, rows, image_dir, label_dir)
    write_json(output / "operating_metrics.json", report)
    print("实际检测阈值下的指标（与predict.py设置一致）：")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("评估结果：", output)


if __name__ == "__main__":
    main()
