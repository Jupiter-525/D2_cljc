"""训练、预测、评估共用的配置和结果记录。"""

from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import math
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "project_config.json"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _json_value(value):
    """把 NumPy 数值和路径转成 JSON 可以保存的类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能保存为 JSON 的类型：{type(value).__name__}")


def write_json(path, value):
    """先写入临时文件，再替换目标，避免留下写了一半的记录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_value),
        encoding="utf-8",
    )
    temp.replace(path)


def config():
    cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
    for key in ("conf", "iou"):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key}必须在0到1之间")
    return cfg


def dataset(cfg):
    p = ROOT / cfg["data"]
    if not p.is_file():
        raise FileNotFoundError(f"数据配置不存在：{p}。请先运行prepare_dataset.py。")
    spec = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
    names = spec.get("names", {})
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    if names != cfg["names"]:
        raise ValueError("数据集类别与project_config.json不一致")
    return p, spec


def load_model(cfg):
    p = ROOT / cfg["model"]
    if not p.is_file():
        raise FileNotFoundError(
            f"模型不存在：{p}。请把model设为训练记录中的best.pt路径。"
        )
    model = YOLO(str(p))
    if [model.names[i] for i in range(len(model.names))] != cfg["names"]:
        raise ValueError("模型类别与配置不一致")
    return model


def output_dir(kind, title):
    p = ROOT / "outputs" / kind / (title + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    p.mkdir(parents=True, exist_ok=False)
    return p


def snapshot(cfg):
    result = dict(cfg)
    p = ROOT / cfg["model"]
    result["resolved_model"] = str(p)
    result["model_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    result["prediction_note"] = (
        "逐张预测，rect=False；NMS使用iou；评估匹配阈值单独为0.5"
    )
    return result


def predict_one(model, path, cfg):
    # 预测和固定阈值评估必须走同一条处理路径。
    return model.predict(
        source=str(path),
        imgsz=cfg["imgsz"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        device=cfg["device"],
        rect=False,
        save=False,
        save_txt=False,
        verbose=False,
    )[0]


def run_prediction(cfg, model, paths, output, relative_root):
    output.mkdir(parents=True, exist_ok=True)
    if not paths:
        raise ValueError("没有可检测图片")
    write_json(output / "settings.json", snapshot(cfg))
    rows = []
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["图片", "检测框数", *cfg["names"], "推理毫秒", "结果图片"])
        try:
            for p in paths:
                result = predict_one(model, p, cfg)
                relative = p.relative_to(relative_root)
                dest = output / "images" / relative.parent / (relative.name + ".jpg")
                dest.parent.mkdir(parents=True, exist_ok=True)
                result.save(filename=str(dest))
                label = output / "labels" / relative.parent / (relative.name + ".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("", encoding="utf-8")
                if len(result.boxes):
                    result.save_txt(str(label), save_conf=True)
                counts = [
                    int((result.boxes.cls == i).sum()) for i in range(len(cfg["names"]))
                ]
                writer.writerow(
                    [
                        relative.as_posix(),
                        len(result.boxes),
                        *counts,
                        result.speed["inference"],
                        str(dest),
                    ]
                )
                f.flush()
                rows.append(
                    dict(
                        image=str(p),
                        boxes=result.boxes.data.cpu().tolist(),
                        speed=result.speed,
                    )
                )
                print(p.name, "检测框：", len(result.boxes))
        finally:
            write_json(output / "detections.json", rows)
    return rows


def match_boxes(predictions, truth, classes, threshold=0.5):
    """按置信度匹配同类GT，一个真实框最多匹配一次；不是COCO AP计算。"""
    tp, fp, fn = np.zeros(classes, int), np.zeros(classes, int), np.zeros(classes, int)
    used = set()
    for p in sorted(predictions, key=lambda x: -x[4]):
        cls = int(p[5])
        best, best_iou = None, -1.0
        for j, t in enumerate(truth):
            if j in used or int(t[4]) != cls:
                continue
            iw = max(0, min(p[2], t[2]) - max(p[0], t[0]))
            ih = max(0, min(p[3], t[3]) - max(p[1], t[1]))
            inter = iw * ih
            union = (
                (p[2] - p[0]) * (p[3] - p[1]) + (t[2] - t[0]) * (t[3] - t[1]) - inter
            )
            overlap = inter / union if union > 0 else 0
            if overlap > best_iou:
                best, best_iou = j, overlap
        if best is not None and best_iou >= threshold:
            used.add(best)
            tp[cls] += 1
        else:
            fp[cls] += 1
    for j, t in enumerate(truth):
        if j not in used:
            fn[int(t[4])] += 1
    return tp, fp, fn


def operating_metrics(cfg, rows, image_dir, label_dir):
    from PIL import Image

    tp, fp, fn = (
        np.zeros(len(cfg["names"]), int),
        np.zeros(len(cfg["names"]), int),
        np.zeros(len(cfg["names"]), int),
    )
    normal = false_alarm = defective = missed_images = 0
    for row in rows:
        p = Path(row["image"])
        label = label_dir / p.relative_to(image_dir).with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"缺少真实标注：{label}")
        with Image.open(p) as im:
            w, h = im.size
        truth = []
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, x, y, bw, bh = map(float, line.split())
            truth.append(
                [
                    (x - bw / 2) * w,
                    (y - bh / 2) * h,
                    (x + bw / 2) * w,
                    (y + bh / 2) * h,
                    int(cls),
                ]
            )
        a, b, c = match_boxes(row["boxes"], truth, len(cfg["names"]))
        tp += a
        fp += b
        fn += c
        if not truth:
            normal += 1
            false_alarm += bool(row["boxes"])
        else:
            defective += 1
            missed_images += not bool(row["boxes"])

    def ratio(a, b):
        return float(a / b) if b else None

    return dict(
        conf=cfg["conf"],
        nms_iou=cfg["iou"],
        matching_iou=0.5,
        classes={
            name: dict(
                tp=int(tp[i]),
                fp=int(fp[i]),
                fn=int(fn[i]),
                precision=ratio(tp[i], tp[i] + fp[i]),
                recall=ratio(tp[i], tp[i] + fn[i]),
            )
            for i, name in enumerate(cfg["names"])
        },
        normal_images=normal,
        false_alarm_images=false_alarm,
        normal_image_false_alarm_rate=ratio(false_alarm, normal),
        defective_images=defective,
        no_detection_defective_images=missed_images,
        note="图片级统计，不是手模级；缺陷图输出错误框不代表正确检出，请同时看类别TP/FN。",
    )
