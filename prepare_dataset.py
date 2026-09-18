"""点击运行即可检查标注、自动分组并生成YOLO数据集；可选参数用于单独执行某一步。"""

from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import csv
import hashlib
import json
import math
import shutil
from PIL import Image
import yaml

NAMES = json.loads(
    (Path(__file__).resolve().parent / "project_config.json").read_text(
        encoding="utf-8-sig"
    )
)["names"]
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def inspect(root):
    errors, records, missing = [], [], []
    images = sorted(
        p
        for p in (root / "raw_images").rglob("*")
        if p.is_file() and p.suffix.lower() in EXT
    )
    annotations = {}
    for p in (root / "annotations").rglob("*.json"):
        key = p.stem.lower()
        if key in annotations:
            errors.append(f"重复JSON文件名：{p.name}")
        annotations[key] = p
    seen, hashes, used = set(), {}, set()
    for p in images:
        key = p.stem.lower()
        if key in seen:
            errors.append(f"重复图片文件名：{p.name}")
        seen.add(key)
        annotation = annotations.get(key)
        if annotation is None:
            missing.append(p.relative_to(root).as_posix())
            continue
        used.add(key)
        try:
            with Image.open(p) as im:
                im.load()
                w, h = im.size
            d = json.loads(annotation.read_text(encoding="utf-8-sig"))
            if (d.get("imageWidth"), d.get("imageHeight")) != (w, h):
                raise ValueError("JSON尺寸与图片不一致")
            shapes = d.get("shapes")
            if not isinstance(shapes, list):
                raise ValueError("缺少shapes列表")
            labels = []
            for s in shapes:
                label, points = s.get("label"), s.get("points", [])
                if label not in NAMES:
                    raise ValueError(
                        f"未知类别：{label}；当前类别为{NAMES}，请复核标注名称。"
                    )
                kind = s.get("shape_type")
                if kind == "circle" and len(points) == 2:
                    (cx, cy), (ex, ey) = points
                    radius = math.hypot(ex - cx, ey - cy)
                    points = [[cx - radius, cy - radius], [cx + radius, cy + radius]]
                    kind = "rectangle"
                if not (
                    (kind == "rectangle" and len(points) == 2)
                    or (kind == "polygon" and len(points) >= 3)
                    or (kind == "linestrip" and len(points) >= 2)
                ):
                    raise ValueError(f"仅支持有效矩形、多边形、圆形或折线：{kind}")
                if any(len(pt) != 2 for pt in points):
                    raise ValueError("坐标格式错误")
                xs, ys = zip(*[(float(x), float(y)) for x, y in points])
                if not all(math.isfinite(v) for v in (*xs, *ys)):
                    raise ValueError("非有限坐标")
                # 折线取所有顶点的外接矩形；不猜测裂纹宽度，零面积需重新框选。
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError("标注越界或框面积为零")
                labels.append(
                    [
                        NAMES.index(label),
                        (x1 + x2) / 2 / w,
                        (y1 + y2) / 2 / h,
                        (x2 - x1) / w,
                        (y2 - y1) / h,
                    ]
                )
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            duplicate = hashes.get(digest)
            hashes[digest] = p.name
            records.append(
                dict(
                    image=p.relative_to(root).as_posix(),
                    annotation=annotation.relative_to(root).as_posix(),
                    labels=labels,
                    sha256=digest,
                    duplicate_of=duplicate,
                )
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f"{p.name}：{exc}")
    errors.extend(
        f"JSON没有同名图片：{annotations[k].name}" for k in annotations.keys() - used
    )
    counts = Counter(NAMES[row[0]] for r in records for row in r["labels"])
    return dict(
        image_count=len(images),
        json_count=len(annotations),
        valid_pairs=len(records),
        normal_images=sum(not r["labels"] for r in records),
        boxes={name: counts[name] for name in NAMES},
        missing_json=missing,
        errors=errors,
        records=records,
    )


def deduplicate(root):
    """完全相同的文件只保留一份；副本和标注归档，可恢复。"""
    buckets = {}
    for p in sorted((root / "raw_images").rglob("*")):
        if p.is_file() and p.suffix.lower() in EXT:
            buckets.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(p)
    annotations = {}
    for p in (root / "annotations").rglob("*.json"):
        annotations.setdefault(p.stem.lower(), []).append(p)
    archive = (
        root
        / "outputs"
        / "archive"
        / "duplicates"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    moved = []
    for paths in buckets.values():
        if len(paths) < 2:
            continue
        # 优先保留有JSON的版本；其次保留较短的原始文件名。
        paths.sort(
            key=lambda p: (
                not bool(annotations.get(p.stem.lower())),
                len(p.name),
                p.name,
            )
        )
        keep = paths[0]
        for duplicate in paths[1:]:
            files = [duplicate]
            if duplicate.stem.lower() != keep.stem.lower():
                files += annotations.get(duplicate.stem.lower(), [])
            for source in files:
                source.resolve().relative_to(root.resolve())
                target = archive / source.relative_to(root)
                target.resolve().relative_to(archive.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
            moved.append(
                dict(
                    kept=keep.relative_to(root).as_posix(),
                    removed=duplicate.relative_to(root).as_posix(),
                )
            )
    if moved:
        (archive / "manifest.json").write_text(
            json.dumps(moved, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已移除完全重复图片 {len(moved)} 张，副本及标注备份：{archive}")


def auto_split(root, report, seed=42):
    """固定图片数量的多标签分层搜索，不使用拍摄时间分组。"""
    import numpy as np

    records = sorted(report["records"], key=lambda r: r["image"])
    n = len(records)
    if n < 3:
        raise ValueError("有效图片不足3张，无法划分train/val/test")
    cfg = json.loads((root / "project_config.json").read_text(encoding="utf-8-sig"))
    splits = ("train", "val", "test")
    ratios = np.array(
        [
            cfg.get("split_ratios", dict(train=0.7, val=0.2, test=0.1))[s]
            for s in splits
        ],
        dtype=float,
    )
    if (
        not np.isfinite(ratios).all()
        or (ratios <= 0).any()
        or not math.isclose(float(ratios.sum()), 1)
    ):
        raise ValueError("split_ratios必须为正数且总和为1")
    sizes = np.floor(n * ratios).astype(int)
    for i in np.argsort(-(n * ratios - sizes), kind="stable")[: n - int(sizes.sum())]:
        sizes[i] += 1
    for i in range(3):
        if sizes[i] == 0:
            donor = int(sizes.argmax())
            sizes[donor] -= 1
            sizes[i] += 1
    features = np.zeros((n, len(NAMES) + 1), dtype=float)
    boxes = np.zeros((n, len(NAMES)), dtype=float)
    for i, record in enumerate(records):
        features[i, 0] = not record["labels"]
        for label in record["labels"]:
            features[i, label[0] + 1] = 1
            boxes[i, label[0]] += 1
    total = features.sum(axis=0)
    box_total = boxes.sum(axis=0)

    def score(order):
        groups = np.split(order, np.cumsum(sizes)[:-1])
        counts = np.array([features[g].sum(axis=0) for g in groups])
        bc = np.array([boxes[g].sum(axis=0) for g in groups])
        train_missing = int(((total > 0) & (counts[0] == 0)).sum())
        coverage_missing = int(
            np.maximum(0, np.minimum(total, 3) - (counts > 0).sum(axis=0)).sum()
        )
        balance = float(
            (((counts[:, total > 0] / total[total > 0]) - ratios[:, None]) ** 2).sum()
        )
        balance += float(
            (
                ((bc[:, box_total > 0] / box_total[box_total > 0]) - ratios[:, None])
                ** 2
            ).sum()
        )
        return (train_missing, coverage_missing, balance)

    rng = np.random.default_rng(seed)
    best = rng.permutation(n)
    best_score = score(best)
    # 固定容量、多次随机起点，加跨集合交换改进；固定seed保证可重复。
    for _ in range(3000):
        candidate = rng.permutation(n)
        value = score(candidate)
        if value < best_score:
            best, best_score = candidate, value
    for _ in range(3000):
        candidate = best.copy()
        i, j = rng.choice(n, 2, replace=False)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        value = score(candidate)
        if value < best_score:
            best, best_score = candidate, value
    out = root / "outputs" / "dataset_check"
    out.mkdir(parents=True, exist_ok=True)
    plan = root / "annotations" / "split_plan.csv"
    if plan.exists():
        history = root / "outputs" / "archive" / "dataset_check"
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            plan,
            history
            / (
                "split_plan_before_"
                + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                + ".csv"
            ),
        )
    summary = {}
    with plan.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "group", "split"])
        writer.writeheader()
        for split, indices in zip(splits, np.split(best, np.cumsum(sizes)[:-1])):
            for i in indices:
                r = records[i]
                writer.writerow(
                    dict(image=r["image"], group="image_" + r["sha256"], split=split)
                )
            summary[split] = dict(
                images=len(indices),
                normal=int(features[indices, 0].sum()),
                boxes={
                    name: int(boxes[indices, k].sum()) for k, name in enumerate(NAMES)
                },
            )
    (out / "split_quality.json").write_text(
        json.dumps(
            dict(
                method="image_multilabel_stratified",
                seed=seed,
                objective=best_score,
                summary=summary,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("按图片进行多标签分层划分（无时间窗口），相同数据和seed得到相同划分。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument(
        "--auto-split", action="store_true", help="自动填写分组表，不开始训练"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_all = not (args.check or args.build or args.auto_split)
    if run_all:
        print("一键准备数据：检查标注 → 自动填写分组 → 转换并生成数据集")
    root = args.root.resolve()
    global NAMES
    cfg = json.loads((root / "project_config.json").read_text(encoding="utf-8-sig"))
    NAMES = cfg["names"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    if not args.check:
        deduplicate(root)
    report = inspect(root)
    out = root / "outputs" / "dataset_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "missing_json.txt").write_text(
        "\n".join(report["missing_json"]), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("records", "missing_json")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"已跳过无JSON图片（不参与训练）：{len(report['missing_json'])}，清单：{out / 'missing_json.txt'}"
    )
    if report["errors"]:
        raise ValueError("已有JSON存在标注错误，请修正后再次运行。无JSON图片自动跳过。")
    if args.check:
        print("检查完成。点击运行将自动分层划分并生成数据集。")
        return
    auto_split(root, report, seed)
    if args.auto_split:
        return
    plan = root / "annotations" / "split_plan.csv"
    with plan.open(encoding="utf-8-sig", newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["image"] not in set(report["missing_json"])
        ]
    assignments, groups, hashes = {}, {}, {}
    for row in rows:
        name, group, split = row["image"], row["group"].strip(), row["split"].strip()
        if name in assignments or not group or split not in ("train", "val", "test"):
            raise ValueError(f"分组表有重复、空白或无效分区：{name}")
        if group in groups and groups[group] != split:
            raise ValueError(f"同组图片不能跨分区：{group}")
        groups[group] = split
        assignments[name] = split
    if set(assignments) != {r["image"] for r in report["records"]}:
        raise ValueError("分组表与当前有效图片不一致，请补齐或移除过期行。")
    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in report["records"] if assignments[r["image"]] == split]
        counts = Counter(NAMES[l[0]] for r in subset for l in r["labels"])
        if not subset:
            raise ValueError(f"{split}没有图片，请调整分组。")
        if split == "train" and not counts:
            raise ValueError("训练集没有缺陷标注，不能仅用正常图片训练检测模型。")
        absent = [n for n in NAMES if not counts[n]]
        if absent:
            print(
                f"[学习提示] {split}缺少类别：{absent}。训练集缺少的类别无法学会，验证/测试集缺少的类别无法评估。"
            )
        summary[split] = dict(
            images=len(subset),
            normal=sum(not r["labels"] for r in subset),
            boxes=dict(counts),
        )
        for r in subset:
            if r["sha256"] in hashes:
                raise ValueError(
                    f"重复图片，请复核：{r['image']} 与 {hashes[r['sha256']]}"
                )
            hashes[r["sha256"]] = r["image"]
    target = root / "datasets" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for split in summary:
        (target / "images" / split).mkdir(parents=True)
        (target / "labels" / split).mkdir(parents=True)
    for r in report["records"]:
        split = assignments[r["image"]]
        source = root / r["image"]
        shutil.copy2(source, target / "images" / split / source.name)
        text = "\n".join(
            str(l[0]) + " " + " ".join(f"{v:.8f}" for v in l[1:]) for l in r["labels"]
        )
        (target / "labels" / split / (source.stem + ".txt")).write_text(
            text, encoding="utf-8"
        )
    spec = dict(
        path=target.as_posix(),
        train="images/train",
        val="images/val",
        test="images/test",
        names=dict(enumerate(NAMES)),
    )
    data = target / "data.yaml"
    data.write_text(
        yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(plan, target / "split_plan.csv")
    (target / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_file = root / "project_config.json"
    cfg = json.loads(config_file.read_text(encoding="utf-8-sig"))
    shutil.copy2(config_file, target / "previous_project_config.json")
    cfg["data"] = data.relative_to(root).as_posix()
    config_file.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已生成数据集并更新配置：{data}")
    print("数据准备完成。下一步打开train.py，点击运行开始训练。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"[错误] {exc}")
