"""点击运行：检测图片，保存图片、框坐标、汇总表和本次设置。"""

from project_runtime import ROOT, EXT, config, load_model, output_dir, run_prediction


def main():
    cfg = config()
    source = ROOT / cfg["source"]
    if not source.exists():
        raise FileNotFoundError(f"图片路径不存在：{source}")
    paths = (
        [source]
        if source.is_file() and source.suffix.lower() in EXT
        else sorted(
            p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXT
        )
    )
    if not paths:
        raise ValueError("输入目录没有图片")
    model = load_model(cfg)
    output = output_dir("predict", "检测结果_")
    print("当前模型：", ROOT / cfg["model"])
    run_prediction(
        cfg, model, paths, output, source.parent if source.is_file() else source
    )
    print("检测完成，图片和summary.csv保存在：", output)


if __name__ == "__main__":
    main()
