from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import sys
import subprocess
import shutil

# 点击运行时，Intel 训练自动使用独立环境；必须放在 torch/YOLO 导入之前。
ROOT = Path(__file__).resolve().parent


def _write_status_file(path, value):
    """在尚未导入项目运行模块时，也能安全更新训练状态。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

def _mark_child_exit(exit_code):
    """Intel 子进程被 C++ 层强制终止时，由外层进程补写失败状态。"""
    summary_path = ROOT / "outputs" / "last_training.json"
    try:
        status = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    if status.get("status") not in {"starting", "running", "resuming"}:
        return
    interrupted = exit_code in {130, -1073741510}
    status["status"] = "interrupted" if interrupted else "failed"
    status["error"] = (
        "用户中断"
        if interrupted
        else f"Intel XPU 训练子进程异常退出，返回码：{exit_code}"
    )
    status["updated_at"] = datetime.now().isoformat()
    _write_status_file(summary_path, status)
    run_dir_value = status.get("run_dir")
    if run_dir_value:
        _write_status_file(Path(run_dir_value) / "training_status.json", status)


if __name__ == "__main__":
    _cfg = json.loads((ROOT / "project_config.json").read_text(encoding="utf-8-sig"))
    if str(_cfg.get("training_device", _cfg.get("device", "cpu"))).startswith("xpu"):
        _python = ROOT / ".venv-intel" / "Scripts" / "python.exe"
        if not _python.is_file():
            raise SystemExit("缺少 Intel 环境：.venv-intel/Scripts/python.exe")
        if Path(sys.executable).resolve() != _python.resolve():
            print("切换到 Intel 训练环境……", flush=True)
            try:
                exit_code = subprocess.call(
                    [str(_python), str(Path(__file__).resolve()), *sys.argv[1:]]
                )
            except KeyboardInterrupt:
                exit_code = 130
            if exit_code:
                _mark_child_exit(exit_code)
            raise SystemExit(exit_code)

import torch
from ultralytics import YOLO
from project_runtime import config, dataset, write_json


def _weight_candidates(filename):
    """按保存时间倒序查找正式训练的权重。"""
    return sorted(
        (
            p
            for p in (ROOT / "outputs" / "training").rglob(filename)
            if p.is_file() and p.parent.name == "weights"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

def _model_names(weight):
    """读取权重中的类别名称，用来防止误用旧类别模型。"""
    model = YOLO(str(weight))
    return [model.names[i] for i in range(len(model.names))]


def _process_is_running(pid):
    """只查询进程状态；Windows 下避免用 os.kill(pid, 0) 造成误终止。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _copy_best_to_weights_library(source, run_name=None):
    """将一次训练的最佳模型复制到 weights，供后续手动选择。"""
    source = Path(source)
    if not source.is_file():
        return None
    library = ROOT / "weights"
    library.mkdir(parents=True, exist_ok=True)
    name = run_name or source.parent.parent.name
    target = library / f"{name}_best.pt"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def sync_training_weights_library():
    """同步所有已结束训练的 best.pt；跳过仍在运行的任务。"""
    exported = []
    training_root = ROOT / "outputs" / "training"
    if not training_root.is_dir():
        return exported
    active_states = {"starting", "running", "resuming"}
    for source in training_root.rglob("best.pt"):
        if not source.is_file() or source.parent.name != "weights":
            continue
        run_dir = source.parent.parent
        status_path = run_dir / "training_status.json"
        if status_path.is_file():
            try:
                state = json.loads(
                    status_path.read_text(encoding="utf-8-sig")
                ).get("status")
            except (OSError, json.JSONDecodeError):
                state = None
            if state in active_states:
                continue
        target = _copy_best_to_weights_library(source, run_dir.name)
        if target is not None:
            exported.append(target)
    return exported


def select_training_start(cfg):
    """直接使用 project_config.json 中手动指定的 pretrained 权重。"""
    configured = cfg.get("pretrained")
    if not configured:
        raise ValueError("project_config.json 缺少 pretrained 训练起点")
    selected = Path(configured)
    if not selected.is_absolute():
        selected = ROOT / selected
    if not selected.is_file():
        raise FileNotFoundError(f"pretrained 指定的训练起点不存在：{selected}")
    if selected.suffix.lower() != ".pt":
        raise ValueError(f"pretrained 必须指向 .pt 权重文件：{selected}")
    return selected, "project_config.json 中手动指定的 pretrained"


def select_resume_checkpoint(cfg):
    """查找被中断或失败任务中，类别一致且含优化器状态的 last.pt。"""
    candidates = _weight_candidates("last.pt")

    for weight in candidates:
        run_dir = weight.parent.parent
        status_path = run_dir / "training_status.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8-sig"))
            state = status.get("status")
            if state in {"starting", "running", "resuming"}:
                pid = status.get("pid")
                if pid and _process_is_running(pid):
                    print(f"[跳过] 训练任务仍在运行：{run_dir}（PID {pid}）")
                    continue
                print(f"[恢复] 检测到异常退出后遗留的 {state} 状态：{run_dir}")
            elif state not in {"interrupted", "failed"}:
                continue
            candidate = YOLO(str(weight))
            if [candidate.names[i] for i in range(len(candidate.names))] != cfg[
                "names"
            ]:
                continue
            checkpoint = candidate.ckpt
            if (
                not checkpoint
                or checkpoint.get("epoch", -1) < 0
                or not checkpoint.get("optimizer")
            ):
                print(f"[跳过] 检查点没有可恢复的轮数或优化器：{weight}")
                continue
        except Exception as exc:
            print(f"[跳过] 无法读取断点：{weight}（{exc}）")
            continue
        return weight, run_dir, status

    raise FileNotFoundError(
        "没有找到可续训的 last.pt。需要中断或失败任务中，类别一致且含优化器状态的检查点。"
    )


def configure_intel_optimizer(trainer):
    """在恢复优化器后设置；避免 CPU 检查点覆盖 Intel 兼容选项。"""
    if trainer.device.type != "xpu":
        return
    optimizer = trainer.optimizer
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return
    # foreach 会合并许多张量提交运算，在此 Intel 驱动上可能触发资源错误。
    # 保留动量、学习率与步数，只切换为逐张量的 Adam 更新实现。
    for group in optimizer.param_groups:
        group["foreach"] = False
        group["fused"] = False
    optimizer.defaults["foreach"] = False
    optimizer.defaults["fused"] = False
    print("Intel 优化器兼容模式：foreach=False, fused=False（保留原优化器状态）")


def cleanup_intel_xpu_memory(trainer):
    """在轮次边界同步并释放未使用的 Intel XPU 缓存，减少长期训练的显存碎片。"""
    if trainer.device.type != "xpu" or not hasattr(torch, "xpu"):
        return
    try:
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    except Exception as exc:
        print(f"[警告] Intel XPU 缓存清理失败：{exc}")


def main():
    parser = argparse.ArgumentParser(description="训练手模缺陷检测模型")
    parser.add_argument(
        "--check-device", action="store_true", help="仅检查训练设备，不开始训练"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从最近一次被中断的 last.pt 接着训练",
    )
    parser.add_argument(
        "--check-resume",
        action="store_true",
        help="只检查可续训断点，不开始训练",
    )
    args = parser.parse_args()

    cfg = config()

    # 训练设备单独设置，预测和评估仍可按 device 选择设备。
    device = cfg.get("training_device", cfg["device"])
    if str(device).startswith("xpu"):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError(
                "Intel GPU 不可用，请检查 XPU 版 PyTorch 和 Intel 显卡驱动。"
            )
        torch.empty(1, device=device).add_(1)
        torch.xpu.synchronize()
        print("训练显卡：", torch.xpu.get_device_name(torch.device(device)))
    print("训练设备：", device, "；PyTorch：", torch.__version__)
    if args.check_device:
        return

    if args.check_resume:
        checkpoint, run_dir, checkpoint_status = select_resume_checkpoint(cfg)
        print("可续训断点：", checkpoint)
        print("训练目录：", run_dir)
        print("已保存轮数：", checkpoint_status.get("last_saved_epoch", "未知"))
        print("续训 batch：", cfg["batch"], "；imgsz：", cfg["imgsz"])
        return

    synced_weights = sync_training_weights_library()
    if synced_weights:
        print(f"已同步 {len(synced_weights)} 个历史最佳模型到 weights。")

    if args.resume:
        training_start, run_dir, status = select_resume_checkpoint(cfg)
        training_start_source = "最近一次被中断的 last.pt（断点续训）"
        data = Path(status.get("data", ROOT / cfg["data"]))
    else:
        training_start, training_start_source = select_training_start(cfg)
        data = ROOT / cfg["data"]

    data, _ = dataset(dict(cfg, data=str(data)))

    print("本次训练起点：", training_start)
    print("自动选择依据：", training_start_source)

    if args.resume:
        status["status"] = "resuming"
        status["resumed_at"] = datetime.now().isoformat()
        status["training_start"] = str(training_start)
        status["training_start_source"] = training_start_source
        print(f"将从第 {status.get('last_saved_epoch', '未知')} 轮检查点继续训练。")
    else:
        run_dir = (
            ROOT
            / "outputs"
            / "training"
            / ("模型训练_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        status = dict(
            status="starting",
            started_at=datetime.now().isoformat(),
            data=str(data),
            pretrained=str(training_start),
            training_start=str(training_start),
            training_start_source=training_start_source,
            run_dir=str(run_dir),
            last_saved_epoch=0,
            best=None,
            last=None,
        )
        config_snapshot = dict(cfg)
        config_snapshot["resolved_training_start"] = str(training_start)
        config_snapshot["training_start_source"] = training_start_source
        write_json(run_dir / "project_config_snapshot.json", config_snapshot)

    status["device"] = str(device)
    status["python"] = sys.executable
    status["torch_version"] = torch.__version__
    status["pid"] = os.getpid()
    status["batch"] = cfg["batch"]
    status["imgsz"] = cfg["imgsz"]

    def record(state, trainer=None, error=None):
        status["status"] = state
        status["updated_at"] = datetime.now().isoformat()
        if error is not None:
            status["error"] = str(error)
        else:
            status.pop("error", None)
        for key in ("best", "last"):
            path = Path(getattr(trainer, key, run_dir / "weights" / (key + ".pt")))
            status[key] = str(path) if path.is_file() else None
        if trainer is not None and state == "running" and status["last"]:
            status["last_saved_epoch"] = trainer.epoch + 1
        if state in {"completed", "interrupted", "failed"} and status.get("best"):
            try:
                library_model = _copy_best_to_weights_library(
                    status["best"], run_dir.name
                )
                status["weights_library_model"] = (
                    str(library_model) if library_model is not None else None
                )
            except OSError as exc:
                print(f"[警告] 最佳模型复制到 weights 失败：{exc}")
        write_json(run_dir / "training_status.json", status)
        write_json(ROOT / "outputs" / "last_training.json", status)

    record("resuming" if args.resume else "starting")
    model = None
    try:
        model = YOLO(str(training_start))
        model.add_callback("on_train_start", configure_intel_optimizer)
        model.add_callback("on_train_epoch_end", cleanup_intel_xpu_memory)
        model.add_callback("on_fit_epoch_end", cleanup_intel_xpu_memory)
        model.add_callback("on_model_save", lambda trainer: record("running", trainer))
        if args.resume:
            # 恢复轮数和优化器，同时覆盖原检查点中记录的 CPU 设备。
            # batch/imgsz/workers/cache 是 Ultralytics 明确允许在续训时覆盖的参数。
            model.train(
                resume=True,
                device=device,
                batch=cfg["batch"],
                imgsz=cfg["imgsz"],
                workers=0,
                cache=False,
            )
        else:
            model.train(
                data=str(data),
                epochs=cfg["epochs"],
                patience=cfg["patience"],
                imgsz=cfg["imgsz"],
                batch=cfg["batch"],
                device=device,
                workers=0,
                seed=cfg["seed"],
                optimizer="AdamW",
                lr0=0.001,
                nbs=cfg["batch"],
                mosaic=0.25,
                close_mosaic=10,
                scale=0.2,
                fliplr=0.0,
                flipud=0.0,
                amp=False,
                project=str(run_dir.parent),
                name=run_dir.name,
                exist_ok=True,
                plots=True,
            )
        record("completed", model.trainer)
    except KeyboardInterrupt:
        record("interrupted", getattr(model, "trainer", None), "用户中断")
        print("训练已中断，之前保存的检查点可继续使用。")
    except Exception as exc:
        record("failed", getattr(model, "trainer", None), exc)
        raise
    finally:
        print("训练状态记录：", run_dir / "training_status.json")
        print("最佳模型：", status["best"] or "尚未保存")
        print("最近检查点：", status["last"] or "尚未保存")
        if status.get("weights_library_model"):
            print("已保存到模型库：", status["weights_library_model"])
        print("下次训练起点由 project_config.json 的 pretrained 决定。")


if __name__ == "__main__":
    main()
