# 手模缺陷检测

检测类别：残料 `residue`、裂纹 `crack`、未脱模 `undemolded`。

## 日常只需要这四个入口

| 程序 | 用途 | 怎么运行 |
| --- | --- | --- |
| `prepare_dataset.py` | 检查 Labelme 标注，转换 YOLO 标签，划分训练/验证/测试集 | 点击运行 |
| `train.py` | 自动选择最新且类别一致的最佳权重，开始新训练 | 点击运行 |
| `predict.py` | 检测配置中指定的图片 | 点击运行 |
| `evaluate.py` | 对验证集做标准指标和实际阈值评估 | 点击运行 |

训练采用 Intel 显卡时，`train.py` 会自动切换到 `.venv-intel`。

从已中断或失败任务的检查点继续训练，在项目终端输入：

```powershell
.\.venv-intel\Scripts\python.exe .\train.py --resume
```

仅检查训练设备：

```powershell
.\.venv\Scripts\python.exe .\train.py --check-device
```

仅检查标注（报告写入 outputs，不改写分组表）：

```powershell
.\.venv\Scripts\python.exe .\prepare_dataset.py --check
```

最终测试集评估：

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --split test
```

## 目录说明

```text
D2_cljc/
├── README.md                  使用入口
├── project_config.json        路径、类别、轮数、阈值与设备
├── prepare_dataset.py         数据集准备
├── train.py                   训练与断点续训
├── predict.py                 图片检测
├── evaluate.py                模型评估
├── project_runtime.py         共用函数，通常不单独运行
├── requirements.txt           基础依赖
├── requirements-intel.txt     Intel训练依赖
├── raw_images/                原始图片
├── annotations/               Labelme JSON与分组表
├── datasets/                  已生成的各版本YOLO数据集
├── test_images/               待检测图片
├── weights/                   官方预训练权重
├── outputs/
│   ├── training/              正式训练模型、曲线和状态
│   ├── predict/               检测结果
│   ├── evaluation/            评估结果
│   ├── dataset_check/         当前数据检查报告
│   ├── archive/               发现重复图片时自动建立的临时归档
│   └── last_training.json     最近一次训练状态
├── docs/                      修改记录与详细说明
├── .vscode/                   VS Code训练启动选项
├── .venv/                     原运行环境（含Labelme）
└── .venv-intel/               Intel训练环境
```

## 配置中最常用的项目

| 配置 | 含义 |
| --- | --- |
| `model` | 预测和评估使用的模型文件 |
| `pretrained` | 没有可用历史模型时的训练起点 |
| `data` | 当前数据集 data.yaml，数据准备程序自动更新 |
| `source` | 待检测图片目录 |
| `training_device` | 训练设备，Intel 为 `xpu:0`，CPU为 `cpu` |
| `device` | 预测和评估设备 |
| `epochs` / `patience` | 新训练的最大轮数 / 无改善时等待轮数 |
| `imgsz` / `batch` | 输入尺寸 / 每批图片数 |
| `conf` / `iou` | 检测阈值 / 检测后处理参数 |

`--resume` 主要恢复检查点中的原训练参数；修改配置中的轮数不会自动改变续训目标。
“最新 best.pt”表示最近一次训练中的最佳模型，不代表它一定优于所有历史模型。
没有 JSON 的图片自动跳过；合法的空 shapes JSON 表示已确认正常的图片。

数据准备会重新分配样本；比较模型时应固定同一个数据集版本。当前按图片分层，不保证同一手模的相似照片不会跨集合。

Intel 环境已完成小样本验证，但曾发生底层崩溃，不能视为长期稳定性保证。模型不会因目录整理提高准确率。

详细说明见 [train与predict代码详解](docs/train与predict代码详解.md)、[目录与代码整理说明](docs/目录与代码整理说明.md) 和 [Intel训练说明](docs/Intel显卡训练_修改说明.md)。
