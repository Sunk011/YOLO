# -*- coding: utf-8 -*-

from ultralytics import YOLO

if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # 1. 加载模型
    #    - 可以从一个`.yaml`配置文件开始，从头构建一个新模型。
    #    - 也可以直接加载一个预训练的`.pt`模型，在此基础上进行微调（推荐）。
    # ---------------------------------------------------------------------------
    # 方式一：从YAML构建并加载预训练权重（适合想调整模型结构的用户）
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # 方式二：直接加载预训练模型（最常用）
    model = YOLO('yolov5m.yaml')

    # ---------------------------------------------------------------------------
    # 2. 模型训练 - 所有可选参数详解
    #    下面的参数涵盖了数据、模型、超参数、增强、日志、硬件等各个方面。
    #    您可以根据自己的需求取消注释并修改这些参数的值。
    # ---------------------------------------------------------------------------
    results = model.train(
        # ======================================================================
        # 核心训练设置 (Core Training Settings)
        # ======================================================================
        data='/home/sk/project/datasets/VisDrone_yolo/VisDrone.yaml',   # 数据集配置文件的路径。
        epochs=300,                     # 训练的总轮次 (Total number of training epochs)。
        batch=32,                       # 每批次的图像数量 (Batch size)。-1会自动根据GPU显存调整。
        imgsz=1280,                      # 输入图像的尺寸 (Image size)。可以是整数640，也可以是矩形[640, 480]。
        
        # # ======================================================================
        # # 模型与数据路径 (Model & Data Paths)
        # # ======================================================================
        # # model='yolov8n.pt',             # 初始权重路径。在代码开头已指定，这里通常不用重复。
        # resume=False,                   # 是否从上一个断点继续训练 (Resume training from last checkpoint)。
        
        # # ======================================================================
        # # 优化器与超参数 (Optimizer & Hyperparameters)
        # # ======================================================================
        optimizer='AdamW',               # 选择优化器 ('SGD', 'Adam', 'AdamW', 'auto')。'auto'会自动选择最佳的。
        conf=0.2,
        iou=0.5,
        lr0=0.001,                       # 初始学习率 (Initial learning rate)。
        # lrf=0.01,                       # 最终学习率 (Final learning rate)，最终学习率 = lr0 * lrf。
        # momentum=0.937,                 # SGD优化器的动量 (SGD momentum)。
        # weight_decay=0.0005,            # 优化器的权重衰减 (Optimizer weight decay)。
        
        # ======================================================================
        # 数据增强与正则化 (Augmentation & Regularization)
        # ======================================================================
        # hsv_h=0.015,                    # 图像色调增强的系数 (Image HSV-Hue augmentation fraction)。
        # hsv_s=0.7,                      # 图像饱和度增强的系数 (Image HSV-Saturation augmentation fraction)。
        # hsv_v=0.4,                      # 图像亮度增强的系数 (Image HSV-Value augmentation fraction)。
        degrees=10.0,                    # 图像旋转的角度范围 (+/- deg)。
        # translate=0.1,                  # 图像平移的范围 (+/- fraction)。
        scale=0,                      # 图像缩放的范围 (+/- gain)。
        # shear=0.0,                      # 图像剪切的角度范围 (+/- deg)。
        perspective=0.00001,                # 图像透视变换的系数。
        # flipud=0.0,                     # 图像垂直翻转的概率 (Probability of flipping image up-down)。
        # fliplr=0.5,                     # 图像水平翻转的概率 (Probability of flipping image left-right)。
        # mosaic=1.0,                     # 马赛克数据增强的概率 (Mosaic augmentation): 将四张图片拼接成一张。
        # mixup=0.4,                      # MixUp数据增强的概率 (MixUp augmentation): 将两张图片混合。
        copy_paste=0.5,                 # Copy-Paste数据增强的概率 (Segment task)。
        # dropout=0.0,                    # Dropout正则化率 (Use dropout regularization)。
        # crop_fraction=1.0,
        
        # # ======================================================================
        # # 日志、保存与可视化 (Logging, Saving, & Visualization)
        # # ======================================================================
        # project='usage/visdrone_args',           # 训练结果保存的项目目录 (Project directory)。
        # name='5m-1280-b32-e300',                     # 本次训练的名称 (Experiment name)。
        # exist_ok=False,                 # 如果实验名称已存在，是否覆盖 (Overwrite existing experiment)。
        # save=True,                      # 是否保存训练检查点和最终模型 (Save checkpoints and final model)。
        save_period=1,                 # 每隔多少个epoch保存一次检查点 (Save checkpoint every x epochs)。-1表示只在最后保存。
        # save_json=False,                # 是否将结果保存为COCO格式的JSON文件。
        # plots=True,                     # 是否在训练结束后生成并保存各种图表（如混淆矩阵、PR曲线等）。
        # verbose=True,                   # 是否打印详细的日志输出。
        # show_labels=True,               # 在验证时是否显示图像的标签 (Show labels during validation)。
        # show_conf=True,                 # 在验证时是否显示置信度 (Show confidence scores during validation)。
        # show_boxes=True,                # 在验证时是否显示边界框 (Show bounding boxes during validation)。
        
        # # ======================================================================
        # # 硬件与性能 (Hardware & Performance)
        # # ======================================================================
        device='4, 5, 6, 7',                    # 指定运行设备 ('cpu', '0', '0,1,2,3', or None)。None会自动选择可用GPU。
        # device='0, 1, 2, 3, 4, 5, 6, 7',                    # 指定运行设备 ('cpu', '0', '0,1,2,3', or None)。None会自动选择可用GPU。

        # workers=1,                      # 数据加载时使用的工作线程数 (Number of worker threads for data loading)。
        # amp=True,                       # 是否使用自动混合精度 (AMP) 训练 (Automatic Mixed Precision)。
        patience=10,                    # 早停耐心值 (Early Stopping Patience): 连续N个epoch性能未提升则提前停止训练。
        
        # # ======================================================================
        # # 验证与导出 (Validation & Export)
        # # ======================================================================
        # val=True,                       # 是否在训练结束后进行一次最终验证 (Validate during training)。
        # split='val',                    # 用于验证的数据集划分 ('val', 'test', 'train')。
        # cache=False,                    # 是否缓存数据集到内存或磁盘以加快加载速度 ('ram', 'disk', or False)。
        # seed=0,                         # 设置随机种子以保证结果可复现 (Random seed for reproducibility)。
        # deterministic=True,             # 是否启用确定性算法，有助于可复现性。
        # profile=False,                  # 是否在训练时记录ONNX和TensorRT的速度。
        # format='pt',                    # 最终模型的导出格式 ('pt', 'torchscript', 'onnx', 'engine', etc.)。
        
        # # ======================================================================
        # # 任务特定参数 (Task-Specific Parameters)
        # # ======================================================================
        # box=7.5,                        # 边界框损失的权重 (Box loss gain)。
        # cls=0.5,                        # 分类损失的权重 (Classification loss gain)。
        # dfl=1.5,                        # DFL损失（用于边界框回归）的权重 (Distribution Focal Loss gain)。
        # pose=12.0,                      # 姿态损失的权重 (Pose loss gain, for pose estimation models)。
        # kobj=1.0,                       # 关键点可见性损失的权重 (Keypoint object loss gain)。
    )

    print("="*40)
    print("训练已启动！请查看 'runs/train' 目录下的输出")
    # print(f"最终模型和日志保存在: {results.save_dir}")
    print("="*40)
