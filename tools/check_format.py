# -*- coding: utf-8 -*-
"""
输入指定的图像目录和标注目录，检查它们之间的对应关系，验证标注文件格式的正确性，并生成详细的检查报告。
Author: sunkang
Date: 2025-09-15
"""
import os
import cv2
import re
import random
from datetime import datetime  # 替换os.popen('date')，跨平台兼容

def validate_yolo_dataset(images_dir, labels_dir, max_class_id=None, report_path=None):
    """
    检查YOLO格式数据集的完整性和正确性
    
    Args:
        images_dir: 图像目录路径
        labels_dir: 标注目录路径
        max_class_id: 最大类别ID（可选，例如4类则设为3，设为None则不检查）
        report_path: 保存报告的路径（可选，如果未指定则保存到图像目录的上级目录）
    
    Returns:
        dict: 包含检查结果的字典
    """
    
    # 图像文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.PNG')
    
    # 记录所有错误
    all_errors = []
    
    # 正则表达式匹配YOLO标注行（class_id + 4个浮点数）
    yolo_line_pattern = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
    
    print(f"开始检查数据集:")
    print(f"图像目录: {images_dir}")
    print(f"标注目录: {labels_dir}\n")
    
    # 检查目录是否存在
    if not os.path.exists(images_dir):
        error = f"错误: 图像目录不存在 - {images_dir}"
        print(error)
        return {"success": False, "error": error, "report_path": None}
        
    if not os.path.exists(labels_dir):
        error = f"错误: 标注目录不存在 - {labels_dir}"
        print(error)
        return {"success": False, "error": error, "report_path": None}
    
    # 获取所有图像文件和标注文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(image_extensions)]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print(f"图像文件数量: {len(image_files)}")
    print(f"标注文件数量: {len(label_files)}\n")
    
    # 记录各文件夹文件数量
    folder_counts = {
        "images": len(image_files),
        "labels": len(label_files)
    }
    
    # 提取文件名（不含扩展名）用于匹配
    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    label_basenames = {os.path.splitext(f)[0] for f in label_files}
    
    # 1. 检查图像与标注文件的对应性
    images_without_label = image_basenames - label_basenames
    labels_without_image = label_basenames - image_basenames
    
    correspondence_info = {
        "total_images": len(image_files),
        "total_labels": len(label_files),
        "matched_pairs": len(image_basenames & label_basenames),
        "images_without_label": list(images_without_label),
        "labels_without_image": list(labels_without_image)
    }
    
    if images_without_label:
        error = f"警告: {len(images_without_label)}个图像没有对应标注文件"
        print(error)
        all_errors.append(error)
        all_errors.extend([f"  - {f}.*" for f in sorted(images_without_label)[:5]])
    
    if labels_without_image:
        error = f"警告: {len(labels_without_image)}个标注文件没有对应图像"
        print(error)
        all_errors.append(error)
        all_errors.extend([f"  - {f}.txt" for f in sorted(labels_without_image)[:5]])
    
    # 2. 检查图像文件完整性（随机抽查）
    broken_images = []
    sample_size = min(100, len(image_files))
    sampled_images = random.sample(image_files, sample_size) if image_files else []
    
    for img_file in sampled_images:
        img_path = os.path.join(images_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                broken_images.append(img_file)
        except Exception:
            broken_images.append(img_file)
    
    if broken_images:
        error = f"警告: {len(broken_images)}个图像文件损坏或无法打开"
        print(error)
        all_errors.append(error)
        all_errors.extend([f"  - {f}" for f in broken_images[:5]])
    
    # 3. 检查标注文件格式
    invalid_annotations = []
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 检查空标注文件
            if not any(line.strip() for line in lines):
                invalid_annotations.append(f"{label_file} (空文件)")
                continue
            
            # 检查每行格式
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                match = yolo_line_pattern.match(line)
                if not match:
                    invalid_annotations.append(
                        f"{label_file} (第{line_num}行格式错误: '{line}')"
                    )
                    continue
                
                # 检查坐标范围
                class_id = int(match.group(1))
                cx, cy, w, h = map(float, match.groups()[1:])
                
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    invalid_annotations.append(
                        f"{label_file} (第{line_num}行坐标超出0-1范围)"
                    )
                
                # 检查类别ID
                if max_class_id is not None and class_id > max_class_id:
                    invalid_annotations.append(
                        f"{label_file} (第{line_num}行类别ID {class_id} 超过最大允许值 {max_class_id})"
                    )
                elif class_id < 0:
                    invalid_annotations.append(
                        f"{label_file} (第{line_num}行类别ID为负数: {class_id})"
                    )
        
        except Exception as e:
            invalid_annotations.append(f"{label_file} (读取错误: {str(e)})")
    
    if invalid_annotations:
        error = f"警告: {len(invalid_annotations)}个标注文件格式错误"
        print(error)
        all_errors.append(error)
        all_errors.extend([f"  - {err}" for err in invalid_annotations[:5]])
    
    print("检查完成\n")
    
    # 输出文件数量汇总和对应关系
    print("===== 文件数量和对应关系汇总 =====")
    print(f"图像文件数量: {correspondence_info['total_images']}")
    print(f"标注文件数量: {correspondence_info['total_labels']}")
    print(f"匹配的文件对数: {correspondence_info['matched_pairs']}")
    print(f"无标注的图像数: {len(correspondence_info['images_without_label'])}")
    print(f"无图像的标注数: {len(correspondence_info['labels_without_image'])}")
    print()
    
    # 生成检查报告
    print("===== 数据集检查总结 =====")
    success = len(all_errors) == 0
    
    if success:
        print("恭喜！数据集检查通过，未发现问题。")
    else:
        print(f"共发现{len(all_errors)}个问题：")
        for i, error in enumerate(all_errors, 1):
            print(f"{i}. {error}")
    
    # 确定报告保存路径
    if report_path is None:
        # 如果未指定报告路径，保存到图像目录的上级目录
        parent_dir = os.path.dirname(images_dir)
        report_path = os.path.join(parent_dir, "dataset_validation_report.txt")
    
    # 保存详细报告到文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("数据集检查报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"图像目录: {images_dir}\n")
        f.write(f"标注目录: {labels_dir}\n")
        f.write(f"最大类别ID: {max_class_id if max_class_id is not None else '未限制'}\n\n")
        
        f.write("文件数量统计：\n")
        f.write(f"图像文件数量: {correspondence_info['total_images']}\n")
        f.write(f"标注文件数量: {correspondence_info['total_labels']}\n")
        f.write(f"匹配的文件对数: {correspondence_info['matched_pairs']}\n")
        f.write(f"无标注的图像数: {len(correspondence_info['images_without_label'])}\n")
        f.write(f"无图像的标注数: {len(correspondence_info['labels_without_image'])}\n\n")
        
        if correspondence_info['images_without_label']:
            f.write("无标注的图像文件：\n")
            for img in correspondence_info['images_without_label']:
                f.write(f"  - {img}.*\n")
            f.write("\n")
        
        if correspondence_info['labels_without_image']:
            f.write("无图像的标注文件：\n")
            for label in correspondence_info['labels_without_image']:
                f.write(f"  - {label}.txt\n")
            f.write("\n")
        
        f.write("问题列表：\n")
        if not all_errors:
            f.write("未发现问题\n")
        else:
            for i, error in enumerate(all_errors, 1):
                f.write(f"{i}. {error}\n")
    
    print(f"\n详细报告已保存至: {report_path}")
    
    # 返回检查结果
    result = {
        "success": success,
        "report_path": report_path,
        "correspondence_info": correspondence_info,
        "error_count": len(all_errors),
        "errors": all_errors
    }
    
    return result

if __name__ == "__main__":
    # 示例使用方式
    # 方式1: 直接指定图像和标注目录
    images_dir = "/home/sk/datasets/car_vis_v2/images/train"  # 图像目录路径
    labels_dir = "/home/sk/datasets/car_vis_v2/labels/train"  # 标注目录路径
    max_class_id = 3  # 最大类别ID（例如4类则设为3，设为None则不检查类别范围）
    report_path = "/home/sk/datasets/car_vis_v2/validation_report.txt"  # 报告保存路径（可选）
    
    # 方式2: 检查整个数据集的train和val集合
    # 如果要检查完整数据集，可以分别调用train和val
    
    print("正在检查训练集...")
    result = validate_yolo_dataset(images_dir, labels_dir, max_class_id, report_path)
    
    if result["success"]:
        print(f"✅ 数据集检查通过！")
    else:
        print(f"❌ 发现 {result['error_count']} 个问题")
    
    print(f"📊 对应关系统计:")
    print(f"   匹配文件对: {result['correspondence_info']['matched_pairs']}")
    print(f"   无标注图像: {len(result['correspondence_info']['images_without_label'])}")
    print(f"   无图像标注: {len(result['correspondence_info']['labels_without_image'])}")
    print(f"📄 详细报告: {result['report_path']}")
    
    # 如果需要检查验证集，取消下面的注释
    # print("\n" + "="*50)
    # print("正在检查验证集...")
    # val_images_dir = "/home/sk/datasets/car_vis_v2/images/val"
    # val_labels_dir = "/home/sk/datasets/car_vis_v2/labels/val"
    # val_report_path = "/home/sk/datasets/car_vis_v2/val_validation_report.txt"
    # val_result = validate_yolo_dataset(val_images_dir, val_labels_dir, max_class_id, val_report_path)
    
    # print(f"📄 验证集报告: {val_result['report_path']}")