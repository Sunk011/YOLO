# -*- coding: utf-8 -*-
"""
yolo格式类别分布查看
输入标注目录，输出类别分布统计信息，重点分析每张图片的目标数量分布，并将结果保存到txt文件
Author: sunkang
Date: 2025-09-15
Updated: 2025-09-27
"""

import os
import glob
import yaml
import argparse
from collections import Counter
from datetime import datetime


class YOLOLabelAnalyzer:
    """YOLO数据集标签分布分析器"""
    
    def __init__(self, labels_path, classes_file=None, output_dir="./output"):
        """
        初始化分析器
        
        Args:
            labels_path: 标注目录路径
            classes_file: 类别文件路径 (yaml或txt文件)
            output_dir: 输出目录路径
        """
        self.labels_path = labels_path
        self.classes_file = classes_file
        self.output_dir = output_dir
        self.class_names = {}
        self.labels_counter = Counter()
        self.total_objects = 0
        
        # 新增：用于存储新功能的统计数据
        self.total_label_files = 0  # 标签文件总数
        self.objects_per_image = Counter()  # 每张图片的目标数量分布
        self.class_objects_per_image = {}  # 每个类别在图片中的数量分布
        
        # 验证路径
        self._validate_paths()
        
        # 加载类别信息
        self._load_class_names()
    
    def _validate_paths(self):
        """验证路径有效性"""
        if not os.path.exists(self.labels_path):
            raise ValueError(f"标注目录不存在: {self.labels_path}")
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")
    
    def _load_yolo_classes_from_yaml(self, yaml_path):
        """从YOLO格式的yaml文件中加载类别信息"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            # 获取类别名称列表
            if 'names' in data:
                if isinstance(data['names'], dict):
                    # 如果names是字典格式 {0: 'class1', 1: 'class2'}
                    return data['names']
                elif isinstance(data['names'], list):
                    # 如果names是列表格式 ['class1', 'class2']
                    return {i: name for i, name in enumerate(data['names'])}
            
            print(f"Warning: No 'names' field found in {yaml_path}")
            return {}
        
        except Exception as e:
            print(f"Error loading yaml file {yaml_path}: {e}")
            return {}
    
    def _load_classes_from_txt(self, txt_path):
        """从classes.txt文件中加载类别信息"""
        try:
            class_names = {}
            with open(txt_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    class_name = line.strip()
                    if class_name:
                        class_names[i] = class_name
            return class_names
        
        except Exception as e:
            print(f"Error loading classes file {txt_path}: {e}")
            return {}
    
    def _load_class_names(self):
        """加载类别名称映射"""
        if self.classes_file is None:
            # 在标注目录中查找yaml或classes.txt文件
            yaml_files = glob.glob(os.path.join(self.labels_path, '*.yaml')) + \
                        glob.glob(os.path.join(self.labels_path, '*.yml'))
            txt_files = glob.glob(os.path.join(self.labels_path, 'classes.txt'))
            
            if yaml_files:
                self.classes_file = yaml_files[0]
                print(f"找到yaml文件: {self.classes_file}")
                self.class_names = self._load_yolo_classes_from_yaml(self.classes_file)
            elif txt_files:
                self.classes_file = txt_files[0]
                print(f"找到classes.txt文件: {self.classes_file}")
                self.class_names = self._load_classes_from_txt(self.classes_file)
            else:
                print("未找到类别文件，将使用类别索引作为名称")
        else:
            # 使用指定的类别文件
            if os.path.exists(self.classes_file):
                if self.classes_file.endswith(('.yaml', '.yml')):
                    self.class_names = self._load_yolo_classes_from_yaml(self.classes_file)
                elif self.classes_file.endswith('.txt'):
                    self.class_names = self._load_classes_from_txt(self.classes_file)
                else:
                    print(f"Warning: 不支持的文件格式: {self.classes_file}")
            else:
                print(f"类别文件不存在: {self.classes_file}")
                self.classes_file = None

                print(f"类别文件不存在: {self.classes_file}")
                self.classes_file = None
    
    def analyze_labels_distribution(self):
        """分析标签分布"""
        print(f"开始处理标注文件...")
        
        # 遍历数据集目录下的所有.txt标注文件
        txt_files = glob.glob(os.path.join(self.labels_path, "**", "*.txt"), recursive=True)
        
        if not txt_files:
            print(f"在 {self.labels_path} 中未找到txt文件")
            return False
        
        # 过滤掉可能的配置文件
        exclude_files = ['classes.txt', 'train.txt', 'val.txt', 'test.txt']
        txt_files = [f for f in txt_files if os.path.basename(f) not in exclude_files]
        
        print(f"处理 {len(txt_files)} 个标注文件...")
        
        # 统计标签文件总数
        self.total_label_files = len(txt_files)
        
        # 初始化类别在图片中的分布统计
        for class_id in range(100):  # 假设最多100个类别，后续会根据实际情况调整
            self.class_objects_per_image[class_id] = Counter()
        
        for label_file in txt_files:
            try:
                file_objects = Counter()  # 当前文件中各类别的目标数量
                total_objects_in_file = 0  # 当前文件中的总目标数量
                
                with open(label_file, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()
                        if line:  # 跳过空行
                            parts = line.split()
                            if len(parts) >= 5:  # YOLO格式至少有5个值：class_id x y w h
                                class_id = int(parts[0])
                                self.labels_counter[class_id] += 1
                                file_objects[class_id] += 1
                                total_objects_in_file += 1
                
                # 统计每张图片的目标数量分布
                self.objects_per_image[total_objects_in_file] += 1
                
                # 统计各类别在图片中的数量分布
                for class_id, count in file_objects.items():
                    self.class_objects_per_image[class_id][count] += 1
                
            except Exception as e:
                print(f"处理文件 {label_file} 时出错: {e}")
        
        # 清理未使用的类别
        self.class_objects_per_image = {k: v for k, v in self.class_objects_per_image.items() 
                                       if v and k in self.labels_counter}
        
        if not self.labels_counter:
            print("数据集中未找到标签")
            return False
        
        self.total_objects = sum(self.labels_counter.values())
        return True
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("数据集基本统计信息")
        print("="*60)
        print(f"标签文件总数: {self.total_label_files}")
        print(f"总目标数量: {self.total_objects}")
        
        print("\n" + "="*60)
        print("类别分布统计")
        print("="*60)
        for class_id, count in sorted(self.labels_counter.items()):
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            percentage = (count / self.total_objects) * 100
            print(f'{class_name} (ID: {class_id}): {count} ({percentage:.1f}%)')
        
        # 打印每张图片目标数量分布
        print("\n" + "="*60)
        print("图片目标数量分布")
        print("="*60)
        for obj_count, img_count in sorted(self.objects_per_image.items()):
            percentage = (img_count / self.total_label_files) * 100
            print(f'{obj_count}个目标: {img_count}张图片 ({percentage:.1f}%)')
        
        # 打印各类别在图片中的数量分布
        print("\n" + "="*60)
        print("各类别在图片中的分布")
        print("="*60)
        for class_id in sorted(self.class_objects_per_image.keys()):
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            print(f"\n类别 {class_name} (ID: {class_id}):")
            class_distribution = self.class_objects_per_image[class_id]
            for obj_count, img_count in sorted(class_distribution.items()):
                print(f'  {obj_count}个目标: {img_count}张图片')
        
        print("\n" + "="*60)
    
    def save_statistics_to_txt(self, save=True):
        """将统计结果保存到txt文件"""
        if not save:
            return
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建输出内容
        output_lines = []
        output_lines.append("YOLO数据集分析报告")
        output_lines.append("=" * 60)
        output_lines.append(f"分析时间: {timestamp}")
        output_lines.append(f"标注目录: {self.labels_path}")
        output_lines.append(f"类别文件: {self.classes_file if self.classes_file else '自动查找'}")
        output_lines.append("")
        
        # 数据集基本统计信息
        output_lines.append("数据集基本统计信息")
        output_lines.append("-" * 40)
        output_lines.append(f"标签文件总数: {self.total_label_files}")
        output_lines.append(f"总目标数量: {self.total_objects}")
        output_lines.append("")
        
        # 类别分布统计
        output_lines.append("类别分布统计")
        output_lines.append("-" * 40)
        for class_id, count in sorted(self.labels_counter.items()):
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            percentage = (count / self.total_objects) * 100
            output_lines.append(f'{class_name} (ID: {class_id}): {count} ({percentage:.1f}%)')
        output_lines.append("")
        
        # 图片目标数量分布 - 这是重点内容
        output_lines.append("图片目标数量分布")
        output_lines.append("-" * 40)
        for obj_count, img_count in sorted(self.objects_per_image.items()):
            percentage = (img_count / self.total_label_files) * 100
            output_lines.append(f'有{obj_count}个目标的图片: {img_count}张 ({percentage:.1f}%)')
        output_lines.append("")
        
        # 各类别在图片中的分布
        output_lines.append("各类别在图片中的分布")
        output_lines.append("-" * 40)
        for class_id in sorted(self.class_objects_per_image.keys()):
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            output_lines.append(f"\n类别 {class_name} (ID: {class_id}):")
            class_distribution = self.class_objects_per_image[class_id]
            for obj_count, img_count in sorted(class_distribution.items()):
                output_lines.append(f'  有{obj_count}个该类别目标的图片: {img_count}张')
        
        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append("分析完成")
        
        # 保存到文件
        filename = os.path.join(self.output_dir, "YOLO_Dataset_Analysis_Report.txt")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"分析报告已保存: {filename}")
        except Exception as e:
            print(f"保存分析报告时出错: {e}")
    
    def analyze_dataset_structure(self):
        """分析YOLO数据集结构"""
        print(f"分析数据集结构: {self.labels_path}")
        
        # 查找图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.labels_path, ext)))
        
        # 查找标注文件
        txt_files = glob.glob(os.path.join(self.labels_path, '*.txt'))
        exclude_files = ['classes.txt', 'train.txt', 'val.txt', 'test.txt']
        annotation_files = [f for f in txt_files if os.path.basename(f) not in exclude_files]
        
        # 查找yaml文件
        yaml_files = glob.glob(os.path.join(self.labels_path, '*.yaml')) + \
                    glob.glob(os.path.join(self.labels_path, '*.yml'))
        
        print(f"图片文件数量: {len(image_files)}")
        print(f"标注文件数量: {len(annotation_files)}")
        print(f"YAML文件数量: {len(yaml_files)}")
        
        if yaml_files:
            print(f"YAML文件: {yaml_files[0]}")
    
    def run_analysis(self, show=False, save=True):
        """运行完整分析"""
        print("=" * 60)
        print("YOLO数据集标签分布分析")
        print("=" * 60)
        print(f"标注目录: {self.labels_path}")
        print(f"类别文件: {self.classes_file if self.classes_file else '自动查找'}")
        print(f"输出目录: {self.output_dir}")
        print(f"保存报告: {'是' if save else '否'}")
        print("=" * 60)
        
        # 分析标签分布
        if not self.analyze_labels_distribution():
            return False
        
        # 打印统计信息
        self.print_statistics()
        
        # 保存统计结果到txt文件
        self.save_statistics_to_txt(save=save)
        
        print("\n分析完成！")
        return True


# 为了兼容性，保留原有的函数接口
def count_yolo_labels_distribution(labels_path, classes_file=None, output_dir=".", show=False, save=True):
    """
    统计YOLO格式数据集的类别分布 (兼容性函数)
    
    Args:
        labels_path: 标注目录路径，包含标注txt文件
        classes_file: 类别文件路径 (yaml文件或classes.txt文件)，如果为None则在labels_path中查找
        output_dir: 分析结果保存路径
        show: 已废弃，保留为兼容性
        save: 是否保存分析报告到txt文件
    """
    try:
        analyzer = YOLOLabelAnalyzer(labels_path, classes_file, output_dir)
        return analyzer.run_analysis(show=show, save=save)
    except Exception as e:
        print(f"分析过程中出错: {e}")
        return False



def analyze_yolo_dataset_structure(dataset_path):
    """分析YOLO数据集结构 (兼容性函数)"""
    try:
        analyzer = YOLOLabelAnalyzer(dataset_path)
        analyzer.analyze_dataset_structure()
    except Exception as e:
        print(f"分析数据集结构时出错: {e}")


def get_user_inputs():
    """获取用户输入的参数"""
    print("=" * 60)
    print("YOLO数据集标签分布分析工具")
    print("=" * 60)
    
    # 获取标注目录
    while True:
        labels_path = input("请输入标注目录路径: ").strip()
        if os.path.exists(labels_path):
            break
        else:
            print(f"错误：目录 '{labels_path}' 不存在，请重新输入")
    
    # 获取类别文件
    classes_file = input("请输入类别文件路径 (yaml/classes.txt，留空自动查找): ").strip()
    if classes_file and not os.path.exists(classes_file):
        print(f"警告：类别文件 '{classes_file}' 不存在，将自动查找")
        classes_file = None
    
    # 获取输出目录
    output_dir = input("请输入可视化结果保存路径 (默认: ./output): ").strip()
    if not output_dir:
        output_dir = "./output"
    
    return labels_path, classes_file, output_dir


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLO数据集标签分布分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 交互式输入模式
  python yolo_labels_distribution.py -i
  
  # 命令行模式 - 只指定标注目录，自动查找类别文件
  python yolo_labels_distribution.py -l /path/to/labels
  
  # 命令行模式 - 指定所有参数
  python yolo_labels_distribution.py -l /path/to/labels -c /path/to/classes.yaml -o ./results
  
  # 显示图表但不保存
  python yolo_labels_distribution.py -l /path/to/labels --show --no-save
        ''')
    
    parser.add_argument('--labels-path', '-l', type=str, required=False,
                        help='标注目录路径')
    parser.add_argument('--classes-file', '-c', type=str, required=False,
                        help='类别文件路径 (yaml文件或classes.txt)')
    parser.add_argument('--output-dir', '-o', type=str, default='./output',
                        help='可视化结果保存路径 (默认: ./output)')
    parser.add_argument('--show', action='store_true',
                        help='显示图表')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存图表文件')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='交互式输入模式')
    
    return parser.parse_args()


if __name__ == '__main__':
    # 使用面向对象的方式 - 推荐用法
    try:
        # 创建分析器实例 - 使用VisDrone数据集进行演示
        analyzer = YOLOLabelAnalyzer(
            labels_path='/home/sk/datasets/VisDrone_yolo/labels/train',  # 使用训练集标签
            classes_file='/home/sk/datasets/VisDrone_yolo/VisDrone.yaml',  # 指定类别文件
            output_dir='./output'  # 输出目录
        )
        
        # 运行分析
        analyzer.run_analysis(show=False, save=True)
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        print("请检查标注目录路径是否正确，或使用交互式模式：")
    
    # 以下是命令行和交互式模式的代码（可选）
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 确定输入方式
    if args.interactive or not args.labels_path:
        # 交互式输入
        labels_path, classes_file, output_dir = get_user_inputs()
        show = True  # 交互模式默认显示图表
        save = True  # 交互模式默认保存图表
    else:
        # 命令行参数输入
        labels_path = args.labels_path
        classes_file = args.classes_file
        output_dir = args.output_dir
        show = args.show
        save = not args.no_save
    
    # 验证标注目录
    if not os.path.exists(labels_path):
        print(f"错误：标注目录 '{labels_path}' 不存在")
        exit(1)
    
    # 验证类别文件
    if classes_file and not os.path.exists(classes_file):
        print(f"警告：类别文件 '{classes_file}' 不存在，将自动查找")
        classes_file = None
    
    try:
        # 使用类进行分析
        analyzer = YOLOLabelAnalyzer(labels_path, classes_file, output_dir)
        analyzer.run_analysis(show=show, save=save)
    except Exception as e:
        print(f"分析过程中出错: {e}")
    """
    
 