import os
import random
from PIL import Image, ImageDraw, ImageFont
import yaml

# --- Functions for loading class names (from your request) ---

def load_yolo_classes_from_yaml(yaml_path):
    """从YOLO格式的yaml文件中加载类别信息"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        if 'names' in data:
            if isinstance(data['names'], dict):
                # names is a dict: {0: 'class1', 1: 'class2'}
                return data['names']
            elif isinstance(data['names'], list):
                # names is a list: ['class1', 'class2']
                return {i: name for i, name in enumerate(data['names'])}
        
        print(f"警告: 在 {yaml_path} 中未找到 'names' 字段。")
        return {}
    
    except Exception as e:
        print(f"错误: 加载yaml文件 {yaml_path} 时出错: {e}")
        return {}

def load_classes_from_txt(txt_path):
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
        print(f"错误: 加载classes文件 {txt_path} 时出错: {e}")
        return {}

def load_class_names(file_path):
    """
    根据文件类型（.yaml或.txt）加载类别名称。
    """
    if not file_path or not os.path.exists(file_path):
        print("警告: 未提供或未找到类别定义文件。将仅显示类别ID。")
        return {}
    
    if file_path.lower().endswith('.yaml'):
        print(f"从YAML文件加载类别: {file_path}")
        return load_yolo_classes_from_yaml(file_path)
    elif file_path.lower().endswith('.txt'):
        print(f"从TXT文件加载类别: {file_path}")
        return load_classes_from_txt(file_path)
    else:
        print(f"警告: 不支持的类别文件格式 {file_path}。请使用 .yaml 或 .txt。")
        return {}

# --- Main visualization function ---

def resize_with_padding(img, target_size, padding_color=(128, 128, 128)):
    """
    将图片调整到目标尺寸，保持横纵比，多余部分用指定颜色填充
    
    参数:
    - img: PIL Image对象
    - target_size: 目标尺寸 (width, height)
    - padding_color: 填充颜色，默认为灰色
    
    返回: 调整后的PIL Image对象
    """
    original_width, original_height = img.size
    target_width, target_height = target_size
    
    # 计算缩放比例，保持横纵比
    scale = min(target_width / original_width, target_height / original_height)
    
    # 计算缩放后的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 缩放图片
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 创建目标尺寸的背景图片
    padded_img = Image.new('RGB', target_size, padding_color)
    
    # 计算居中位置
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # 将缩放后的图片粘贴到背景中央
    padded_img.paste(resized_img, (paste_x, paste_y))
    
    return padded_img

def visualize_and_stitch(image_dir, label_dir, output_path, class_file_path=None, grid_size=8, cell_size=640):
    """
    随机选择图片，将YOLO标签绘制在图上，然后拼接成网格大图。

    参数:
    - image_dir (str): 存放图片的目录路径。
    - label_dir (str): 存放YOLO txt标签的目录路径。
    - output_path (str): 拼接后大图的保存路径。
    - class_file_path (str, optional): 类别定义文件的路径 (.yaml or .txt)。
    - grid_size (int): 网格大小，默认8x8
    - cell_size (int): 每个单元格的尺寸，默认640x640像素
    """
    # --- 1. 获取并筛选图片文件 ---
    try:
        all_images = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        num_needed = grid_size * grid_size
        if len(all_images) < num_needed:
            print(f"警告：图片目录 {image_dir} 中图片数量不足{num_needed}张 ({len(all_images)} 张)，将使用所有可用图片。")
            selected_images = all_images
            # 如果图片不够，随机重复选择一些图片来填满网格
            while len(selected_images) < num_needed:
                selected_images.extend(random.choices(all_images, k=min(len(all_images), num_needed - len(selected_images))))
        else:
            selected_images = random.sample(all_images, num_needed)
        
        print(f"已从 {len(all_images)} 张图片中选择 {len(selected_images)} 张。")
    except FileNotFoundError:
        print(f"错误：找不到图片目录 {image_dir}")
        return

    # --- 2. 加载类别名称 ---
    class_names = load_class_names(class_file_path)

    # 定义更多颜色，采用更鲜艳的配色方案
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8C471", "#82E0AA", "#F1948A", "#85C1E9", "#D5A6BD",
        "#AED6F1", "#A9DFBF", "#F9E79F", "#D7BDE2", "#A3E4D7"
    ]
    
    # 尝试加载字体，使用更小的字体以适应更小的图片
    font_size = max(12, cell_size // 40)  # 根据cell_size动态调整字体大小
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        try:
            # 尝试加载其他常见字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=font_size)
        except IOError:
            print("警告：未找到系统字体，将使用PIL默认字体。")
            font = ImageFont.load_default()

    processed_images = []
    
    print(f"开始处理 {len(selected_images)} 张图片...")
    
    # --- 3. 遍历处理每张选中的图片 ---
    for i, img_name in enumerate(selected_images):
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(selected_images)} 张图片...")
            
        img_path = os.path.join(image_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)

        try:
            with Image.open(img_path).convert("RGB") as img:
                draw = ImageDraw.Draw(img)
                img_width, img_height = img.size

                # 绘制标注框
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f.readlines():
                            line = line.strip()
                            if not line:
                                continue
                                
                            try:
                                parts = line.split()
                                if len(parts) < 5:
                                    continue
                                    
                                class_id = int(float(parts[0]))
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                # 验证坐标是否在有效范围内
                                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                       0 <= width <= 1 and 0 <= height <= 1):
                                    continue
                                
                                # 转换为像素坐标
                                box_w = width * img_width
                                box_h = height * img_height
                                x1 = (x_center * img_width) - (box_w / 2)
                                y1 = (y_center * img_height) - (box_h / 2)
                                x2 = x1 + box_w
                                y2 = y1 + box_h
                                
                                # 确保坐标在图片范围内
                                x1 = max(0, min(x1, img_width))
                                y1 = max(0, min(y1, img_height))
                                x2 = max(0, min(x2, img_width))
                                y2 = max(0, min(y2, img_height))
                                
                                # 选择颜色
                                color = colors[class_id % len(colors)]
                                
                                # 绘制边界框，使用更细的线条
                                line_width = max(1, img_width // 200)
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                                
                                # 获取类别名称
                                label_text = class_names.get(class_id, str(class_id))
                                
                                # 绘制标签背景和文字
                                try:
                                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                                    text_w = text_bbox[2] - text_bbox[0]
                                    text_h = text_bbox[3] - text_bbox[1]
                                except:
                                    # 如果textbbox不可用，使用估算
                                    text_w = len(label_text) * font_size * 0.6
                                    text_h = font_size
                                
                                # 确保标签不会超出图片边界
                                label_x = min(x1, img_width - text_w - 4)
                                label_y = max(0, y1 - text_h - 2)
                                
                                # 绘制标签背景
                                draw.rectangle([label_x, label_y, label_x + text_w + 4, label_y + text_h + 2], 
                                             fill=color)
                                
                                # 绘制文字
                                draw.text((label_x + 2, label_y + 1), label_text, fill="white", font=font)

                            except (ValueError, IndexError) as e:
                                print(f"警告：跳过格式错误的行 '{line}' 在文件 {label_path} 中。错误: {e}")
                                continue

                # 调整图片尺寸，保持横纵比并添加灰色填充
                processed_img = resize_with_padding(img, (cell_size, cell_size))
                processed_images.append(processed_img)
                
        except Exception as e:
            print(f"警告：处理图片 {img_name} 时出错: {e}")
            # 创建一个空白图片作为占位符
            blank_img = Image.new('RGB', (cell_size, cell_size), (128, 128, 128))
            processed_images.append(blank_img)

    # --- 4. 拼接图片 ---
    total_width = cell_size * grid_size
    total_height = cell_size * grid_size
    stitched_image = Image.new('RGB', (total_width, total_height), (64, 64, 64))  # 深灰色背景
    
    print(f"开始拼接 {grid_size}x{grid_size} 图片...")
    for index, img in enumerate(processed_images):
        row = index // grid_size
        col = index % grid_size
        x = col * cell_size
        y = row * cell_size
        stitched_image.paste(img, (x, y))

    # --- 5. 保存最终结果 ---
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: 
            os.makedirs(output_dir, exist_ok=True)
        stitched_image.save(output_path, quality=95)
        print(f"拼接完成！图片已保存至: {output_path}")
        print(f"最终图片尺寸: {total_width}x{total_height} 像素")
    except Exception as e:
        print(f"错误：保存图片失败。{e}")

# --- 使用示例 ---
if __name__ == '__main__':
    # --- 配置参数 ---
    
    # 1. 指定你的图片目录
    # Windows 示例: "C:\\Users\\YourUser\\Desktop\\my_dataset\\images"
    # Linux/macOS 示例: "/home/user/data/my_dataset/images"
    image_directory = "/home/sk/datasets/VisDrone_yolo/images/val"
    
    # 2. 指定你的标签目录 (YOLO .txt 格式)
    # Windows 示例: "C:\\Users\\YourUser\\Desktop\\my_dataset\\labels"
    # Linux/macOS 示例: "/home/user/data/my_dataset/labels"
    label_directory = "/home/sk/datasets/VisDrone_yolo/labels/val"
    
    # 3. 指定拼接后大图的保存路径和文件名
    output_image_path = "./tmp/stitched_visualization.jpg"
    
    # 4. (重要) 指定你的类别定义文件路径 (data.yaml 或 classes.txt)
    #    - 如果是YOLO的 'data.yaml' 文件, 路径示例: "C:\\data\\coco\\data.yaml"
    #    - 如果是 'classes.txt' 文件, 路径示例: "/home/user/data/classes.txt"
    #    - 如果设为 None 或路径无效, 则只在图上显示类别ID数字
    class_definition_file = "/home/sk/datasets/VisDrone_yolo/VisDrone.yaml"

    # --- 运行主函数 ---
    visualize_and_stitch(
        image_dir=image_directory,
        label_dir=label_directory,
        output_path=output_image_path,
        class_file_path=class_definition_file
    )