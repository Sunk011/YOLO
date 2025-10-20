#!/usr/bin/env python3
"""
数据集组织工具 - 使用软链接方式组织图片和XML标注文件

功能：
- 支持处理多个指定目录
- 在每个目录下创建 images 和 xml 子目录
- 使用软链接将 .jpg 图片和 .xml 标注文件分别组织到对应目录

使用方法：
    python organize_dataset_symlinks.py /path/to/dataset1 /path/to/dataset2
    或
    python organize_dataset_symlinks.py  (交互式输入)
"""

import os
import sys
from pathlib import Path
from typing import List


def create_symlinks_for_directory(source_dir: str) -> None:
    """
    在指定目录下创建 images 和 xml 子目录，并使用软链接组织文件
    
    Args:
        source_dir: 源数据目录路径
    """
    source_path = Path(source_dir).resolve()
    
    # 检查源目录是否存在
    if not source_path.exists():
        print(f"❌ 错误: 目录不存在 - {source_path}")
        return
    
    if not source_path.is_dir():
        print(f"❌ 错误: 不是有效的目录 - {source_path}")
        return
    
    print(f"\n📁 处理目录: {source_path}")
    
    # 创建 images 和 xml 子目录
    images_dir = source_path / "images"
    xml_dir = source_path / "xml"
    
    images_dir.mkdir(exist_ok=True)
    xml_dir.mkdir(exist_ok=True)
    print(f"✓ 创建目录: {images_dir.name}/")
    print(f"✓ 创建目录: {xml_dir.name}/")
    
    # 统计信息
    jpg_count = 0
    xml_count = 0
    skipped_count = 0
    
    # 扫描源目录中的所有文件（不包括子目录）
    for file_path in source_path.iterdir():
        # 跳过目录
        if file_path.is_dir():
            continue
        
        file_name = file_path.name
        file_ext = file_path.suffix.lower()
        
        # 处理 .jpg 图片文件
        if file_ext in ['.jpg', '.jpeg']:
            target_link = images_dir / file_name
            
            # 如果软链接已存在，先删除
            if target_link.exists() or target_link.is_symlink():
                target_link.unlink()
            
            # 创建相对路径的软链接
            relative_source = os.path.relpath(file_path, images_dir)
            try:
                target_link.symlink_to(relative_source)
                jpg_count += 1
            except OSError as e:
                print(f"  ⚠ 警告: 创建软链接失败 {file_name} - {e}")
                skipped_count += 1
        
        # 处理 .xml 标注文件
        elif file_ext == '.xml':
            target_link = xml_dir / file_name
            
            # 如果软链接已存在，先删除
            if target_link.exists() or target_link.is_symlink():
                target_link.unlink()
            
            # 创建相对路径的软链接
            relative_source = os.path.relpath(file_path, xml_dir)
            try:
                target_link.symlink_to(relative_source)
                xml_count += 1
            except OSError as e:
                print(f"  ⚠ 警告: 创建软链接失败 {file_name} - {e}")
                skipped_count += 1
    
    # 输出统计信息
    print(f"\n📊 处理结果:")
    print(f"  - 图片文件: {jpg_count} 个")
    print(f"  - XML文件: {xml_count} 个")
    if skipped_count > 0:
        print(f"  - 跳过文件: {skipped_count} 个")
    print(f"✅ 完成处理: {source_path.name}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据集组织工具 - 软链接方式")
    print("=" * 60)
    
    directories: List[str] = []
    
    # 从命令行参数获取目录列表
    if len(sys.argv) > 1:
        directories = sys.argv[1:]
        print(f"\n📝 从命令行参数获取到 {len(directories)} 个目录")
    else:
        # 交互式输入
        print("\n请输入要处理的目录路径（每行一个，输入空行结束）:")
        while True:
            try:
                dir_path = input("目录路径: ").strip()
                if not dir_path:
                    break
                directories.append(dir_path)
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠ 用户取消输入")
                break
        
        if not directories:
            print("\n❌ 错误: 未指定任何目录")
            print("\n使用方法:")
            print("  python organize_dataset_symlinks.py /path/to/dataset1 /path/to/dataset2")
            print("  或直接运行脚本，交互式输入目录路径")
            sys.exit(1)
    
    # 处理每个目录
    print(f"\n准备处理 {len(directories)} 个目录...")
    success_count = 0
    
    for idx, directory in enumerate(directories, 1):
        print(f"\n[{idx}/{len(directories)}]", end=" ")
        try:
            create_symlinks_for_directory(directory)
            success_count += 1
        except Exception as e:
            print(f"❌ 处理失败: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"🎉 全部完成! 成功处理 {success_count}/{len(directories)} 个目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
