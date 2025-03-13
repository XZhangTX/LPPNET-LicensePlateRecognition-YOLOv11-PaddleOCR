import os
import random
from sklearn.model_selection import train_test_split

def read_image_paths(data_dir):
    """
    读取指定目录下的所有图片路径。
    :param data_dir: 数据集目录路径
    :return: 包含所有图片路径的列表
    """
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # 构造完整路径
                full_path = os.path.join(root, file)
                # 将路径转换为相对于data_dir的相对路径，方便后续处理
                relative_path = os.path.relpath(full_path, data_dir)
                image_paths.append(relative_path)
    return image_paths

def split_dataset(image_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    按照给定的比例划分数据集。
    :param image_paths: 图片路径列表
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :return: 划分后的训练集、验证集和测试集路径列表
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须等于1"

    # 第一步：将数据集划分为训练集+验证集 和 测试集
    train_val_set, test_set = train_test_split(image_paths, test_size=test_ratio, random_state=42)

    # 第二步：将训练集+验证集进一步划分为训练集和验证集
    train_set, val_set = train_test_split(train_val_set, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

    return train_set, val_set, test_set

def save_splits_to_file(splits, output_dir):
    """
    将划分后的数据集保存为txt文件。
    :param splits: 包含三个元素的元组，分别为训练集、验证集和测试集路径列表
    :param output_dir: 输出目录路径
    """
    names = ['train', 'val', 'test']
    for name, split in zip(names, splits):
        output_path = os.path.join(output_dir, f'{name}.txt')
        with open(output_path, 'w') as f:
            for path in split:
                f.write(f"{path}\n")
        print(f"Saved {len(split)} paths to {output_path}")

# 新数据集目录
new_data_dir = r"F:\ultralytics\data\CCPD2019.tar\CCPD2019\ccpd_yellow"

# 输出目录
output_dir = r"/data/output"
os.makedirs(output_dir, exist_ok=True)

# 读取所有图片路径
image_paths = read_image_paths(new_data_dir)

# 打乱顺序
random.shuffle(image_paths)

# 划分数据集
train_set, val_set, test_set = split_dataset(image_paths)

# 保存划分结果
save_splits_to_file((train_set, val_set, test_set), output_dir)