import os
import random
import shutil
from PIL import Image


def parse_filename(filename):
    """
    解析文件名，提取车牌的位置信息。
    :param filename: 文件名字符串
    :return: 四个顶点坐标
    """
    parts = filename.split('.')[0].split('-')

    # 提取四个顶点的坐标
    vertices_part = parts[3]
    vertices = [tuple(map(int, v.split('&'))) for v in vertices_part.split('_')]

    return vertices


def create_label_file(image_path, label_dir):
    """
    根据图像路径创建YOLO格式的标签文件。
    :param image_path: 图像文件路径
    :param label_dir: 输出标签文件的目录路径
    """
    filename = os.path.basename(image_path)
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    vertices = parse_filename(filename)
    x_coords = [vertex[0] for vertex in vertices]
    y_coords = [vertex[1] for vertex in vertices]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    center_x = (min_x + max_x) / 2.0 / img_width
    center_y = (min_y + max_y) / 2.0 / img_height
    bbox_width = (max_x - min_x) / img_width
    bbox_height = (max_y - min_y) / img_height

    content = f"0 {center_x} {center_y} {bbox_width} {bbox_height}\n"

    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
    with open(label_path, 'w') as f:
        f.write(content)


def split_dataset(data_dir, image_output_root, label_output_root, train_ratio=0.8):
    """
    将数据集划分为训练集和测试集，并为每个子集生成对应的标签文件，同时复制图片到相应目录。
    :param data_dir: 数据集目录路径
    :param image_output_root: 输出图像根目录路径
    :param label_output_root: 输出标签根目录路径
    :param train_ratio: 训练集所占比例，默认值为0.8
    """
    images = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files if
              file.endswith('.jpg') or file.endswith('.png')]
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)
    train_set = images[:train_size]
    test_set = images[train_size:]

    print(f"Total images: {len(images)}, Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    # 创建输出目录
    os.makedirs(os.path.join(image_output_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(label_output_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(image_output_root, 'test'), exist_ok=True)
    os.makedirs(os.path.join(label_output_root, 'test'), exist_ok=True)

    for image_path in train_set:
        create_label_file(image_path, os.path.join(label_output_root, 'train'))
        shutil.copy(image_path, os.path.join(image_output_root, 'train'))
    for image_path in test_set:
        create_label_file(image_path, os.path.join(label_output_root, 'test'))
        shutil.copy(image_path, os.path.join(image_output_root, 'test'))


# 配置你的参数
data_dir = '../data/plate/ccpd_yellow_img'  # 替换为你的数据集路径
image_output_root = 'data/plate/images'  # 替换为你想要保存图像结果的根目录
label_output_root = 'data/plate/labels2'  # 替换为你想要保存标签结果的根目录

if __name__ == '__main__':
    split_dataset(data_dir, image_output_root, label_output_root)


