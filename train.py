import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO('yolov11/cfg/models/11/yolov11_plate.yaml')  # 指定YOLO模型对象，并加载指定配置文件中的模型配置
    model.load('yolo11n.pt')  # 加载预训练的权重文件'yolov11s.pt'，加速训练并提升模型性能

    model.train(data='yolov11/cfg/datasets/car_plate.yaml',  # 指定训练数据集的配置文件路径，这个.yaml文件包含了数据集的路径和类别信息
                cache=True,  # 是否缓存数据集以加快后续训练速度，False表示不缓存
                imgsz=300,  # 指定训练时使用的图像尺寸，640表示将输入图像调整为640x640像素
                epochs=200,  # 设置训练的总轮数为200轮
                batch=4,  # 设置每个训练批次的大小为16，即每次更新模型时使用16张图片
                close_mosaic=10,  # 设置在训练结束前多少轮关闭 Mosaic 数据增强，10 表示在训练的最后 10 轮中关闭 Mosaic
                workers=2,  # 设置用于数据加载的线程数为8，更多线程可以加快数据加载速度
                patience=50,  # 在训练时，如果经过50轮性能没有提升，则停止训练（早停机制）
                device='cpu',  # 指定使用的设备，'0'表示使用第一块GPU进行训练
                optimizer='SGD'  # 设置优化器为SGD（随机梯度下降），用于模型参数更新
                )
