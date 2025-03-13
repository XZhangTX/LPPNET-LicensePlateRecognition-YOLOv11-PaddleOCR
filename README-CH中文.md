<h2>
<p>License plate recognition based on YOLOv11 and paddleOCR<p>  
<p>基于<strong>YOLOv11与<strong>PaddleOCR的车牌识别<p>
</h2>
车牌识别的项目一般由两部分组成，第一部分是定位，第二部分是识别。常用的组合是YOLO+CRNN，这个项目直接调用了PaddleOCR，不需要单独训练crnn，加之YOLOv11方便快捷的属性，极其适合小白直接部署，想要自行训练也很简单。

<h2>部署流程<h2>
<li>环境搭建</li></h2>

1.  搭建虚拟环境后可以直接安装文件中的requirement.txt

<!---->

    #运行以下代码
    conda create -n <your_env_name> python=3.10 #<your_env_name>是环境名，自行选择；python版本由于是运行yolov11，需要在3.8及其之上，本项目选用3.10
    conda activate <your_env_name>
    cd path#path 是项目代码所在路径
    pip install requirement.txt

1.2 也可直接安装ultralytics pillow tqdm opencv torch

<h2><li>项目运行</li>
<h4>
1.数据准备</h4>

本项目使用CCPD2021数据集进行训练，该数据集是黄色车牌数据集。想要增加识别范围与类别的小伙伴可以看训练指引。

数据集下载前往[：CCPD数据集](https://github.com/detectRecog/CCPD) |[CCPD2021黄牌数据集](https://aistudio.baidu.com/datasetdetail/101671)

另外：CCPD数据集没有相应的label文件，其标注信息均在文件名上，对于yolo训练，数据集中是要有label文件的，++/tools++文件夹中准备了++split.py++可以将文件名中的识别框位置提取成txt的label文件。
本项目也准备了经过labelimg标注过的适用于yolo的数据集，可以在/data文件夹下找到

<h4>
2.运行结果</h4>

运行前，需要先修改yaml下的图像路径

    #yolov11/cfg/datasets/car_plate.yaml
    #dataset下的数据路径
    #此处修改图像数据路径
    path: F:\LPPNET-LicensePlateRecognition-YOLOv11-PaddleOCR\car_plate # data path
    train: train/images  # train images
    val: val/images  # val images
    test: test/images  # test images

运行++run.py++,程序会处理++car\_plate++下的数据，结果保存在++result++下

    #run文件下可以修改输入以及输出路径
    input_dir = "car_plate/test/img"
    output_dir = "car_plate/results"

<h2><li>项目训练</li></h2>

运行++train.py++，程序会训练yolov11模型
模型可以更改的参数如下：

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

如果想让项目可以识别其他种类的车牌：需要准备不同种类的数据集以及相应的yaml文件，该项目的文件可以在++yolov11/cfg/datasets++以及++yolov11/cfg/model++下找到

<h2><li>参考及引用</li></h2>

项目参考
Github: <https://github.com/sirius-ai/LPRNet_Pytorch>
Github：<https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition.git>
YOLO：<https://github.com/ultralytics/ultralytics.git>
