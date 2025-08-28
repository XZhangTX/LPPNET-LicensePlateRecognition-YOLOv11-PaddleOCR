---
<p align="center">
  🌐 <b>Language:</b> English 🇬🇧 | 中文 🇨🇳
</p>

---

<details open>
<summary><b>English Version 🇬🇧</b></summary>
<h2>License Plate Recognition Based on YOLOv11 and PaddleOCR(基于yolov11与ocr的车牌识别)</h2>

The license plate recognition project generally consists of two parts: localization and recognition. The common combination is YOLO+CRNN. This project directly utilizes PaddleOCR, eliminating the need for separate CRNN training. With YOLOv11's convenience and efficiency, it's extremely suitable for beginners to deploy directly. Training your own model is also straightforward.


only English version in README-EN.md

未来可能会基于模型性能进行进一步优化，以及探索可以实现端到端的同时改进的方式

In the future, further optimization may be based on model performance, as well as exploring ways to achieve simultaneous improvements in an end-to-end manner.

<h2>Deployment Process</h2>
<li>Environment Setup</li>

1. After setting up a virtual environment, you can directly install from the requirement.txt file in the package.
   
   \# Run the following code
   conda create -n \<your\_env\_name> python=3.10 #\<your\_env\_name> stands for the environment name, choose as you like; due to running YOLOv11, Python version should be 3.8 or above, this project uses 3.10
   conda activate \<your\_env\_name>
   cd path#path represents the project code location path
   pip install requirement.txt

1.2 You can also directly install ultralytics pillow tqdm opencv torch

<h2><li>Running the Project</li>
<h4>
1. Data Preparation</h4>

This project uses the CCPD2021 dataset for training, which is a yellow license plate dataset. If you want to increase the recognition range and categories, please refer to the training guide.

Dataset download: [CCPD Dataset](https://github.com/detectRecog/CCPD) |[CCPD2021 Yellow Plate Dataset](https://aistudio.baidu.com/datasetdetail/101671)

Additionally, the CCPD dataset does not have corresponding label files; its annotation information is all in the filenames. For YOLO training, there needs to be label files within the dataset. The ++/tools++ folder contains ++split.py++, which can extract bounding box positions from filenames into txt label files. This project also includes data labeled with labelimg suitable for YOLO, which can be found under the /data folder.

<h4>
2. Running Results</h4>

Before running, you need to modify the image paths under yaml first.

    #yolov11/cfg/datasets/car_plate.yaml
    #dataset path below
    #Modify the image data path here
    path: F:\LPPNET-LicensePlateRecognition-YOLOv11-PaddleOCR\car_plate # data path
    train: train/images  # train images
    val: val/images  # val images
    test: test/images  # test images

Run ++run.py++, the program will process the ++car\_plate++ data, results are saved under ++result++

    #In run file, you can modify input and output paths
    input_dir = "car_plate/test/img"
    output_dir = "car_plate/results"
    
模型结果可以在run文件夹下的train13下找到，下图为<strong>混淆矩阵</strong>

<strong>confusion—matrix</strong> as follow:
![image](https://github.com/user-attachments/assets/a6376a23-7d8b-460e-8df7-8e4b747da711)

<h2><li>Project Training</li></h2>

Run ++train.py++, the program will train the YOLOv11 model.
Parameters that can be changed in the model are as follows:

    if __name__ == '__main__':
        model = YOLO('yolov11/cfg/models/11/yolov11_plate.yaml')  # Specify the YOLO model object and load the model configuration from the specified configuration file
        model.load('yolo11n.pt')  # Load the pre-trained weight file 'yolov11s.pt' to accelerate training and improve model performance

        model.train(data='yolov11/cfg/datasets/car_plate.yaml',  # Specify the path to the training dataset configuration file, this .yaml file contains the dataset path and category information
                    cache=True,  # Whether to cache the dataset to speed up subsequent training, False means not caching
                    imgsz=300,  # Specify the image size used during training, 640 indicates resizing the input image to 640x640 pixels
                    epochs=200,  # Set the total number of training rounds to 200 rounds
                    batch=4,  # Set the batch size per training round to 16, i.e., use 16 images each time the model updates
                    close_mosaic=10,  # Set how many rounds before the end of training to turn off Mosaic data augmentation, 10 means turning off Mosaic in the last 10 rounds of training
                    workers=2,  # Set the number of threads used for data loading to 8, more threads can speed up data loading
                    patience=50,  # During training, if after 50 rounds there is no improvement in performance, stop training (early stopping mechanism)
                    device='cpu',  # Specify the device to use, '0' means using the first GPU for training
                    optimizer='SGD'  # Set the optimizer to SGD (Stochastic Gradient Descent), used for updating model parameters
                    )

If you want the project to recognize other types of license plates: prepare datasets of different kinds along with their respective yaml files. These files can be found under ++yolov11/cfg/datasets++ and ++yolov11/cfg/model++

<h2><li>References and Citations</li></h2>

Project References:
Github: <https://github.com/sirius-ai/LPRNet_Pytorch>
Github：<https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition.git>
YOLO：<https://github.com/ultralytics/ultralytics.git>
</details>
<details open>
<summary><b>中文版 🇨🇳</b></summary> <h2>基于 YOLOv11 与 PaddleOCR 的车牌识别</h2>
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
</details>

