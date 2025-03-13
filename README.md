---

<h2>License Plate Recognition Based on YOLOv11 and PaddleOCR(基于yolov11与ocr的车牌识别)</h2>

The license plate recognition project generally consists of two parts: localization and recognition. The common combination is YOLO+CRNN. This project directly utilizes PaddleOCR, eliminating the need for separate CRNN training. With YOLOv11's convenience and efficiency, it's extremely suitable for beginners to deploy directly. Training your own model is also straightforward.

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
