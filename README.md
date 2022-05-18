# 基于YOLO的移动物体检测分类系统 使用🧭

## 简介

本系统是一个基于YOLOv5的物体检测分类系统，具体的检测能力根据模型而定。

本系统基于YOLOv5 6.0：<https://github.com/ultralytics/yolov5>

图形化界面基于PyQT5：<https://www.qt.io/qt-for-python>

摄像头列表读取并选择基于PyCameraList：<https://gitee.com/jiangbin2020/py-camera-list-project>

## 环境搭建

获取项目：

```bash
git clone https://gitee.com/ayo_yo/yolov5.git

cd yolov5
```

目前已支持Python 3.10，推荐使用Anaconda。

```bash
conda create -n yolov5 python=3.10

conda activate yolov5
```

由于`PyTroch`的特殊性，如果有可用的`cuda`设备建议单独安装，安装后在`requirements.txt`注释掉`PyTroch`、`torchvision`即可。访问此页面，获取对应版本的`PyTorch`

https://pytorch.org/get-started/locally/

安装依赖环境，建议使用以下安装，自动使用Conda安装支持Anaconda的包，对于不支持Anaconda的则使用PyPi安装：另外，`Tensorflow`暂不支持使用conda 安装至Python 3.10版本，使用Python3.10可以使用PyPi安装。

```bash
$ while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

## 使用说明

运行`main.py`。即自动打开窗口。

若需修改识别种类，可修改`weights`和`data`指向的模型和数据集的路径，注意在推理中`data.yaml`仅仅用到了里边的类别数量和类别名称。

```python
self.weights = ROOT / 'weights/mycoco_m.pt'  # 权重模型
self.data = 'data/mycoco.yaml'  # 数据集.yaml路径
```

整体分为三个主要的功能模块。

### 图片检测

图片检测功能分为单张图片检测和文件夹批量检测。

#### 单张图片检测

选择图片后进行检测标注并直接展示结果。

#### 文件夹检测

选择文件夹后对文件夹中的所有图片进行批量检测，检测结果将存放至`tmp/cls`文件夹下的对应类别中。一张图片中若包含多种类别，那么该图片将会被保存至多个类别文件夹。例如一张图片即包含了人和公交车，那么`tmp/cls/人`以及`tmp/cls/公交车`这两个分类文件夹中都可以找到这张图片

### 视频检测

视频检测支持本地视频检测及网络视频检测。

#### 本地视频检测

本地视频检测只需要点击选择视频按钮，打开一个本地视频即可。

视频检测将实时检测视频的每一帧，并将其进行标注后实时展示。同图片文件夹检测，可以将每一帧的结果分类保存到类别文件夹中，由于涉及大量IO操作，创建子线程保存图片以保证流畅度。

#### 网络视频检测

网络视频检测暂只支持视频源，视频网站上的视频需要先手动进行解析。

然后会自动将视频保存至`tmp/video/`中，然后进行检测。

### 摄像头检测

摄像头检测分为本地摄像头检测及网络摄像头检测。

#### 本地摄像头检测

通过PyCameraList读取本机摄像头列表，选择后可以打开对应摄像头并读取实时视频流进行检测，同时对结果进行标注并实时展示。

#### 网络摄像头检测

网络摄像头需要在URL栏输入正确的网络摄像头地址，支持RTSP协议，若有账号密码，还需要对应的账号密码。如：

```url
rtsp://user:password@domain:port
```

