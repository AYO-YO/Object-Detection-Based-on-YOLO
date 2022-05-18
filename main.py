# -*- coding: utf-8 -*-

import os
import random
# Form implementation generated from reading ui file '.\project.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import sys
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image
from PyCameraList.camera_device import list_video_devices
from PyQt5 import QtCore, QtGui, QtWidgets
from qt_material import apply_stylesheet

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import LoadImages
from utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, increment_path, strip_optimizer
from utils.plots import save_one_box, Annotator, colors
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将工作目录添加至path
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.im_result = None
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_main()
        self.source = ''  # 文件/目录/URL/通配符批量选择文件, 0 -- 摄像头
        self.weights = ROOT / 'weights/mycoco_m.pt'  # 权重模型
        self.data = 'data/mycoco.yaml'  # 数据集.yaml路径
        self.img_sz = (416, 416)  # 图片大小(height, width)(必须是32的整数倍)
        self.augment = True  # 增强推理
        self.conf_thres = 0.35  # 置信阈值
        self.iou_thres = 0.45  # nms的IOU阈值
        self.max_det = 100  # 每张图像的最大检测次数
        self.device = '0' if torch.cuda.is_available() else 'cpu'  # cuda 设备, 即 0 or 0,1,2,3 or cpu
        self.view_img = False  # 展示结果
        self.save_txt = False  # 保存结果到 *.txt
        self.save_crop = False  # 保存裁剪的预测框
        self.no_save = False  # 不保存图片/视频
        self.classes = None  # 按类别过滤
        self.agnostic_nms = False  # 与类别无关的NMS
        self.visualize = False  # 可视化特征
        self.update = False  # 更新模型(边推理边强化训练)、（低性能设备不推荐）
        self.project = ROOT / 'tmp'  # 运行的目录
        self.name = 'cls'  # 保存结果到 project/name
        self.exist_ok = False  # 是否使用现有的 project/name 若为True，则使用最近的一次结果文件夹
        self.line_thickness = 3  # 边框厚度(px)
        self.hide_labels = False  # 是否隐藏标签
        self.hide_conf = False  # 隐藏置信度
        self.half = False  # 使用 FP16 半精度推理
        self.dnn = False  # 使用 OpenCV DNN 进行 ONNX 推理
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # 保存的文件夹
        self.dt, self.seen = [0.0, 0.0, 0.0], 0
        self.device = select_device(self.device)
        self.last_res = ''  # 上一次推理结果，当视频推理结果未发生变化时，则不打印结果
        cudnn.benchmark = True

        self.cap = None
        self.is_video = False
        self.need_cls = True  # 是否需要分类保存

        # 加载模型
        # TODO: 加入手动选择模型后应将此处的加载模型改为手动选择的路径
        print(self.data)
        with open(self.data, encoding='utf-8') as f:
            print(yaml.safe_load(f)['names'])
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        with open(self.data, encoding='utf-8') as f:
            self.names = (yaml.safe_load(f)['names'])
        self.img_sz = check_img_size(self.img_sz, s=self.stride)  # 检查 img_size
        # 仅在 CUDA 上支持半精度
        self.half &= (self.pt or self.jit or self.onnx or self.engine) and self.device.type != 'cpu'
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        elif self.engine and self.model.trt_fp16_input != self.half:
            LOGGER.info('模型 ' + ('需要' if self.model.trt_fp16_input else '不兼容') + ' --half. 自动调整.')
            self.half = self.model.trt_fp16_input

        # 随机取名称对应的颜色
        self.colors = [[random.randint(125, 255) for _ in range(3)] for _ in self.names]

    @torch.no_grad()
    def detect(self, im, im0s):
        s = ''
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # 扩大批量调暗
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # 前向传播，预推理，拿到推理函数
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        self.dt[2] += time_sync() - t3

        # 二级分类器（可选）
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 过程预测
        for _, det in enumerate(pred):  # 推理图片
            self.seen += 1
            self.im_result = im0s.copy()
            s += '%gx%g ' % im.shape[2:]  # 打印字符串
            imc = self.im_result.copy() if self.save_crop else self.im_result  # for save_crop
            # 标注器
            annotator = Annotator(self.im_result, line_width=self.line_thickness, example=str(self.names))
            clss = set()
            if len(det):
                # 将框从 img_size 重新缩放为 im0 大小
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], self.im_result.shape).round()

                # 打印结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类别的数量
                    s += f"{self.names[int(c)]} --- {n}, "  # 添加到结果字符串

                # 绘制结果图片
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # 类别索引
                    label = None if self.hide_labels else (
                        self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    if self.save_crop:
                        save_one_box(xyxy, imc, BGR=True)
                    if self.need_cls:
                        clss.add(c)

            # 串流结果
            self.im_result = annotator.result()
            self.clss = clss
            self.showResult()
            # 是否需要分类保存，批量图片强制保存（视频检测到包含不同类型的帧时才会保存以减少IO次数）
            if self.need_cls and (not self.is_video):
                for i in self.clss:
                    # t = threading.Thread(target=self.save_cls_result, args=(self.im_result, i))
                    # t.start()
                    self.save_cls_result(self.im_result, i)

        # 输出推理时间
        if self.is_video and s != self.last_res:
            LOGGER.info(f'{s} 推理完成，用时： ({t3 - t2:.3f}s)')
            # 是否需要分类保存
            if self.need_cls:
                t_pool: List[threading.Thread] = []
                for i in self.clss:
                    t = threading.Thread(target=self.save_cls_result, args=(self.im_result, i))
                    t.start()
                    t_pool.append(t)
                for t in t_pool:
                    t.join()
        self.last_res = s

    def after_detect(self):
        # 输出结果
        t = tuple(x / self.seen * 1E3 for x in self.dt)  # 图片的速度
        LOGGER.info(f'大小为{self.img_sz}的图像处理速度: 预处理 --- %.1fms, 推理 --- %.1fms,  NMS --- %.1fms' % t)
        if self.need_cls:
            path = '\\'.join(self.img_path.split("\\")[:-2])
            LOGGER.info(f'已保存至{path}')
        if self.update:
            strip_optimizer(self.weights)  # 更新模型（修复 SourceChangeWarning）

    @torch.no_grad()
    def pre_detect(self):
        self.source = str(self.source)

        # 加载数据
        dataset = LoadImages(self.source, img_size=self.img_sz, stride=self.stride, auto=self.pt)
        bs = 1  # batch_size

        # 开始推理
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_sz), half=self.half)  # 热身
        for path, im, im0s, vid_cap, s in dataset:
            self.detect(im, im0s)

        self.after_detect()

    def save_cls_result(self, im_row, c):
        i = 0
        dirs = os.path.join(self.save_dir, self.names[c])
        self.img_path = os.path.join(dirs, f'{i}.jpg')
        while os.path.exists(self.img_path):
            i += 1
            dirs = os.path.join(self.save_dir, self.names[c])
            self.img_path = os.path.join(dirs, f'{i}.jpg')
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        result = cv2.cvtColor(im_row, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(result)
        im.save(self.img_path)

    def showResult(self):
        result = cv2.cvtColor(self.im_result, cv2.COLOR_BGR2BGRA)
        result = cv2.resize(result, (640, 480), interpolation=cv2.INTER_AREA)

        self.QtImg = QtGui.QImage(result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("基于YOLO的移动物体检测分类系统")
        MainWindow.resize(1200, 800)
        self.setWindowTitle('基于YOLO的移动物体检测分类系统')
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btn_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_img.sizePolicy().hasHeightForWidth())
        self.btn_img.setSizePolicy(sizePolicy)
        self.btn_img.setMinimumSize(QtCore.QSize(180, 100))
        self.btn_img.setMaximumSize(QtCore.QSize(180, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.btn_img.setFont(font)
        self.btn_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(self.btn_img, 0, QtCore.Qt.AlignHCenter)
        self.btn_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_camera.sizePolicy().hasHeightForWidth())
        self.btn_camera.setSizePolicy(sizePolicy)
        self.btn_camera.setMinimumSize(QtCore.QSize(180, 100))
        self.btn_camera.setMaximumSize(QtCore.QSize(180, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.btn_camera.setFont(font)
        self.btn_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.btn_camera, 0, QtCore.Qt.AlignHCenter)
        self.btn_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_video.sizePolicy().hasHeightForWidth())
        self.btn_video.setSizePolicy(sizePolicy)
        self.btn_video.setMinimumSize(QtCore.QSize(180, 100))
        self.btn_video.setMaximumSize(QtCore.QSize(180, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.btn_video.setFont(font)
        self.btn_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.btn_video, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(1, 4)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def init_btn(self):
        self.btn_camera.setDisabled(False)
        self.btn_camera.setText(u'摄像头检测')
        self.btn_img.setDisabled(False)
        self.btn_img.setText(u'图片检测')
        self.btn_video.setDisabled(False)
        self.btn_video.setText(u'视频检测')

    def init_logo(self):
        self.setWindowIcon(QtGui.QIcon('ui/icon.svg'))

    def init_bg(self):
        pix = QtGui.QPixmap('ui/bg.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def init_main(self):
        self.init_logo()
        self.init_btn()
        self.init_bg()
        self.init_slots()

    def init_slots(self):
        self.btn_img.clicked.connect(self.image_guide)
        self.btn_video.clicked.connect(self.video_guide)
        self.btn_camera.clicked.connect(self.camera_guide)
        self.timer_video.timeout.connect(self.show_video_frame)

    def image_guide(self):
        self.new_ui = ImageGuide()
        self.new_ui.show()
        self.new_ui.signal.connect(self.add_label)

    def video_guide(self):
        if self.timer_video.isActive():
            self.shutdown_stream()
        else:
            self.new_ui = VideoGuide()
            self.new_ui.show()
            self.new_ui.signal.connect(self.add_label)

    def camera_guide(self):
        if self.timer_video.isActive():
            self.shutdown_stream()
        else:
            self.new_ui = CameraGuide()
            self.new_ui.show()
            self.new_ui.signal.connect(self.add_label)

    def add_label(self, cls, file):
        cls_map = {0: '图片检测', 1: '视频检测', 2: '摄像头检测'}
        print(f'接收到{cls_map[cls]}任务,{cls_map[cls][:-2]}地址为：', file)
        self.new_ui.close()
        if cls == 0:  # 图片检测信号
            self.is_video = False
            self.need_cls = True if os.path.isdir(file) else False  # 单张图片无需保存
            self.source = file
            self.pre_detect()
        elif cls == 1:  # 视频检测信号
            self.need_cls = True
            self.is_video = True
            self.cap = cv2.VideoCapture()
            is_url: bool = file.startswith(('http://', 'https://'))
            if is_url:
                if not os.path.exists('tmp/video'):
                    os.makedirs('tmp/video')
                torch.hub.download_url_to_file(file, 'tmp/video/net_video.mp4')
                file = 'tmp/video/net_video.mp4'
            flag = self.cap.open(file)
            if not flag:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_video.start(30)
                self.btn_camera.setDisabled(True)
                self.btn_img.setDisabled(True)
                self.btn_video.setText(u'关闭视频')
        elif cls == 2:  # 摄像头检测信号
            self.need_cls = True
            self.is_video = True
            is_url: bool = file.startswith('rtsp://')
            if is_url:
                self.need_cls = True
                self.cap = cv2.VideoCapture(file)
                self.timer_video.start(30)
                self.btn_video.setDisabled(True)
                self.btn_img.setDisabled(True)
                self.btn_camera.setText(u"关闭摄像头")
            else:
                self.cap = cv2.VideoCapture()
                if not self.timer_video.isActive():
                    pass
                    # 默认使用第一个本地camera
                    if not self.cap.open(int(file)):
                        QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                                      defaultButton=QtWidgets.QMessageBox.Ok)
                    else:
                        self.timer_video.start(30)
                        self.btn_video.setDisabled(True)
                        self.btn_img.setDisabled(True)
                        self.btn_camera.setText(u"关闭摄像头")
                else:
                    self.shutdown_stream()

    def shutdown_stream(self):
        self.timer_video.stop()
        self.cap.release()
        self.label.clear()
        self.init_main()
        self.btn_camera.clicked.connect(self.camera_guide)

    def show_video_frame(self):
        _, img = self.cap.read()
        if img is not None:
            # 数据处理
            self.im_result = img.copy()
            img = [letterbox(img, self.img_sz, stride=self.stride)[0]]
            img = np.stack(img, 0)

            # 转换
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)

            # 推理
            self.detect(img, self.im_result)
        else:
            self.shutdown_stream()


class ImageGuide(QtWidgets.QWidget):
    signal = QtCore.pyqtSignal(int, str)

    def __init__(self):
        super(ImageGuide, self).__init__()
        self.resize(500, 250)
        self.setWindowTitle('图片检测')
        self.image_guide()
        self.setWindowIcon(QtGui.QIcon('ui/icon.svg'))

    def image_guide(self):
        # pass
        self.btn_query_image = QtWidgets.QPushButton('选择图片')
        self.btn_query_image.clicked.connect(self.query_file)
        self.btn_query_folder = QtWidgets.QPushButton('选择文件夹')
        self.btn_query_folder.clicked.connect(self.query_folder)

        self.vbox = QtWidgets.QVBoxLayout(self)
        self.vbox.addWidget(self.btn_query_image)
        self.vbox.addWidget(self.btn_query_folder)
        self.setLayout(self.vbox)

    def query_file(self):
        cls = ''
        file_cls = ['jpg', 'png']
        for i in file_cls:
            cls += f'*.{i};;'
        cls += 'All Files(*)'
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开文件", "", cls)
        if not file:
            return
        self.signal.emit(0, file)

    def query_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件夹", '')
        if not folder:
            return
        self.signal.emit(0, folder)


class VideoGuide(QtWidgets.QWidget):
    signal = QtCore.pyqtSignal(int, str)

    def __init__(self):
        super(VideoGuide, self).__init__()
        self.setWindowTitle('视频检测')
        self.resize(500, 250)
        self.video_guide()
        self.setWindowIcon(QtGui.QIcon('ui/icon.svg'))

    def video_guide(self):
        self.le_url = QtWidgets.QLineEdit()
        self.le_url.setPlaceholderText('请输入URL或点击下方按钮打开本地视频')

        self.btn_video = QtWidgets.QPushButton('选择视频')
        self.btn_video.clicked.connect(self.query_video)

        self.btn_ok = QtWidgets.QPushButton('确定')
        self.btn_ok.clicked.connect(self.post)

        self.vbox = QtWidgets.QVBoxLayout()
        self.hbox = QtWidgets.QHBoxLayout()
        self.vbox.addWidget(self.le_url)
        self.hbox.addWidget(self.btn_video)
        self.hbox.addWidget(self.btn_ok)
        self.vbox.addLayout(self.hbox)

        self.setLayout(self.vbox)

    def query_video(self):
        cls = ''
        file_cls = ['mp4', 'avi', 'mov']
        for i in file_cls:
            cls += f'*.{i};;'
        cls += 'All Files(*)'
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "", cls)
        if not path:
            return
        self.le_url.setText(path)
        self.le_url.setDisabled(True)

    def post(self):
        path = self.le_url.text()
        flag = False
        if os.path.isfile(path):
            flag = True
        if path.startswith(('http://', 'https://')):
            flag = True
        if flag:
            self.signal.emit(1, path)
        else:
            msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, '警告', '请输入有效视频！')
            msg_box.exec_()


class CameraGuide(QtWidgets.QWidget):
    signal = QtCore.pyqtSignal(int, str)

    def __init__(self):
        super(CameraGuide, self).__init__()
        self.setWindowTitle('摄像头检测')
        self.camera_guide()
        self.resize(500, 250)
        self.setWindowIcon(QtGui.QIcon('ui/icon.svg'))

    def camera_guide(self):
        self.le_url = QtWidgets.QLineEdit()
        self.le_url.setPlaceholderText('请输入URL或点击下方按钮打开本地视频')

        self.lab_open_camera = QtWidgets.QLabel('打开本地摄像头：')
        self.cb_camera_list = QtWidgets.QComboBox()
        for i in list_video_devices():
            s = f'{i[0]} - {i[1]}'
            self.cb_camera_list.addItem(s)

        self.btn_ok = QtWidgets.QPushButton('确定')
        self.btn_ok.clicked.connect(self.post)

        self.tips = QtWidgets.QLabel('注：未输入合法URL即自动打开本地摄像头')

        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.le_url)

        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addWidget(self.lab_open_camera)
        self.hbox.addWidget(self.cb_camera_list)

        self.vbox.addLayout(self.hbox)

        self.vbox.addWidget(self.btn_ok)

        self.vbox.addWidget(self.tips)

        self.setLayout(self.vbox)

    def post(self):
        path = self.le_url.text()
        if path.startswith('rtsp://'):
            self.signal.emit(2, path)
        else:
            camera_id = str(self.cb_camera_list.currentIndex())
            self.signal.emit(2, camera_id)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    apply_stylesheet(app, theme='light_blue.xml', invert_secondary=True)
    stylesheet = app.styleSheet()
    with open('ui/custom.css') as file:
        app.setStyleSheet(stylesheet + file.read().format(**os.environ))
    ui.show()
    sys.exit(app.exec_())
