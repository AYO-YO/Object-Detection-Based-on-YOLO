# -*- coding: utf-8 -*-

import os
import random
# Form implementation generated from reading ui file '.\project.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from PyQt5 import QtCore, QtGui, QtWidgets

from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, increment_path, xyxy2xywh
from utils.plots import save_one_box, Annotator, colors
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.weights = ROOT / 'weights/yolov5s.pt'  # 权重模型
        self.source = str(ROOT / 'data/images')  # 文件/目录/URL/通配符批量选择文件, 0 -- 摄像头
        self.data = ROOT / 'data/coco128.yaml'  # 数据集.yaml路径
        self.imgsz = (640, 640)  # 图片大小(height, width)
        self.out = None
        self.augment = True  # 增强推理
        self.visualize = False  # 可视化特征
        self.conf_thres = 0.25  # 置信阈值
        self.iou_thres = 0.45  # nms的IOU阈值
        self.max_det = 1000  # 每张图像的最大检测次数
        self.device = '0'  # cuda 设备, 即 0 or 0,1,2,3 or cpu
        self.view_img = False  # 展示结果
        self.save_txt = False  # 保存结果到 *.txt
        self.save_conf = False  # 保存置信度 --save-txt labels
        self.save_crop = False  # 保存裁剪的预测框
        self.nosave = False  # 不保存图片/视频
        self.classes = None  # 按类别过滤
        self.agnostic_nms = False  # 与类别无关的NMS
        self.augment = False  # 增强推理
        self.visualize = False  # 可视化特征
        self.update = False  # 更新所有模型
        self.name = 'exp'  # 保存结果到 project/name
        self.exist_ok = False  # 是否使用现有的 project/name 若为True，则使用最近的一次结果文件夹
        self.line_thickness = 3  # 边框厚度(px)
        self.hide_labels = False  # 是否隐藏标签
        self.hide_conf = False  # 隐藏置信度
        self.half = False  # 使用 FP16 半精度推理
        self.dnn = False  # 使用 OpenCV DNN 进行 ONNX 推理
        self.project = ROOT / 'runs/detect'  # 运行的目录
        self.device = select_device(self.device)
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        cudnn.benchmark = True

        # 加载模型
        # TODO: 加入手动选择模型后应将此处的加载模型改为手动选择的路径
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        # 仅在 CUDA 上支持半精度
        self.half &= (self.pt or self.jit or self.onnx or self.engine) and self.device.type != 'cpu'
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        elif self.engine and self.model.trt_fp16_input != self.half:
            LOGGER.info('模型 ' + (
                '需要' if self.model.trt_fp16_input else '不兼容') + ' --half. 自动调整.')
            self.half = self.model.trt_fp16_input

        # 获取名称和颜色
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(125, 255)
                        for _ in range(3)] for _ in self.names]

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("识别系统")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(
            QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_img.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(
            self.pushButton_img, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(
            self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_video.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(
            self.pushButton_video, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PyQt5+YOLOv5示例"))
        self.pushButton_img.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('wechat.jpg')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def button_image_open(self):
        print('button_image_open')

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            return

        print(f'已选择图片{img_name}...')

        with torch.no_grad():
            img = LoadImages(img_name, img_size=self.imgsz, stride=self.stride, auto=self.pt)

            # 开始推理
            self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in img:
                t1 = time_sync()
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1

                # 推理
                self.visualize = increment_path(Path(path).stem, mkdir=True) if self.visualize else False
                pred = self.model(im, augment=self.augment, visualize=self.visualize)
                t3 = time_sync()
                dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                           max_det=self.max_det)
                dt[2] += time_sync() - t3
                print(pred)

                # 第二阶段分类器（可选）
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # 处理预测
                for i, det in enumerate(pred):
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(img, 'frame', 0)
                    p = Path(p)
                    save_path = str(p.name)  # im.jpg
                    txt_path = str(f'labels/{p.stem}') + ('' if img.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if self.save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if self.save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if self.hide_labels else (
                                    self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                if self.save_crop:
                                    save_one_box(xyxy, imc,
                                                 file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg',
                                                 BGR=True)
        cv2.imwrite(save_path, im0)
        self.result = cv2.cvtColor(im0, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(
            self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")

        if not video_name:
            return

        flag = self.cap.open(video_name)
        if not flag:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)

    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头检测")

    def show_video_frame(self):

        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = LoadStreams('0', img_size=self.imgsz, stride=self.stride, auto=self.pt)

                # 开始推理
                self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)
                dt, seen = [0.0, 0.0, 0.0], 0
                for path, im, im0s, vid_cap, s in img:
                    t1 = time_sync()
                    im = torch.from_numpy(im).to(self.device)
                    im = im.half() if self.half else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    t2 = time_sync()
                    dt[0] += t2 - t1

                    # 推理
                    self.visualize = False
                    pred = self.model(im, augment=self.augment, visualize=self.visualize)
                    t3 = time_sync()
                    dt[1] += t3 - t2

                    # NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                               max_det=self.max_det)
                    dt[2] += time_sync() - t3
                    print(pred)

                    # 处理预测
                    for i, det in enumerate(pred):
                        seen += 1
                        im0, frame = im0s.copy(), getattr(img, 'frame', 0)
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if self.save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if self.save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format

                                if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if self.hide_labels else (
                                        self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    if self.save_crop:
                                        save_one_box(xyxy, imc,
                                                     file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg',
                                                     BGR=True)
            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.init_logo()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
