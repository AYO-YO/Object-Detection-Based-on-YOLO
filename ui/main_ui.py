# -*- coding: utf-8 -*-
import os
import sys

from PyCameraList.camera_device import list_video_devices
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPropertyAnimation
from qt_material import apply_stylesheet


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.init()

    def init(self):
        self.init_logo()
        self.init_bg()
        self.init_btn()
        self.init_slots()

    def init_logo(self):
        self.setWindowIcon(QtGui.QIcon('./icon.svg'))

    def init_bg(self):
        pix = QtGui.QPixmap('bg.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def init_btn(self):
        self.btn_camera.setDisabled(False)
        self.btn_camera.setText(u'摄像头检测')
        self.btn_img.setDisabled(False)
        self.btn_img.setText(u'图片检测')
        self.btn_video.setDisabled(False)
        self.btn_video.setText(u'视频检测')

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("基于YOLO的移动物体检测分类系统")
        MainWindow.resize(1200, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btn_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy()
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
        self.verticalLayout.addWidget(self.btn_img, 0)
        self.btn_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy()
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
        self.verticalLayout.addWidget(self.btn_camera, 0)
        self.btn_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy()
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
        self.verticalLayout.addWidget(self.btn_video, 0)
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

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def setAnimation(self):
        self.animation = QPropertyAnimation(self, b'background-color')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于YOLO的移动物体检测分类系统"))
        self.btn_img.setText(_translate("MainWindow", "图片检测"))
        self.btn_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.btn_video.setText(_translate("MainWindow", "视频检测"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.btn_img.clicked.connect(self.image_guide)
        self.btn_video.clicked.connect(self.video_guide)
        self.btn_camera.clicked.connect(self.camera_guide)

    def image_guide(self):
        self.new_ui = ImageGuide()
        self.new_ui.show()
        self.new_ui.signal.connect(self.add_label)

    def video_guide(self):
        self.new_ui = VideoGuide()
        self.new_ui.show()
        self.new_ui.signal.connect(self.add_label)

    def camera_guide(self):
        self.new_ui = CameraGuide()
        self.new_ui.show()
        self.new_ui.signal.connect(self.add_label)

    def add_label(self, cls, file):
        cls_map = {0: '图片检测', 1: '视频检测', 2: '摄像头检测'}
        print(f'接收到{cls_map[cls]}任务，文件地址为：', file)
        self.new_ui.close()


class ImageGuide(QtWidgets.QWidget):
    signal = QtCore.pyqtSignal(int, str)

    def __init__(self):
        super(ImageGuide, self).__init__()
        self.resize(500, 250)
        self.setWindowTitle('图片检测')
        self.image_guide()

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
        self.signal.emit(0, file)

    def query_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件夹", '')
        self.signal.emit(0, folder)


class VideoGuide(QtWidgets.QWidget):
    signal = QtCore.pyqtSignal(int, str)

    def __init__(self):
        super(VideoGuide, self).__init__()
        self.setWindowTitle('视频检测')
        self.resize(500, 250)
        self.video_guide()

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
        self.resize(500, 200)
        self.camera_guide()

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

    def my_style(self):
        self.btn_ok.setProperty('class', 'big_button')

    def post(self):
        path = self.le_url.text()
        if path.startswith(('http://', 'https://', 'rtsp://')):
            self.signal.emit(2, path)
        else:
            camera_id = str(self.cb_camera_list.currentIndex())
            self.signal.emit(2, camera_id)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    apply_stylesheet(app, theme='light_blue.xml', invert_secondary=True)
    stylesheet = app.styleSheet()
    with open('custom.css') as file:
        app.setStyleSheet(stylesheet + file.read().format(**os.environ))
    ui.show()
    sys.exit(app.exec_())
