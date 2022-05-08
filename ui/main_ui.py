# -*- coding: utf-8 -*-
import sys

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.init_slots()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("基于YOLO的移动物体检测分类系统")
        MainWindow.resize(1200, 800)
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
        self.btn_img.setMinimumSize(QtCore.QSize(150, 100))
        self.btn_img.setMaximumSize(QtCore.QSize(150, 100))
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
        self.horizontalLayout.setStretch(1, 5)
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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于YOLO的移动物体检测分类系统"))
        self.btn_img.setText(_translate("MainWindow", "图片检测"))
        self.btn_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.btn_video.setText(_translate("MainWindow", "视频检测"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.btn_img.clicked.connect(self.image_guide)

    def image_guide(self):
        self.new_ui = ImageGuide()
        self.new_ui.show()


class ImageGuide(QtWidgets.QWidget):
    def __init__(self):
        super(ImageGuide, self).__init__()
        self.resize(400, 250)
        self.setWindowTitle('图片检测')
        self._signal = QtCore.pyqtSignal()
        self.image_guide()

    def image_guide(self):
        # pass
        self.btn_query_image = QtWidgets.QPushButton('选择图片')
        self.btn_query_image.clicked.connect(self.image_file)
        self.btn_query_folder = QtWidgets.QPushButton('选择文件夹')
        self.btn_query_folder.clicked.connect(self.image_folder)

        self.vbox = QtWidgets.QVBoxLayout(self)
        self.vbox.addWidget(self.btn_query_image)
        self.vbox.addWidget(self.btn_query_folder)
        self.setLayout(self.vbox)

    def image_file(self):
        cls = ('jpg', 'png')
        self.query_file(cls)

    def image_folder(self):
        print(self.query_folder())

    def query_file(self, file_cls: iter):
        cls = ''
        for i in file_cls:
            cls += f'*.{i};;'
        cls += 'All Files(*)'
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开文件", "", cls)

        self._signal.emit(file)

    def query_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件夹", '')
        return folder


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
