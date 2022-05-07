import sys

from PyQt5 import QtWidgets


class UiWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 250)
        self.setWindowTitle('图片检测')
        self.image_guide()

    def image_guide(self):
        btn_query_image = QtWidgets.QPushButton('选择图片')
        btn_query_image.clicked.connect(self.image_file)
        btn_query_folder = QtWidgets.QPushButton('选择文件夹')
        btn_query_folder.clicked.connect(self.image_folder)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(btn_query_image)
        vbox.addWidget(btn_query_folder)
        self.setLayout(vbox)

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
        return file

    def query_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件夹", '')
        return folder


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = UiWidget()
    ui.show()
    sys.exit(app.exec_())
