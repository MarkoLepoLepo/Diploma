import os
import sys
import argparse
import cv2
import numpy as np
import easygui
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi
from scipy.ndimage import imread

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "C:\Python352\Lib\site-packages\PyQt5\plugins\platforms"


class SaveWin(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        super(SaveWin, self).__init__()
        loadUi('SaveInfo.ui', self)
        self.setWindowTitle('Save')


class ProcessedWin(QtWidgets.QMainWindow):
    def __init__(self, path):
        QtWidgets.QMainWindow.__init__(self)
        super(ProcessedWin, self).__init__()
        loadUi('Result.ui', self)
        self.pushButton.clicked.connect(self.goon)
        self.pushButton_2.clicked.connect(self.save)
        imagelabel = self.labelBefore
        input_image = imread(path)
        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        imagelabel.setPixmap(pixmap_image)
        imagelabel.setAlignment(QtCore.Qt.AlignCenter)
        imagelabel.setScaledContents(True)
        imagelabel.setMinimumSize(1, 1)
        imagelabel.show()

        processing_image = cv2.imread(path)
        response = processing_image
        response = cv2.cvtColor(response, cv2.COLOR_BGR2HSV)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # loop over the boundaries
        # create NumPy arrays from the boundaries
        lower_blue = np.array([110, 50, 50], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(response, lower_blue, upper_blue)
        output = cv2.bitwise_and(processing_image, processing_image, mask=mask)
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY);
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        _, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # inverted threshold (light obj on dark bg)
        bin = cv2.dilate(bin, None)  # fill some holes
        bin = cv2.dilate(bin, None)
        bin = cv2.erode(bin, None)  # dilate made our shape larger, revert that
        bin = cv2.erode(bin, None)
        bin, contours1, hierarchy1 = cv2.findContours(bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hie = hierarchy1[0]
        for x in zip(contours1, hie):
            currentContour = x[0]
            currentHierarchy = x[1]
            if currentHierarchy[2] < 0:
                epsilon = 0.1 * cv2.arcLength(currentContour, True)
                approx = cv2.approxPolyDP(currentContour, epsilon, True)
                hull = cv2.convexHull(currentContour)
                print(cv2.contourArea(currentContour))
                M = cv2.moments(currentContour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(processing_image, (cx, cy), 3, (255, 0, 255), -1)
                cv2.putText(processing_image, 'C', (cx, cy), font, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        imageforshow = processing_image
        imagelabel_after = self.labelAfter
        height, width, channels = imageforshow.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(imageforshow.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        imagelabel_after.setPixmap(pixmap_image)
        imagelabel_after.setAlignment(QtCore.Qt.AlignCenter)
        imagelabel_after.setScaledContents(True)
        imagelabel_after.setMinimumSize(1, 1)
        imagelabel_after.show()


    def goon(self):
        self.close()

    def save(self):
        self.dialog = SaveWin()
        self.dialog.show()


class MyWin(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        super(MyWin, self).__init__()
        loadUi('Base.ui', self)
        self.setWindowTitle('TestDrive')
        self.pushButtonUpload.clicked.connect(self.upload_click)
        self.pushButton_2.clicked.connect(self.process_click)
        self.pushButton.clicked.connect(self.exit)


    @pyqtSlot()
    def upload_click(self):
        imagelabel = self.label_5
        filepath = easygui.fileopenbox()
        print(filepath)
        if not filepath:
            return 0
        input_image = imread(filepath)
        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        imagelabel.setPixmap(pixmap_image)
        imagelabel.setAlignment(QtCore.Qt.AlignCenter)
        imagelabel.setScaledContents(True)
        imagelabel.setMinimumSize(1, 1)
        imagelabel.show()
        self.pathforsend = filepath
    def process_click(self):
        pth = self.imagepath()
        if not pth:
            return 0
        else:
            self.dialog = ProcessedWin(pth)
            self.dialog.show()

    def imagepath(self):
        return self.pathforsend

    def exit(self):
        self.close()

class Recognition:
    def __init__(self, path):
        self.image = cv2.imread(path)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = MyWin()
    widget.show()
    sys.exit(app.exec_())
