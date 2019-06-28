# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'carSearch.ui'
#
# Created by: Lin QiuLi 2018-7-28

from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras import backend as K
from keras_frcnn import config
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img
from skimage import data, filters, segmentation, measure, morphology, color
from ctypes import cdll,c_int,POINTER
import xlwt
import xlrd
import scipy.io as sio
import shutil

from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.Qt as qt


class Ui_carSearchSystem(qt.QObject):

    def __init__(self, parent=qt.QObject()):  # QObject类可以用来传递信号
        super(Ui_carSearchSystem,self).__init__(parent)
        self.carSearchSystem = qt.QWidget()  # 定义ImageSystem是一个Widget对象
        self.setupUi(self.carSearchSystem)  # 初始化Widget类
        self.carSearchSystem.show()  # 显示窗口
        self.selectImg.clicked.connect(self.selectImgs)  # 选择文件夹和函数连接
        self.selectVideo.clicked.connect(self.selectVideos)
        self.startButton.clicked.connect(self.startCarsearch)
        self.thread_dic = {}  # 线程字典用来存储信息

    def setupUi(self, carSearchSystem):
        carSearchSystem.setObjectName("carSearchSystem")
        carSearchSystem.resize(1140, 727)
        #carSearchSystem.setStyleSheet("background-image:url(background1.jpg)")
        self.startButton = QtWidgets.QPushButton(carSearchSystem)
        self.startButton.setGeometry(QtCore.QRect(90, 610, 100, 30))
        self.startButton.setObjectName("startButton")
        self.Area1 = QtWidgets.QLabel(carSearchSystem)
        self.Area1.setGeometry(QtCore.QRect(15, 8, 400, 20))
        self.Area1.setObjectName("Area1")
        #self.Area1.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   #"background-color: rgb(0, 255, 0);")
        self.Area2 = QtWidgets.QLabel(carSearchSystem)
        self.Area2.setGeometry(QtCore.QRect(440, 9, 54, 16))
        self.Area2.setObjectName("Area2")
        self.PlateInfoButton = QtWidgets.QPushButton(carSearchSystem)
        self.PlateInfoButton.setGeometry(QtCore.QRect(90, 510, 100, 30))
        self.PlateInfoButton.setAccessibleName("")
        self.PlateInfoButton.setObjectName("PlateInfoButton")
        self.lineEdit = QtWidgets.QLineEdit(carSearchSystem)
        self.lineEdit.setGeometry(QtCore.QRect(210, 510, 120, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.ColorInfoButton = QtWidgets.QPushButton(carSearchSystem)
        self.ColorInfoButton.setGeometry(QtCore.QRect(90, 470, 100, 30))
        self.ColorInfoButton.setAccessibleName("")
        self.ColorInfoButton.setObjectName("PlateInfoButton")
        self.ComboBox = QtWidgets.QComboBox(carSearchSystem)
        self.ComboBox.addItem("black")
        self.ComboBox.addItem("white")
        self.ComboBox.addItem("red")
        self.ComboBox.addItem("green")
        self.ComboBox.addItem("yellow")
        self.ComboBox.addItem("gray")
        self.ComboBox.addItem("cyan")
        self.ComboBox.addItem("blue")
        self.ComboBox.setGeometry(QtCore.QRect(220, 470, 100, 30))
        self.radioButton = QtWidgets.QRadioButton(carSearchSystem)
        self.radioButton.setGeometry(QtCore.QRect(100, 560, 100, 30))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(carSearchSystem)
        self.radioButton_2.setGeometry(QtCore.QRect(230, 560, 100, 30))
        self.radioButton_2.setObjectName("radioButton_2")
        self.textEdit = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit.setGeometry(QtCore.QRect(420, 210, 350, 40))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_2.setGeometry(QtCore.QRect(780, 210, 350, 40))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_3.setGeometry(QtCore.QRect(420, 440,350, 40))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_4 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_4.setGeometry(QtCore.QRect(780, 440, 350, 40))
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_5 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_5.setGeometry(QtCore.QRect(420, 670, 350, 40))
        self.textEdit_5.setObjectName("textEdit_5")
        self.textEdit_6 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_6.setGeometry(QtCore.QRect(780, 670, 350, 40))
        self.textEdit_6.setObjectName("textEdit_6")
        self.textEdit_7 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_7.setGeometry(QtCore.QRect(210, 610, 120, 30))
        self.textEdit_7.setObjectName("textEdit_7")
        self.selectImg = QtWidgets.QPushButton(carSearchSystem)
        self.selectImg.setGeometry(QtCore.QRect(90, 420, 100, 30))
        self.selectImg.setObjectName("selectImg")
        self.selectVideo = QtWidgets.QPushButton(carSearchSystem)
        self.selectVideo.setGeometry(QtCore.QRect(220, 420, 100, 30))
        self.selectVideo.setObjectName("selectImg_2")
        self.label = QtWidgets.QLabel(carSearchSystem)
        self.label.setGeometry(QtCore.QRect(180, 350, 60, 20))
        self.label.setObjectName("label")
        self.showImg = QtWidgets.QLabel(carSearchSystem)
        self.showImg.setGeometry(QtCore.QRect(15, 30, 400, 340))
        self.showImg.setMaximumSize(QtCore.QSize(400, 340))
        self.showImg.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                          "background-color: rgb(255, 255, 255);")
        self.showImg.setAlignment(QtCore.Qt.AlignCenter)
        self.showImg.setObjectName("showImg")
        self.label_2 = QtWidgets.QLabel(carSearchSystem)
        self.label_2.setGeometry(QtCore.QRect(420, 30, 350, 180))
        self.label_2.setMaximumSize(QtCore.QSize(350, 180))
        self.label_2.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                         "background-color: rgb(255, 255, 255);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(carSearchSystem)
        self.label_3.setGeometry(QtCore.QRect(780, 30, 350, 180))
        self.label_3.setMaximumSize(QtCore.QSize(350, 180))
        self.label_3.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                         "background-color: rgb(255, 255, 255);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(carSearchSystem)
        self.label_4.setGeometry(QtCore.QRect(420, 260, 350, 180))
        self.label_4.setMaximumSize(QtCore.QSize(350, 180))
        self.label_4.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                        "background-color: rgb(255, 255, 255);")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(carSearchSystem)
        self.label_5.setGeometry(QtCore.QRect(780, 260, 350, 180))
        self.label_5.setMaximumSize(QtCore.QSize(350, 180))
        self.label_5.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                          "background-color: rgb(255, 255, 255);")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(carSearchSystem)
        self.label_6.setGeometry(QtCore.QRect(420, 490, 350, 180))
        self.label_6.setMaximumSize(QtCore.QSize(350, 180))
        self.label_6.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                           "background-color: rgb(255, 255, 255);")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(carSearchSystem)
        self.label_7.setGeometry(QtCore.QRect(780, 490, 350, 180))
        self.label_7.setMaximumSize(QtCore.QSize(350, 180))
        self.label_7.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                           "background-color: rgb(255, 255, 255);")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(carSearchSystem)
        self.label_8.setGeometry(QtCore.QRect(180,380, 54, 16))
        self.label_8.setObjectName("carAreaName")

        self.retranslateUi(carSearchSystem)
        #self.exitButton.clicked.connect(carSearchSystem.close)
        QtCore.QMetaObject.connectSlotsByName(carSearchSystem)

    def retranslateUi(self, carSearchSystem):
        _translate = QtCore.QCoreApplication.translate
        carSearchSystem.setWindowTitle(_translate("carSearchSystem", "车辆检索系统"))
        self.startButton.setText(_translate("carSearchSystem", "开始"))
        #self.exitButton.setText(_translate("carSearchSystem", "退出"))
        self.Area1.setText(_translate("carSearchSystem", "采集中心"))
        self.Area2.setText(_translate("carSearchSystem", "检索结果"))
        self.label_8.setText(_translate("carSearchSystem", "目标车辆"))
        self.PlateInfoButton.setText(_translate("carSearchSystem", "请输入车辆号牌"))
        self.ColorInfoButton.setText(_translate("carSearchSystem", "请选择车辆颜色"))
        #self.selectColorButton.setText(_translate("carSearchSystem", "black"))
        self.radioButton.setText(_translate("carSearchSystem", "按车牌检索"))
        self.radioButton_2.setText(_translate("carSearchSystem", "按特征检索"))
        self.selectImg.setText(_translate("carSearchSystem", "选择目标车辆"))
        self.selectVideo.setText(_translate("carSearchSystem", "选择目标视频"))
        self.label.setText(_translate("carSearchSystem", "目标车辆"))
        self.showImg.setText(_translate("carSearchSystem", "显示目标车辆"))
        self.label_2.setText(_translate("carSearchSystem", "rank1"))
        self.label_3.setText(_translate("carSearchSystem", "rank2"))
        self.label_4.setText(_translate("carSearchSystem", "rank3"))
        self.label_5.setText(_translate("carSearchSystem", "rank4"))
        self.label_6.setText(_translate("carSearchSystem", "rank5"))
        self.label_7.setText(_translate("carSearchSystem", "rank6"))

    def selectImgs(self):
        print("load--Img")
        self.ImgName, _ = qt.QFileDialog.getOpenFileName(None, '选择图片', os.getcwd() , 'Image files(*.jpg *.gif *.png)')
        print(self.ImgName)
        pixmap = QtGui.QPixmap(self.ImgName)
        self.showImg.setPixmap(pixmap)  # 在label上显示图片
        self.showImg.setScaledContents(True)
        return self.ImgName

    def selectVideos(self):
        print("load--Video")
        self.VideoName, _ = qt.QFileDialog.getOpenFileName(None, '选择视频',os.getcwd() , 'Image files(*.MP4 *.avi)')
        print(self.VideoName)
        return self.VideoName

    def resultShow(self):
        ImgPath = os.getcwd() + '/rankImage/'
        txtResult = open('rankResult.txt', 'r')
        imgs = os.listdir(ImgPath)
        Num = len(imgs)
        if Num==1:
            ImgName1 = ImgPath+'1.jpg'
            pixmap1 = QtGui.QPixmap(ImgName1)
            self.label_2.setPixmap(pixmap1)  # 在label上显示图片
            self.label_2.setScaledContents(True)
            result1 = txtResult.readline()
            self.textEdit.setText(result1)

        if Num == 2:
            ImgName1 = ImgPath + '1.jpg'
            pixmap1 = QtGui.QPixmap(ImgName1)
            self.label_2.setPixmap(pixmap1)  # 在label上显示图片
            self.label_2.setScaledContents(True)
            result1 = txtResult.readline()
            self.textEdit.setText(result1)

            ImgName2 = ImgPath + '2.jpg'
            pixmap2 = QtGui.QPixmap(ImgName2)
            self.label_3.setPixmap(pixmap2)  # 在label上显示图片
            self.label_3.setScaledContents(True)
            result2 = txtResult.readline()
            self.textEdit_2.setText(result2)

        if Num == 3:
            ImgName1 = ImgPath + '1.jpg'
            pixmap1 = QtGui.QPixmap(ImgName1)
            self.label_2.setPixmap(pixmap1)  # 在label上显示图片
            self.label_2.setScaledContents(True)
            result1 = txtResult.readline()
            self.textEdit.setText(result1)

            ImgName2 = ImgPath + '2.jpg'
            pixmap2 = QtGui.QPixmap(ImgName2)
            self.label_3.setPixmap(pixmap2)  # 在label上显示图片
            self.label_3.setScaledContents(True)
            result2 = txtResult.readline()
            self.textEdit_2.setText(result2)

            ImgName3 = ImgPath + '3.jpg'
            pixmap3 = QtGui.QPixmap(ImgName3)
            self.label_4.setPixmap(pixmap3)  # 在label上显示图片
            self.label_4.setScaledContents(True)
            result3 = txtResult.readline()
            self.textEdit_3.setText(result3)

        if Num == 4:
            ImgName1 = ImgPath + '1.jpg'
            pixmap1 = QtGui.QPixmap(ImgName1)
            self.label_2.setPixmap(pixmap1)  # 在label上显示图片
            self.label_2.setScaledContents(True)
            result1 = txtResult.readline()
            self.textEdit.setText(result1)

            ImgName2 = ImgPath + '2.jpg'
            pixmap2 = QtGui.QPixmap(ImgName2)
            self.label_3.setPixmap(pixmap2)  # 在label上显示图片
            self.label_3.setScaledContents(True)
            result2 = txtResult.readline()
            self.textEdit_2.setText(result2)

            ImgName3 = ImgPath + '3.jpg'
            pixmap3 = QtGui.QPixmap(ImgName3)
            self.label_4.setPixmap(pixmap3)  # 在label上显示图片
            self.label_4.setScaledContents(True)
            result3 = txtResult.readline()
            self.textEdit_3.setText(result3)

            ImgName4 = ImgPath + '4.jpg'
            pixmap4 = QtGui.QPixmap(ImgName4)
            self.label_5.setPixmap(pixmap4)  # 在label上显示图片
            self.label_5.setScaledContents(True)
            result4 = txtResult.readline()
            self.textEdit_4.setText(result4)

        if Num == 5:
            ImgName1 = ImgPath + '1.jpg'
            pixmap1 = QtGui.QPixmap(ImgName1)
            self.label_2.setPixmap(pixmap1)  # 在label上显示图片
            self.label_2.setScaledContents(True)
            result1 = txtResult.readline()
            self.textEdit.setText(result1)

            ImgName2 = ImgPath + '2.jpg'
            pixmap2 = QtGui.QPixmap(ImgName2)
            self.label_3.setPixmap(pixmap2)  # 在label上显示图片
            self.label_3.setScaledContents(True)
            result2 = txtResult.readline()
            self.textEdit_2.setText(result2)

            ImgName3 = ImgPath + '3.jpg'
            pixmap3 = QtGui.QPixmap(ImgName3)
            self.label_4.setPixmap(pixmap3)  # 在label上显示图片
            self.label_4.setScaledContents(True)
            result3 = txtResult.readline()
            self.textEdit_3.setText(result3)

            ImgName4 = ImgPath + '4.jpg'
            pixmap4 = QtGui.QPixmap(ImgName4)
            self.label_5.setPixmap(pixmap4)  # 在label上显示图片
            self.label_5.setScaledContents(True)
            result4 = txtResult.readline()
            self.textEdit_4.setText(result4)

            ImgName5 = ImgPath + '5.jpg'
            pixmap5 = QtGui.QPixmap(ImgName5)
            self.label_6.setPixmap(pixmap5)  # 在label上显示图片
            self.label_6.setScaledContents(True)
            result5 = txtResult.readline()
            self.textEdit_5.setText(result5)

        if Num>=6:
            ImgName1 = ImgPath+'1.jpg'
            pixmap1 = QtGui.QPixmap(ImgName1)
            self.label_2.setPixmap(pixmap1)  # 在label上显示图片
            self.label_2.setScaledContents(True)
            result1 = txtResult.readline()
            self.textEdit.setText(result1)

            ImgName2 = ImgPath + '2.jpg'
            pixmap2 = QtGui.QPixmap(ImgName2)
            self.label_3.setPixmap(pixmap2)  # 在label上显示图片
            self.label_3.setScaledContents(True)
            result2 = txtResult.readline()
            self.textEdit_2.setText(result2)

            ImgName3 = ImgPath + '3.jpg'
            pixmap3 = QtGui.QPixmap(ImgName3)
            self.label_4.setPixmap(pixmap3)  # 在label上显示图片
            self.label_4.setScaledContents(True)
            result3 = txtResult.readline()
            self.textEdit_3.setText(result3)

            ImgName4 = ImgPath + '4.jpg'
            pixmap4 = QtGui.QPixmap(ImgName4)
            self.label_5.setPixmap(pixmap4)  # 在label上显示图片
            self.label_5.setScaledContents(True)
            result4 = txtResult.readline()
            self.textEdit_4.setText(result4)

            ImgName5 = ImgPath + '5.jpg'
            pixmap5 = QtGui.QPixmap(ImgName5)
            self.label_6.setPixmap(pixmap5)  # 在label上显示图片
            self.label_6.setScaledContents(True)
            result5 = txtResult.readline()
            self.textEdit_5.setText(result5)

            ImgName6 = ImgPath + '6.jpg'
            pixmap6 = QtGui.QPixmap(ImgName6)
            self.label_7.setPixmap(pixmap6)  # 在label上显示图片
            self.label_7.setScaledContents(True)
            result6 = txtResult.readline()
            self.textEdit_6.setText(result6)

    def startCarsearch(self):  # 开始处理函数
        VideoPath = self.VideoName
        ObjectImg = self.ImgName

        self.controller = Controller(VideoPath,ObjectImg)  # Controller的实例化对象
        self.textEdit_7.setText("任务执行")  # 正在进行的状态
        thread = qt.QThread()  # 开启一个线程
        self.VideoPath = thread# 线程thread1 = qt.QThread()
        #self.sg_continue.connect(self.controller.work_continue)  # 信号链接的是线程的阻塞函数
        #self.sg_pause.connect(self.controller.work_pause)  #
        self.controller.sg_finished.connect(self.finished)
        self.controller.sg_setTaskStatus.connect(self.setTaskStatus)
        self.controller.moveToThread(thread)
        thread.started.connect(self.controller.startProcess)
        thread.start()

    # def __del__(self):
    #         self.thread.quit()
    #         self.thread.wait()

    def finished(self,VideoName):
        print('任务结束')
        del self.VideoName
        self.textEdit_7.setText("任务结束")
        self.resultShow()

    def setTaskStatus(self,TaskStatus):
        self.textEdit_7.setText(TaskStatus)

class ThreadProcess(qt.QObject):  # 这个将来需要改成线程池
    def __init__(self, VideoName, ObjectCar):
        super(ThreadProcess, self).__init__()
        self.maxThreads = qt.QThreadPool.globalInstance().maxThreadCount()
        self.mutex_process = qt.QMutex(qt.QMutex.Recursive)  # 互斥锁：用于阻塞线程
        self.currentThreads = 0
        self.VideoName = VideoName
        self.ObjectCar = ObjectCar

class Controller(qt.QObject):
    sg_finished = qt.pyqtSignal(str)
    sg_setTaskStatus = qt.pyqtSignal(str)
    def __init__(self, OpenFiles,SelectedImg):
        super(Controller, self).__init__()
        self.VideoName = OpenFiles
        self.ObjectCar = SelectedImg
        self.mutex_process = qt.QMutex(qt.QMutex.Recursive)  # 互斥锁：用于阻塞线程
        self.threadpause = False
        self.threadcontinue = False

    def work_continue(self, VideoName):
        if VideoName != self.Videofile or self.threadcontinue:
            return
        self.mutex_process.unlock()  # 解锁
        self.threadcontinue = True
        self.threadpause = False

    def work_pause(self, VideoName):
        if VideoName != self.Videofile or self.threadpause:
            return
        self.mutex_process.lock()
        self.threadpause = True
        self.threadcontinue = False

    def format_img_size(self,img, C):
        """ formats the image size based on config """
        img_min_side = float(C.im_size)
        (height, width, _) = img.shape

        if width <= height:
            ratio = img_min_side / width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side / height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img

    def format_img_channels(self, img, C):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
        img /= C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def demo_frame(self,img, X, img1, C, model_rpn, model_classifier_only, bbox_threshold, class_mapping):
        # st = time.time()
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []
        id = 0
        car_position = []  # 需要定位车辆的坐标信息

        for key in bboxes:  # 车辆的候选区域
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            # fp = open(save_txt_file, 'w+')
            for jk in range(new_boxes.shape[0]):
                id += 1
                (x1, y1, x2, y2) = new_boxes[jk, :]
                # img_car = img1[y1: y2, x1: x2]
                # cv2.imshow('img', img_car)
                # car_name = img_name[:len(img_name) - 4] + '_' + str(id) + '.jpg' #底下main里有imagename声明
                # cv2.imwrite(self.car_save_path + car_name, img_car)   #定位的车辆
                all_dets.append((key, 100 * new_probs[jk]))

                # fp.write( str(x1) + ',' )  #在csv中村车辆的坐标信息，置信度
                # fp.write( str(y1) + ',' )
                # fp.write( str(x2) + ',' )
                # fp.write( str(y2) + ',' )
                # fp.write( str(new_probs[jk]) + '\n' )
                car_position.append([x1, y1, x2, y2])
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # fp.close()


                # print('Detect time = {}'.format(time.time() - st))
        print(all_dets)  # 检测到的目标种类和概率
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return img, car_position  # 加了车辆坐标信息

    def to_bytes(self,bytes_or_str):
        if isinstance(bytes_or_str, str):
            return bytes_or_str.encode('utf-8')
        return bytes_or_str

    def avg(self,array):
        sum = 0
        n = len(array)
        for num in array:
            sum = sum + num
        avge = 1.0 * sum / (n + 0.1)  # 改过
        return avge

    def seg_4_plate(self,img):
        img = cv2.resize(img, (136, 36), interpolation=cv2.INTER_CUBIC)
        imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imgg = cv2.resize(imgg, (409, 90), interpolation=cv2.INTER_CUBIC)
        imgg = cv2.resize(imgg, (136, 36), interpolation=cv2.INTER_CUBIC)
        th, img2 = cv2.threshold(imgg, 0, 255, cv2.THRESH_OTSU)  # 大津算法二值化图像
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        img5 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
        img6 = morphology.remove_small_objects(img5, min_size=50, connectivity=2,
                                               in_place=True)  # connectivity表示连接的模式，1代表4邻接，2代表8邻接
        labels = measure.label(img6, connectivity=2)  # 8连通区域标记
        # cv2.imshow('10', img5)
        # cv2.waitKey()
        # cv2.imshow('11', img6)
        # cv2.waitKey()

        m = labels.max()
        # print(m)
        a = measure.regionprops(labels)
        aa = [[0 for i in range(1, 5)] for j in range(1, m + 1)]  # aa是m行4列的矩阵，记录m个连通区域信息，从（0,0）开始

        ###显示二值化图像
        # fig, (ax1)= plt.subplots(1, 1)
        # ax1.imshow(img6)

        for j in range(0, m):
            if a[j].area < 50:
                continue
            minr, minc, maxr, maxc = a[j].bbox  # minc=x0，minr=y0,maxc=x1,maxr=y1,(x0,y0)是左上角坐标
            aa[j][0] = minc  # x
            aa[j][1] = minr  # y
            aa[j][2] = maxc - minc  # w
            aa[j][3] = maxr - minr  # h

        ###在图像中连通区域加上框
        #     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        #     ax1.add_patch(rect)
        # fig.tight_layout()
        # plt.show()

        aa = np.array(aa)

        #####去除两边边框，当w过宽且y太大太小时，去除
        id = np.where(aa[:, 2] >= 10 * aa[:, 3])
        for i in id:
            aa[i, :] = [0]

        id = np.where(aa[:, 3] >= 10 * aa[:, 2])
        for i in id:
            aa[i, :] = [0]

        id = np.where(aa[:, 1] >= 85 * 136 / 409)
        for i in id:
            aa[i, :] = [0]

        # id = np.where(aa[:, 1] <= 3)
        # for i in id:
        #     aa[i, :] = [0]

        #####按照x坐标排序，按照字符顺序从左到右
        aa = aa[np.lexsort(aa[:, ::-1].T)]

        #####去除aa中为0 的行
        idx = np.where(aa > 0)
        idx = np.array(idx)
        # print (idx)
        aa = aa[idx[0, 0]:m, :]
        # print(aa)
        #####找出框宽过大或过小的数目
        km = np.where(aa[:, 2] > 136 / 14) and np.where(aa[:, 2] < 136 / 4)
        km = np.array(km)
        plateflag = 1
        # print(km[0])
        if len(aa) < 2 or km.size == 0:
            plateflag = 0  ###是否为残缺车牌的标记
            print("bad plate")

        elif km.size >= 3:
            bb = aa[km, 2]
            bb = np.array(bb)
            # bb = avg(bb)
            bb = bb[0]
            # print(bb)
            bb = sorted(bb)
            # print(bb)
            if bb[0] >= 35 * 136 / 409:
                bb = bb
            else:
                bb = bb[1:len(bb)]
            # print(bb)
            if bb[len(bb) - 1] <= 55 * 136 / 409:
                bb = bb
            else:
                bb = bb[0:len(bb) - 1]  # 改过
                # bb = bb[0:len(bb) - 1]
            # print(bb)
            bb = np.array(bb)
            # w_mean = avg(bb[1:len(bb) - 1])
            w_mean = self.avg(bb[1:(len(bb))])  # 修改的
            # print(w_mean)
            buf = np.where(abs(bb[:] - w_mean) <= 10 * 136 / 409)
            buf = np.array(buf)
            # print(buf)
            # print(buf.size)
            if buf.size != 1:
                ind = np.where(abs(bb[:] - w_mean) > 10 * 136 / 409)
                for i in ind:
                    bb[i] = [0]
                bb = list(filter(lambda x: x != 0, bb))
                g = 0
                ww = 0
                #####……#####
                for i in range(len(bb)):
                    g = g + abs(bb[i] - w_mean)
                ww = self.avg(bb)
            else:
                ww = 45 * 136 / 409
                # print(ww)
        elif km.size == 2:
            ww = (45 * 136 / 409 + aa[km[0][0], 2] + aa[km[0][1], 2]) / 3

        elif km.size == 1:
            ww = (45 * 136 / 409 + aa[km[0][0], 2]) / 2
        ### 上面得到的ww就是车牌中字符的宽度
        if len(aa) < 2 or km.size == 0:
            print("bad plate")
            plateflag = 0  # 标记，若plateflag为0，说明车牌不完整，后续识别可以以此区别，若不完整，则不识别
        else:
            aa_min = abs(aa[0, 2] - ww)
            f = 0
            # print(aa[6,2])
            for i in range(0, len(aa)):
                if aa_min > abs(aa[i, 2] - ww):
                    aa_min = abs(aa[i, 2] - ww)
                    f = i  ### 第几个字符最接近平均宽度，以此为最佳字符
            # print(f)
            ac_center = aa[f, 0] + aa[f, 2] / 2
            # print(ac_center)
            space = (12 + 45) / 45 * ww
            half_w = (12 + 45) / 45 * ww / 2
            num = int(136 / (ww + 1))  # 改过
            # print(ac_center)
            # print(num)
            if num >= 7:
                if (0 < ac_center) and (ac_center <= 51 * 136 / 409):
                    ac_num = 1
                    for s in range(1, 3):
                        k = s - ac_num
                        left = int(ac_center + k * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                    for s in range(3, 8):
                        k = s - ac_num
                        left = int(ac_center + k * space + 22 / 57 * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space + 22 / 57 * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                if (51 * 136 / 409 < ac_center) and (ac_center <= 119 * 136 / 409):
                    ac_num = 2
                    for s in range(1, 3):
                        k = s - ac_num
                        left = int(ac_center + k * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        # char_1 = img[:, left:right]
                        globals()['char_' + str(s)] = img[:, left:right]
                    # char_1 = cv2.resize(char_1, (70, 110), interpolation=cv2.INTER_CUBIC)
                    # cv2.imshow('1', char_1)
                    # cv2.waitKey()

                    for s in range(3, 8):
                        k = s - ac_num
                        left = int(ac_center + k * space + 22 / 57 * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space + 22 / 57 * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                if (119 * 136 / 409 < ac_center) and (ac_center <= 187 * 136 / 409):
                    ac_num = 3
                    for s in range(1, 3):
                        k = s - ac_num
                        left = int(ac_center + k * space - 22 / 57 * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space - 22 / 57 * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                    for s in range(3, 8):
                        k = s - ac_num
                        left = int(ac_center + k * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                if (187 * 136 / 409 < ac_center) and (ac_center <= 244 * 136 / 409):
                    ac_num = 4
                    for s in range(1, 3):
                        k = s - ac_num
                        left = int(ac_center + k * space - 22 / 57 * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space - 22 / 57 * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                    for s in range(3, 8):
                        k = s - ac_num
                        left = int(ac_center + k * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = right - ww
                        right = int(ac_center + k * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                            left = int(left)  # 改过
                        globals()['char_' + str(s)] = img[:, left:right]

                if (244 * 136 / 409 < ac_center) and (ac_center <= 301 * 136 / 409):
                    ac_num = 5
                    for s in range(1, 3):
                        k = s - ac_num
                        left = int(ac_center + k * space - 22 / 57 * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space - 22 / 57 * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                    for s in range(3, 8):
                        k = s - ac_num
                        left = int(ac_center + k * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                if (301 * 136 / 409 < ac_center) and (ac_center <= 358 * 136 / 409):
                    ac_num = 6
                    for s in range(1, 3):
                        k = s - ac_num
                        left = int(ac_center + k * space - 22 / 57 * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space - 22 / 57 * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                    for s in range(3, 8):
                        k = s - ac_num
                        left = int(ac_center + k * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                if (358 * 136 / 409 < ac_center) and (ac_center <= 409 * 136 / 409):
                    ac_num = 7
                    for s in range(1, 3):
                        k = s - ac_num
                        left = int(ac_center + k * space - 22 / 57 * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                            left = int(right - ww)
                        right = int(ac_center + k * space - 22 / 57 * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]

                    for s in range(3, 8):
                        k = s - ac_num
                        left = int(ac_center + k * space - half_w)
                        if left <= 0:
                            left = 0
                        if left >= 136:
                            left = int(right - ww)
                        right = int(ac_center + k * space + half_w)
                        if right <= 0:
                            right = int(left + ww)
                        if right >= 136:
                            right = 136
                        globals()['char_' + str(s)] = img[:, left:right]
            else:
                print("bad plate")
                plateflag = 0
        # print(f)
        # print(char_1.shape)

        if plateflag != 0:
            return char_1, char_2, char_3, char_4, char_5, char_6, char_7
        else:
            return 0

    def char_recog(self, char2, char3, char4, char5, char6, char7):
        # 读取model
        # K.clear_session()
        model = model_from_json(open('carplatenew.json').read())
        model.load_weights('carplatenew.h5')

        img_char2 = cv2.resize(char2, (70, 110), interpolation=cv2.INTER_CUBIC)
        img_char3 = cv2.resize(char3, (70, 110), interpolation=cv2.INTER_CUBIC)
        img_char4 = cv2.resize(char4, (70, 110), interpolation=cv2.INTER_CUBIC)
        img_char5 = cv2.resize(char5, (70, 110), interpolation=cv2.INTER_CUBIC)
        img_char6 = cv2.resize(char6, (70, 110), interpolation=cv2.INTER_CUBIC)
        img_char7 = cv2.resize(char7, (70, 110), interpolation=cv2.INTER_CUBIC)

        x2 = img_to_array(img_char2)
        x3 = img_to_array(img_char3)
        x4 = img_to_array(img_char4)
        x5 = img_to_array(img_char5)
        x6 = img_to_array(img_char6)
        x7 = img_to_array(img_char7)

        x2 = np.expand_dims(x2, axis=0)
        x3 = np.expand_dims(x3, axis=0)
        x4 = np.expand_dims(x4, axis=0)
        x5 = np.expand_dims(x5, axis=0)
        x6 = np.expand_dims(x6, axis=0)
        x7 = np.expand_dims(x7, axis=0)

        preds2 = model.predict_classes(x2)
        prob2 = model.predict_proba(x2)
        preds3 = model.predict_classes(x3)
        prob3 = model.predict_proba(x3)
        preds4 = model.predict_classes(x4)
        prob4 = model.predict_proba(x4)
        preds5 = model.predict_classes(x5)
        prob5 = model.predict_proba(x5)
        preds6 = model.predict_classes(x6)
        prob6 = model.predict_proba(x6)
        preds7 = model.predict_classes(x7)
        prob7 = model.predict_proba(x7)

        table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
                 'Q',
                 'R',
                 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        num2 = int(preds2)
        num3 = int(preds3)
        num4 = int(preds4)
        num5 = int(preds5)
        num6 = int(preds6)
        num7 = int(preds7)

        # print("图片",char_7,"以概率",prob[0][num],"识别为:",table[num])
        # print("图片以概率", prob2[0][num2], "识别为:", table[num2])
        char_result = []
        char_result.append(str(table[num2]))
        char_result.append(str(table[num3]))
        char_result.append(str(table[num4]))
        char_result.append(str(table[num5]))
        char_result.append(str(table[num6]))
        char_result.append(str(table[num7]))
        prob = str([prob2[0][num2], prob3[0][num3], prob4[0][num4], prob5[0][num5], prob6[0][num6], prob7[0][num7]])
        return char_result, prob

    def startProcess(self):
        print('开始处理')

        self.sg_setTaskStatus.emit("正在进行车辆检测")
        Videoname = self.VideoName

        parser = OptionParser()
        # parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
        parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                          help="Number of ROIs per iteration. Higher means more memory use.", default=32)
        parser.add_option("-c", "--config_filename", dest="config_filename", help=
        "Location to read the metadata related to the training (generated when training).",
                          default="config.pickle")
        parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                          default='resnet50')

        (options, args) = parser.parse_args()

        # if not options.test_path:   # if filename is not given
        #	parser.error('Error: path to test data must be specified. Pass --path to command line')

        config_output_filename = options.config_filename

        with open(config_output_filename, 'rb') as f_in:
            C = pickle.load(f_in)

        if C.network == 'resnet50':
            import keras_frcnn.resnet as nn
        elif C.network == 'vgg':
            import keras_frcnn.vgg as nn

        # turn off any data augmentation at test time
        C.use_horizontal_flips = False
        C.use_vertical_flips = False
        C.rot_90 = False

        # 根据config文件进行配置网络

        class_mapping = C.class_mapping

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        class_mapping = {v: k for k, v in class_mapping.items()}
        print(class_mapping)
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        C.num_rois = int(options.num_rois)

        if C.network == 'resnet50':
            num_features = 1024
        elif C.network == 'vgg':
            num_features = 512

        # 根据Keras配置判断backend,本系统选用TF作为backend

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # 定义CNN网络 根据--network进行配置可以为VGG或resnet50
        shared_layers = nn.nn_base(img_input, trainable=True)

        # 定义RPN 根据config文件进行配置
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping),
                                   trainable=True)

        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)

        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        all_imgs = []

        classes = {}

        # 准确度限制
        bbox_threshold = 0.95

        visualise = True

        ddd = model_rpn.predict(np.ones((1, 800, 600, 3)))
        aaa = np.ones((1, 50, 38, 1024))
        bbb = np.ones((1, 32, 4))
        ccc = model_classifier_only.predict([aaa, bbb])

        excelResults = []
        excelName = []
        excelLocate = []
        excel = xlwt.Workbook()
        sheet = excel.add_sheet('Results', cell_overwrite_ok=True)
        sheet.write(0, 0, "帧信息")
        sheet.write(0, 1, "坐标信息")
        sheet.write(0, 2, "识别结果")

        VC = cv2.VideoCapture(Videoname)
        total_frames = int(VC.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
        FPS = 100
        count = 1
        if VC.isOpened():
            flag, img_frame = VC.read()
            print('读到视频了', '一共', str(total_frames), '帧')
        else:
            flag = False

        rootdir = os.path.abspath('.') + '\\image_output\\CarImage\\'
        # 清空文件夹
        shutil.rmtree(rootdir)
        os.mkdir(rootdir)
        down = os.path.abspath('.') + '\\image_output\\PlateImage\\'
        # 清空文件夹
        shutil.rmtree(down)
        os.mkdir(down)
        while flag:  # for frame_num in range(int(total_frames)):
            flag, img_frame = VC.read()
            if count % FPS == 0:
                print('现在是第', str(count), '帧')
                frame_resize = self.format_img_size(img_frame, C)
                crop_im = self.format_img_size(img_frame, C)
                # 调用format_img_channels函数对帧图像进行预处理
                frame_im = self.format_img_channels(frame_resize, C)
                # 调用demo_frame 函数对帧图像进行车辆检测，输出车辆置信度及坐标，并在帧图像上对车辆进行标框
                detect_img, car_position = self.demo_frame(frame_resize, frame_im, crop_im, C, model_rpn,
                                                      model_classifier_only, bbox_threshold, class_mapping)

                for car_num in range(np.shape(car_position)[0]):  # 存储车辆图像

                    if car_position[car_num][0] < 0 or car_position[car_num][1] < 0 or car_position[car_num][2] < 0 or \
                                    car_position[car_num][3] < 0:
                        continue
                    img_car = detect_img[car_position[car_num][1]:car_position[car_num][3],
                              car_position[car_num][0]: car_position[car_num][2], :]  # y1:y2,x1:x2

                    car_location = np.array(car_position[car_num]).tolist()
                    car_location = ''.join(str(car_location))
                    print(car_location)
                    excelLocate.append(car_location)
                    carFrame = str(count)
                    excelName.append(carFrame)
                    carFrame = carFrame.zfill(5)
                    car_name = carFrame + '_' + str(car_num) + '.jpg'

                    carpath = os.path.join(rootdir, car_name)
                    cv2.imwrite(carpath, img_car)  # 定位的车辆

                    carpath = self.to_bytes(carpath)
                    down = self.to_bytes(down)
                    dll = cdll.LoadLibrary("PlateDetectDll.dll")
                    dll.fnPlateDetectDll.restype = POINTER(c_int)  # 重载API的返回值类型  C++中返回的是int*
                    locate_out = dll.fnPlateDetectDll(carpath, down)  # 输出了车牌在车辆图像中车牌的坐标信息

                    if bool(locate_out) == True:
                        for j in range(0, 4):  # 算车牌在原图的坐标
                            locate_out[2 * j] = locate_out[2 * j] + car_position[car_num][0]  # car_pos这个场景中第k辆车的坐标信息
                            locate_out[2 * j + 1] = locate_out[2 * j + 1] + car_position[car_num][1]

                        platepath = (os.getcwd() + '\\image_output\\PlateImage\\')  # 车牌图像路径
                        chin_path = (os.getcwd() + '\\image_output\\Chin/')  # 中文字符路径

                        path = os.path.join(platepath, car_name)
                        plate = cv2.imread(path)  # 车牌图像
                        if self.seg_4_plate(plate) == 0:
                            print(path, "Can Not Segment!")
                            final_result = '无法分割'
                            final_result = ''.join(final_result)
                            excelResults.append(final_result)
                            # excelName.append(car_name)
                        else:

                            print(path, "Can Segment")
                            p1, p2, p3, p4, p5, p6, p7 = self.seg_4_plate(plate)  # 分割字符

                            # -------------------------字符识别模块---------------------#
                            cv2.imwrite(chin_path + 'chin.jpg', p1)  # 存储汉字字符
                            chinexe = "chin_rec_build.exe"
                            os.system(chinexe + " " + '"' + chin_path + '"')  # 路径中有空格 需要加""
                            chin_data = sio.loadmat('result.mat')
                            chin_rec = chin_data['result']  # 汉字识别结果写在list里面
                            char_result, recog_prob = self.char_recog(p2, p3, p4, p5, p6, p7)  # recog_result要显示在list里面
                            chin_rec = np.array(chin_rec).tolist()  # 数组变为列表
                            final_result = np.append(chin_rec, char_result)
                            final_result = np.array(final_result).tolist()
                            final_result = ''.join(final_result)
                            excelResults.append(final_result)
                            # excelName.append(car_name)

                    else:

                        final_result = '无法定位'
                        excelResults.append(final_result)
                        # excelName.append(car_name)

            count = count + 1
        # VC.release()
        K.clear_session()
        for names in range(len(excelName)):
            sheet.write(names + 1, 0, excelName[names])
            print(excelName[names])
        for rec in range(len(excelResults)):
            sheet.write(rec + 1, 2, excelResults[rec])
            print(excelResults[rec])
        for loc in range(len(excelLocate)):
            sheet.write(loc + 1, 1, excelLocate[loc])
            print(excelLocate[loc])
            # realtime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # 标准化时间

        xlsName = 'result.xls'
        # excel.save(realtime + '.xls')
        #excel.save(xlsName)#记得去掉注释符

        # 车辆检索
        self.sg_setTaskStatus.emit("正在进行车辆检索")
        srcImg_path = os.getcwd() + '/image_output/CarImage/'#'./image_output/CarImage/'#这个可能需要改一下
        templateImg = self.ObjectCar
        #templateImg = 'E:/Pycharm Projects/MatchVersion/template/00800_0.jpg'
        #templateImg_path = 'F:/CompleteVehicleReID/ImageUI/image_output/CarImage/'
        # vehicleSearch(xlsName, Videoname, srcImg_path, templateImg_path)
        vehicleReIDPath = "vehicleSearch.exe"
        os.system(vehicleReIDPath + " " + '"' + srcImg_path + '"' + " " + '"' + templateImg + '"')

        # os.popen(r"vehicleSearch.exe F:/VehicleReID_20180615/src/ F:/VehicleReID_20180615/template/")
        vehicleSearchResult = sio.loadmat('vehicleSearch.mat')
        searchResult = vehicleSearchResult['FeatureMatchResult']
        #rankcarPath = (os.getcwd() + '\\image_output\\rankImage\\')  # 车牌图像路径 'F:\\CompleteVehicleReID\\ImageUI\\rankImage\\'
        rank = np.array(searchResult)
        # print(carNum)
        imgs = os.listdir(srcImg_path)
        carsNum = len(imgs)
        data = xlrd.open_workbook(xlsName)#读了车牌检测的xls
        table = data.sheets()[0]
        t = os.path.getctime(Videoname)
        timeStruct = time.localtime(t)
        print(timeStruct)
        year = time.strftime('%Y', timeStruct)
        month = time.strftime('%m', timeStruct)
        day = (time.strftime('%d', timeStruct))
        hour = int(time.strftime('%H', timeStruct))
        minute = int(time.strftime('%M', timeStruct))
        second = int((time.strftime('%S', timeStruct)))
        print(carsNum)
        f = open('rankResult.txt','w')
        #rankcarPath = os.path.abspath('.') + '\\rankImage\\'
        rankcarPath = os.path.abspath('.') + '\\rankImage\\'
        shutil.rmtree( rankcarPath)
        os.mkdir( rankcarPath)

        for i in range(0, carsNum):
            print('第'+str(i+1)+'幅')
            rankCar = rank[:, i]
            rankCar = np.int(rankCar)
            row = table.row_values(rankCar)
            row = np.array(row)
            frameNum = row[0]
            carlocation = row[1]
            PlateNum = row[2]
            # frameTime = getFrameTime(Videoname, frameNum)

            # videoTime=time.strftime('%Y-%m-%d %H:%M:%S',timeStruct)
            # 获取视频FPS
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
            if int(major_ver) < 3:
                fps = VC.get(cv2.cv.CV_CAP_PROP_FPS)
                # print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
            else:
                fps = VC.get(cv2.CAP_PROP_FPS)
                # print( "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
            inter_time = round(int(frameNum) * (1 / fps))
            h, s1 = divmod(inter_time + second, 3600)
            m, s = divmod(s1, 60)
            hour1 = hour + h
            hour2 = str(hour1)
            if hour1 < 10:
                hour2 = '0' + hour2
            minute1 = minute + m
            minute2 = str(minute1)
            if minute1 < 10:
                minute2 = '0' + minute2
            second1 = s
            second2 = str(second1)
            if second1 < 10:
                second2 = '0' + second2
            frameTime = year + '-' + month + '-' + day + ' ' + hour2 + ':' + minute2 + ':' + second2
            rankInfo = '排名第' + str(i+1) + '的车辆出现在：' + frameTime
            print(rankInfo+'  '+'车牌号为：'+PlateNum)
            f.write(rankInfo+' '+'车牌号为：'+PlateNum+'\n')
            VC.set(cv2.CAP_PROP_POS_FRAMES, int(frameNum))
            a, rankImg = VC.read()
            rankImg = self.format_img_size(rankImg, C)
            b1 = carlocation.find(',')
            x1 = int(carlocation[1:b1])
            #print(x1)
            b2 = carlocation.find(',', b1 + 1)
            y1 = int(carlocation[b1 + 2:b2])
            #print(y1)
            b3 = carlocation.find(',', b2 + 1)
            x2 = int(carlocation[b2 + 2:b3])
            #print(x2)
            b4 = carlocation.find(',', b3 + 1)
            y2 = int(carlocation[b3 + 2:b4])
            #print(y2)
            cv2.rectangle(rankImg, (x1, y1),( x2, y2), (0, 255, 0), 3)


            rankcarName= str(i+1)+'.jpg'
            rankpath = os.path.join(rankcarPath, rankcarName)
            cv2.imwrite(rankpath, rankImg)

        f.close()
        VC.release()
        self.sg_finished.emit(self.VideoName)

app = qt.QApplication(sys.argv)
tool = Ui_carSearchSystem()
sys.exit(app.exec_())


