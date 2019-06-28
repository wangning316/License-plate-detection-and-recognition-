# -*- coding: utf-8 -*-

'''
这版上YOLO,我需要解决一些车牌定位问题，先在传统方法上做一些尝试
这个版本有一些小的BUG没有解决
车辆检测：YOLO v3
颜色识别：呵呵
车牌定位:EasyPR
车牌分割：连通域外接矩形
汉字识别：双字典模型 DPL
字符识别：CNN
'''
"""
Created on Sat Jul 28 19:02:17 2018

@author: Warning

"""
import os
import cv2
import sys
import gc
import time
import subprocess
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from skimage import measure, morphology
from ctypes import cdll, c_int, POINTER
import xlwt
import xlrd
import scipy.io as sio
import shutil

import colorsys
from timeit import default_timer as timer
import numpy as np
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from keras.utils.training_utils import multi_gpu_model
from keras_yolo3.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras_yolo3.yolo3.utils import letterbox_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_num = 0
#  YOLO GPU

from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.Qt as qt

sys.setrecursionlimit(40000)


# -------------------------------------------- 主界面 ------------------------------------------#
class Ui_ImageSystem(qt.QObject):  # QObject类可以用来传递信号
    sg_pause = qt.pyqtSignal(str)  # 信号
    sg_continue = qt.pyqtSignal(str)
    sg_visualize = qt.pyqtSignal(bool)

    def __init__(self, parent=qt.QObject()):  # 初始化实例调用数据
        super(Ui_ImageSystem, self).__init__(parent)  # 使用super查找

        self.ImageSystem = qt.QWidget()  # 定义ImageSystem是一个Widget对象
        self.setupUi(self.ImageSystem)  # 初始化Widget类
        # self.ImageSystem.show()             #显示窗口
        # self.ImageSystem.setWindowTitle('车辆车牌检测识别系统')

        self.CarSearchWindow = qt.QWidget()  # 子窗口CarSearchWindow是一个QWidget类
        self.carsearch = Ui_carSearchSystem()  # 实例化Ui_CarSearch
        self.carsearch.setupUi(self.CarSearchWindow)  # setupUi实际上是对CarSearchWindow做初始化

        self.Tab = qt.QTabWidget()  # TabWidget
        self.Tab.resize(1000, 785)
        self.Tab.addTab(self.ImageSystem, u"车牌识别")
        self.Tab.addTab(self.CarSearchWindow, u"车辆检索")
        self.Tab.show()
        self.Tab.setWindowTitle('车辆识别检索应用系统')
        self.Tab.setStyleSheet("border-color: rgb(245, 245, 245);\n"
                               "background-color: rgb(245, 245, 245);\n"
                               "")

        self.SelectMenu = qt.QMenu()
        self.SelectMenu.addAction('图像源', self.selectFiles)
        self.SelectMenu.addAction('视频源', self.selectVideos)
        self.SelectFile.setMenu(self.SelectMenu)  # 选择按钮变成菜单

        self.Workmenu = qt.QMenu()  # 定义menu对象
        self.Workmenu.addAction('开始', self.StartProcess)  # menu对象添加开始操作
        self.Workmenu.addAction('暂停', self.workPause)  # menu对象添加暂停操作
        self.Workmenu.addAction('继续', self.workContinue)  # menu对象添加继续操作
        self.Start_Btn.setMenu(self.Workmenu)  # Button控件变成下拉菜单

        self.Visible_Btn.clicked.connect(self.ResultVisualize)
        self.Exit_Btn.clicked.connect(self.ExitUI)  # 退出和函数连接
        self.thread_dic = dict()  # 线程字典用来存储信息
        self.backwatch_dic = dict()
        self.TableNum = 0  # 显示结果TableWidget的行数
        self.filename = ''  # 初始化图像文件夹路径
        self.VideoName = ''  # 初始化视频文件
        self.visualize_flag = False
        # 修改环境变量
        # mypath = os.path.abspath('.') + '\\v83\\runtime\\win64'
        # os.environ["path"] = mypath + ";" + os.environ["path"]

    def setupUi(self, ImageSystem):  # 初始化ImageSystem
        ImageSystem.setObjectName("ImageSystem")
        ImageSystem.resize(1000, 785)
        self.SelectFile = QtWidgets.QPushButton(ImageSystem)
        self.SelectFile.setGeometry(QtCore.QRect(710, 670, 91, 23))
        self.SelectFile.setObjectName("SelectFile")
        self.Exit_Btn = QtWidgets.QPushButton(ImageSystem)
        self.Exit_Btn.setGeometry(QtCore.QRect(830, 710, 91, 23))
        self.Exit_Btn.setObjectName("Exit_Btn")
        self.Rec_Results = QtWidgets.QTableWidget(ImageSystem)
        self.Rec_Results.setGeometry(QtCore.QRect(710, 70, 221, 580))
        self.Rec_Results.setObjectName("Rec_Results")
        self.Rec_Results.setColumnCount(2)  # 设置两列TableWidget
        self.Rec_Results.setHorizontalHeaderLabels(['图片名称', '识别结果'])  # 设置表头
        self.Rec_Results.verticalHeader().setVisible(False)  # 关闭序号
        self.Rec_Results.horizontalHeader().setStretchLastSection(True)  # 末尾填充满表格
        self.Rec_Results.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                       "background-color: rgb(255, 255, 255);\n"
                                       "")
        # 实现点击TableWidget回放功能
        self.Rec_Results.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.Rec_Results.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.Rec_Results.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.Rec_Results.cellClicked.connect(self.backWatch)  # 将cellClicked信号与函数backwatch绑定

        self.label = QtWidgets.QLabel(ImageSystem)
        self.label.setGeometry(QtCore.QRect(70, 40, 51, 21))
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(ImageSystem)
        self.label_4.setGeometry(QtCore.QRect(710, 50, 54, 12))
        self.label_4.setObjectName("label_4")
        self.Visible_Btn = QtWidgets.QPushButton(ImageSystem)
        self.Visible_Btn.setGeometry(QtCore.QRect(710, 710, 91, 23))
        self.Visible_Btn.setObjectName("CarSearch")
        self.label_2 = QtWidgets.QLabel(ImageSystem)
        self.label_2.setGeometry(QtCore.QRect(70, 70, 580, 580))
        self.label_2.setMaximumSize(QtCore.QSize(580, 580))
        self.label_2.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(255, 255, 255);\n"
                                   "")

        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_2.setAlignment(qt.Qt.AlignCenter)  # 设置label的内容居中
        self.Start_Btn = QtWidgets.QPushButton(ImageSystem)
        self.Start_Btn.setGeometry(QtCore.QRect(830, 670, 91, 23))
        self.Start_Btn.setObjectName("Start_Btn")

        self.label_3 = QtWidgets.QLabel(ImageSystem)
        self.label_3.setGeometry(QtCore.QRect(330, 650, 91, 31))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")

        self.retranslateUi(ImageSystem)
        QtCore.QMetaObject.connectSlotsByName(ImageSystem)

    def retranslateUi(self, ImageSystem):
        _translate = QtCore.QCoreApplication.translate
        ImageSystem.setWindowTitle(_translate("ImageSystem", "车辆检索示例"))
        self.SelectFile.setText(_translate("ImageSystem", "选择"))
        self.Exit_Btn.setText(_translate("ImageSystem", "退出"))
        self.label.setText(_translate("ImageSystem", "检测窗口"))
        self.label_4.setText(_translate("ImageSystem", "识别结果"))
        self.Visible_Btn.setText(_translate("ImageSystem", "可视化关"))
        self.Start_Btn.setText(_translate("ImageSystem", "选项"))

    def selectFiles(self):  # 选择文件路径函数
        filedlg = qt.QFileDialog()  # 文件选择类的对象
        filedlg.setFileMode(qt.QFileDialog.ExistingFile)  # 设置选择类型是文件
        self.filename = filedlg.getExistingDirectory(None, "选择文件", os.getcwd())  # 选择文件夹，返回文件夹路径
        if len(self.filename) == 0:
            qt.QMessageBox.warning(None, "警告", "没有选中文件夹", qt.QMessageBox.Yes)  # 没有选中文件夹弹出MessageBox
        return self.filename

    def selectVideos(self):
        self.VideoName, _ = qt.QFileDialog.getOpenFileName(None, '选择视频', os.getcwd(), 'Image files(*.MP4 *.avi)')
        if len(self.VideoName) == 0:
            qt.QMessageBox.warning(None, "警告", "没有选中视频文件", qt.QMessageBox.Yes)  # 没有选中文件夹弹出MessageBox
        print(self.VideoName)
        return self.VideoName

    def ExitUI(self):  # 退出界面
        reply = qt.QMessageBox.question(None, "车辆车牌识别系统", "退出?", qt.QMessageBox.Yes, qt.QMessageBox.No)  # 退出界面选择
        if reply == qt.QMessageBox.Yes:
            sys.exit(app.exec_())
        else:
            return

    def backWatch(self, row, col):
        print(row)
        backWatch_image = self.backwatch_dic[row]
        self.label_2.setPixmap(backWatch_image)

    def StartProcess(self):  # 开始处理函数
        if len(self.filename) > 0:
            filepath = self.filename
            self.filename2 = filepath
            self.backwatch_dic.clear()
            self.controller = Controller(filepath)  # Controller的实例化对象
            self.controller.sg_show.connect(self.showImage)  # 显示的内容传给showImage函数
            self.controller.sg_addtable.connect(self.addTbaleWidget)  # TableNum要自加1
            self.sg_visualize.connect(self.controller.Visualize)

            self.label_2.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                       "background-color: rgb(255, 255, 255);\n"
                                       "")  # 初始化显示Label的背景和边界，每次开始任务都是白色背景
            self.label_2.setPixmap(qt.QPixmap(""))  # 设置开始显示内容为空
            self.Rec_Results.clearContents()  # 清空上次TabelWidget里的结果
            self.TableNum = 0  # 重新设置行数为0
            self.Rec_Results.setRowCount(self.TableNum)
            self.label_3.setText("任务执行")  # 正在进行的状态

            thread = qt.QThread()  # 开启一个线程
            self.thread_dic[filepath] = thread  # 线程和文件夹名字对应

            self.sg_continue.connect(self.controller.work_continue)  # 信号链接的是线程的阻塞函数
            self.sg_pause.connect(self.controller.work_pause)  # 暂停信号
            self.controller.sg_finished.connect(self.finished)  # controller里的finish信号
            self.controller.moveToThread(thread)  # 移动到子线程
            thread.started.connect(self.controller.ImageProcess)  # 线程连接处理函数
            thread.start()  # 线程开启
            self.filename = ''
        elif len(self.VideoName) > 0:
            print('shipin')
            filepath = self.VideoName
            self.filename2 = filepath
            self.backwatch_dic.clear()
            self.controller = Controller(filepath)  # Controller的实例化对象
            self.controller.sg_show.connect(self.showImage)  # 显示的内容传给showImage函数
            self.controller.sg_addtable.connect(self.addTbaleWidget)  # TableNum要自加1
            self.sg_visualize.connect(self.controller.Visualize)

            self.label_2.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                       "background-color: rgb(255, 255, 255);\n"
                                       "")  # 初始化显示Label的背景和边界，每次开始任务都是白色背景
            self.label_2.setPixmap(qt.QPixmap(""))  # 设置开始显示内容为空
            self.Rec_Results.clearContents()  # 清空上次TabelWidget里的结果
            self.TableNum = 0  # 重新设置行数为0
            self.Rec_Results.setRowCount(self.TableNum)
            self.label_3.setText("任务执行")  # 正在进行的状态

            thread = qt.QThread()  # 开启一个线程
            self.thread_dic[filepath] = thread  # 线程和文件夹名字对应

            self.sg_continue.connect(self.controller.work_continue)  # 信号链接的是线程的阻塞函数
            self.sg_pause.connect(self.controller.work_pause)  # 暂停信号
            self.controller.sg_finished.connect(self.finished)  # controller里的finish信号
            self.controller.moveToThread(thread)  # 移动到子线程
            thread.started.connect(self.controller.VideoProcess)  # 线程连接处理函数
            thread.start()  # 线程开启
            self.VideoName = ''

    def img2pixmap(self, image):  # ndarray 转 QPixMap
        Y, X = image.shape[:2]  # image的长和宽
        if X > Y:  # 设定图像大小最适应显示窗口label2的大小
            ratio = X / self.label_2.width()
            Y = int(Y / ratio)
            X = self.label_2.width()
            newsize = (self.label_2.width(), Y)
            image = cv2.resize(image, newsize, interpolation=cv2.INTER_CUBIC)
        else:
            ratio = Y / self.label_2.height()
            X = int(X / ratio)
            Y = self.label_2.height()
            newsize = (X, self.label_2.height())
            image = cv2.resize(image, newsize, interpolation=cv2.INTER_CUBIC)
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = image[..., 2]
        self._bgra[..., 1] = image[..., 1]
        self._bgra[..., 2] = image[..., 0]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)  # 转化为QImage
        qimage = qimage.rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap

    def showImage(self, image):
        self.label_2.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(0, 0, 0);\n"
                                   "")  # 显示label设置背景为黑色
        pixmap = self.img2pixmap(image)  # CV数据转为Qt的pixmap
        self.backwatch_dic[self.TableNum] = pixmap  # 图像与行数对应
        self.label_2.setPixmap(pixmap)  # 显示图像

    def addTbaleWidget(self, image_name, rec_result):  # 添加TableWidget内容
        self.Rec_Results.setRowCount(self.TableNum + 1)  # 行数加1
        self.Rec_Results.setItem(self.TableNum, 0, QtWidgets.QTableWidgetItem(image_name))  # 设置内容：图片名
        self.Rec_Results.setItem(self.TableNum, 1, QtWidgets.QTableWidgetItem(rec_result))  # 设置内容：识别结果
        self.TableNum = self.TableNum + 1

    def ResultVisualize(self):
        if self.visualize_flag == False:
            self.Visible_Btn.setText('可视化开')
            self.visualize_flag = True
            self.sg_visualize.emit(self.visualize_flag)  # 发送的是True
        else:
            self.Visible_Btn.setText('可视化关')
            self.visualize_flag = False
            self.sg_visualize.emit(self.visualize_flag)  # 发送的是False

    def workPause(self):  # 控制子线程暂停
        self.label_3.setText("任务暂停")
        self.sg_pause.emit(self.filename2)

    def workContinue(self):  # 控制子线程继续
        self.label_3.setText("任务执行")
        self.sg_continue.emit(self.filename2)

    # def workFinished(self):
    #     self.sg_finished.emit(self.filename)

    def finished(self, filename):  # 子线程结束销毁
        self.thread_dic[filename].quit()
        self.thread_dic[filename].wait()
        self.thread_dic.pop(filename)
        del self.controller
        # self.backwatch_dic.clear()
        # gc.collect()
        self.label_3.setText("任务结束")

    def __del__(self):  # 界面的析构函数，删除界面变量ImageSystem之前停掉主线程
        for thread in self.thread_dic:
            thread.quit()
            thread.wait()
            gc.collect()


class TableWidget(qt.QTableWidget):
    def __init__(self, parent=None):
        super(TableWidget, self).__init__(parent)
        self.setGeometry(QtCore.QRect(640, 70, 221, 580))
        self.setObjectName("Rec_Results")
        self.setColumnCount(2)  # 设置两列TableWidget
        self.setHorizontalHeaderLabels(['图片名称', '识别结果'])  # 设置表头
        self.verticalHeader().setVisible(False)  # 关闭序号
        self.horizontalHeader().setStretchLastSection(True)  # 末尾填充满表格
        self.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                           "background-color: rgb(255, 255, 255);\n"
                           "")
        # 实现点击TableWidget回放功能
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)


class ThreadProcess(qt.QObject):  # 这个将来需要改成线程池
    def __init__(self, filename):
        super(ThreadProcess, self).__init__()
        self.maxThreads = qt.QThreadPool.globalInstance().maxThreadCount()
        self.mutex_process = qt.QMutex(qt.QMutex.Recursive)  # 互斥锁：用于阻塞线程
        self.currentThreads = 0
        self.filename = filename


class Controller(qt.QObject):  # 子线程处理类
    sg_show = qt.pyqtSignal(np.ndarray)  # 发送信号，内容为显示图片
    sg_finished = qt.pyqtSignal(str)
    sg_addtable = qt.pyqtSignal(str, str)

    def __init__(self, OpenFiles):
        super(Controller, self).__init__()
        self.image_file = OpenFiles
        # self.video_file = VideoFile
        self.mutex_process = qt.QMutex(qt.QMutex.Recursive)  # 互斥锁：用于阻塞线程
        self.threadpause = False  # 暂停标识
        self.threadcontinue = False  # 继续标识
        self.visualize = False  # 可视化标识
        self.car_save_path = (os.getcwd() + '\\image_output\\CarImage\\')  # 车辆图像存储路径
        #self.detect_img = list()  # python 循环变量结束后会自动释放

    def Visualize(self, vis_flag):
        if vis_flag == True:
            self.visualize = True  # 显示结果
            self.sg_show.emit(self.detect_img)
            print('T')
        else:
            self.visualize = False  # 显示原图
            print('F')
            self.sg_show.emit(self.ori_image)


    def work_continue(self, filename):  # 继续函数 解锁可以访问filename变量
        if filename != self.image_file or self.threadcontinue:
            return
        self.mutex_process.unlock()  # 解锁
        self.threadcontinue = True
        self.threadpause = False


    def work_pause(self, filename):  # 暂停函数 如果锁住无法访问filename变量
        if filename != self.image_file or self.threadpause:
            return
        self.mutex_process.lock()
        self.threadpause = True
        self.threadcontinue = False


    def work_cancel(self, filename):
        if self.videoname == filename:
            self.video_cancel = True
        elif self.image_cancel == False and self.imagepath == filename:
            self.image_cancel = True
        else:
            return

    def _get_class(self):  # YOLO初始化类别，下面几个都是yolo-keras的初始化
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):  # 获取检测的尺度 9个
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        # hsv_tuples = [(x / len(self.class_names), 1., 1.)
        #               for x in range(len(self.class_names))]
        # self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # self.colors = list(
        #     map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        #         self.colors))
        # np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        # np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        # np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):  # 物体检测
        start = timer()
        car_position = list()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # print(self.sess.run(
        #     [self.boxes, self.scores, self.classes],
        #     feed_dict={
        #         self.yolo_model.input: image_data,
        #         self.input_image_shape: [image.size[1], image.size[0]],
        #         K.learning_phase(): 0
        #     }))#总是深度玄学 加这个不报错
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        car_area_threshold = image.size[0] * image.size[1] / 120

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            height = box[2] - box[0]
            width = box[3] - box[1]
            if (predicted_class == 'car' or predicted_class == 'truck') and score > 0.5 and height * width > car_area_threshold \
                    and 0.5 <= (height / width) <= 2:  # 限制条件：检测类别，概率，面积，宽高比
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                car_position.append([left + i, top + i, right - i, bottom - i])

        end = timer()
        print(end - start)
        return car_position

    def close_session(self):
        self.sess.close()

    def to_bytes(self, bytes_or_str):  # 转为byte类型字符串 C++dll不能识别utf-8编码的字符串
        if isinstance(bytes_or_str, str):  # 判断是否是字符串 isinstance判断类型
            return bytes_or_str.encode('utf-8')  # utf-8编码
        return bytes_or_str

    def avg(self, array):
        sum = 0
        n = len(array)
        for num in array:
            sum = sum + num
        avge = 1.0 * sum / (n + 0.1)  # 改过
        return avge

    def seg_4_plate(self, img):
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

        char_result = []
        char_result.append(str(table[num2]))
        char_result.append(str(table[num3]))
        char_result.append(str(table[num4]))
        char_result.append(str(table[num5]))
        char_result.append(str(table[num6]))
        char_result.append(str(table[num7]))
        prob = str([prob2[0][num2], prob3[0][num3], prob4[0][num4], prob5[0][num5], prob6[0][num6], prob7[0][num7]])

        return char_result, prob

    def ImageProcess(self):
        print("进入CONTROLLER")
        image_file = self.image_file
        self.excelResults = list()
        self.excelName = list()
        self.model_path = os.getcwd() + '\\keras-yolo3\\model_data\\yolo.h5'  # model path or trained weights path
        self.anchors_path = os.getcwd() + '\\keras-yolo3\\model_data\\yolo_anchors.txt'
        self.classes_path = os.getcwd() + '\\keras-yolo3\\model_data\\coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

        image_data = np.ones((416, 416, 3))
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        ob, osc, oc = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [654, 512],
                K.learning_phase(): 0
            })  # 深度玄学

        print("开始读图")

        # for img_name in os.listdir(image_file):       改文件名字
        #     imgfilepath = os.path.join(image_file, img_name)
        #     os.rename(imgfilepath,os.path.join(image_file,img_name[1:len(img_name)]))

        for img_name in os.listdir(image_file):
            imgfilepath = os.path.join(image_file, img_name)
            img_frame = cv2.imread(imgfilepath)

            self.ori_image = img_frame.copy()  # 深拷贝
            self.mutex_process.lock()
            print(image_file)
            self.sg_show.emit(img_frame)
            self.mutex_process.unlock()
            self.detect_img = img_frame

            pil_frame = Image.fromarray(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
            car_position = self.detect_image(pil_frame)

            for car_num in range(np.shape(car_position)[0]):  # 存储车辆图像
                self.mutex_process.lock()
                print(image_file)
                if car_position[car_num][0] < 0 or car_position[car_num][1] < 0 or car_position[car_num][2] < 0 or \
                        car_position[car_num][3] < 0:
                    continue
                img_car = img_frame[car_position[car_num][1]:car_position[car_num][3],
                          car_position[car_num][0]: car_position[car_num][2], :]  # y1:y2,x1:x2
                car_name = img_name[:len(img_name) - 4] + '_' + str(car_num) + '.jpg'  # 第i辆车
                cv2.imwrite(self.car_save_path + car_name, img_car)  # 定位的车辆

                print(car_position)
                print(car_position[car_num])
                cv2.rectangle(self.detect_img, (car_position[car_num][0], car_position[car_num][1]),
                              (car_position[car_num][2], car_position[car_num][3]), (0, 255, 0), 2)

                if self.visualize:
                    img_show = self.detect_img[:, :, :]
                    print('发结果')
                else:
                    img_show = self.ori_image
                    print('发原图')

                self.sg_show.emit(img_show)
                self.mutex_process.unlock()
                # self.sg_finished.emit(self.image_file)

                # exe隐藏cmd窗口
                st = subprocess.STARTUPINFO()
                st.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
                st.wShowWindow = subprocess.SW_HIDE
                # 颜色识别暂时不用
                # colorexe = "ColorRec.exe"
                # subprocess.call(colorexe + " " + '"' + self.car_save_path + car_name + '"',startupinfo=st)  # 颜色识别  原来用的os.system
                # color_data = sio.loadmat('color.mat')
                # colorid = color_data['colorid']
                # colorid = np.array(colorid).tolist()  # 数组变为列表
                # colorid = ''.join(str(ccc)for ccc in colorid[0])
                # colordict = {'1':"Black",'2':"Blue",'3':"Cyan",'4':"Green",'5':"Gray",'6':"Red",'7':"White",'8':"Yellow"}
                #
                # self.mutex_process.lock()
                # print(image_file)
                # font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                # cv2.putText(self.detect_img, colordict[colorid], (car_position[car_num][0]-20, car_position[car_num][1]),font, 1.0, (255, 255, 255), 2)
                # img_show = self.detect_img[:, :, :]
                # self.sg_show.emit(img_show)
                # self.mutex_process.unlock()

                rootdir = os.path.abspath('.') + '\\image_output\\CarImage\\'
                down = os.path.abspath('.') + '\\image_output\\PlateImage\\'
                path = os.path.join(rootdir, car_name)
                path = self.to_bytes(path)  # 转为byte型的字符
                down = self.to_bytes(down)

                dll = cdll.LoadLibrary("PlateDetectDll.dll")
                dll.fnPlateDetectDll.restype = POINTER(c_int)  # 重载API的返回值类型  C++中返回的是int*
                locate_out = dll.fnPlateDetectDll(path, down)  # 输出了车牌在车辆图像中车牌的坐标信息

                if bool(locate_out) == True:
                    for j in range(0, 4):  # 算车牌在原图的坐标
                        locate_out[2 * j] = locate_out[2 * j] + car_position[car_num][0]  # car_pos这个场景中第k辆车的坐标信息
                        locate_out[2 * j + 1] = locate_out[2 * j + 1] + car_position[car_num][1]
                    cv2.line(self.detect_img, (locate_out[0], locate_out[1]), (locate_out[2], locate_out[3]),
                             (0, 0, 255), 2)  # 在原图里的画出车牌位置
                    cv2.line(self.detect_img, (locate_out[2], locate_out[3]), (locate_out[4], locate_out[5]),
                             (0, 0, 255), 2)
                    cv2.line(self.detect_img, (locate_out[4], locate_out[5]), (locate_out[6], locate_out[7]),
                             (0, 0, 255), 2)
                    cv2.line(self.detect_img, (locate_out[6], locate_out[7]), (locate_out[0], locate_out[1]),
                             (0, 0, 255), 2)

                    self.mutex_process.lock()
                    print(image_file)
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # opencv的默认字体
                    cv2.putText(self.detect_img, "Located!", (car_position[car_num][0] + 50, car_position[car_num][1]),
                                font, 1.0, (0, 255, 0), 2)

                    if self.visualize:
                        img_show = self.detect_img[:, :, :]
                        print('发结果')
                    else:
                        img_show = self.ori_image
                        print('发原图')
                    self.sg_show.emit(img_show)
                    self.mutex_process.unlock()

                    platepath = (os.getcwd() + '\\image_output\\PlateImage\\')  # 车牌图像路径
                    chin_path = (os.getcwd() + '\\image_output\\Chin/')  # 中文字符路径

                    path = os.path.join(platepath, car_name)
                    plate = cv2.imread(path)  # 车牌图像
                    if self.seg_4_plate(plate) == 0:
                        print(path, "Can Not Segment!")

                        self.mutex_process.lock()
                        print(image_file)
                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                        cv2.putText(self.detect_img, "Segment Error!",
                                    (car_position[car_num][0] + 50, car_position[car_num][1]), font, 1.0, (0, 0, 255),
                                    2)
                        final_result = ['无法分割']
                        final_result = ''.join(final_result)
                        self.excelResults.append(final_result)
                        self.excelName.append(car_name)
                        self.sg_addtable.emit(str(car_name), str(final_result))

                        if self.visualize:
                            img_show = self.detect_img[:, :, :]
                            print('发结果')
                        else:
                            img_show = self.ori_image
                            print('发原图')

                        self.sg_show.emit(img_show)
                        self.mutex_process.unlock()


                    else:
                        print(path, "Can Segment")
                        p1, p2, p3, p4, p5, p6, p7 = self.seg_4_plate(plate)  # 分割字符

                        # -------------------------字符识别模块---------------------#
                        cv2.imwrite(chin_path + 'chin.jpg', p1)  # 存储汉字字符
                        chinexe = "chin_rec_build.exe"
                        subprocess.call(chinexe + " " + '"' + chin_path + '"', startupinfo=st)  # 路径中有空格 需要加""
                        chin_data = sio.loadmat('result.mat')
                        chin_rec = chin_data['result']  # 汉字识别结果写在list里面
                        char_result, recog_prob = self.char_recog(p2, p3, p4, p5, p6, p7)  # recog_result要显示在list里面
                        chin_rec = np.array(chin_rec).tolist()  # 数组变为列表
                        final_result = np.append(chin_rec, char_result)
                        final_result = np.array(final_result).tolist()  # array转为list
                        final_result = ''.join(final_result)  # list转为str
                        self.excelResults.append(final_result)
                        self.excelName.append(car_name)

                        cv2img = cv2.cvtColor(self.detect_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                        pilimg = Image.fromarray(cv2img)
                        draw = ImageDraw.Draw(pilimg)  # 图片上打印
                        font = ImageFont.truetype("simsun.ttc", 20,
                                                  index=1)  # 参数1：字体文件路径，参数2：字体大小"simhei.ttf",20, encoding="utf-8"
                        draw.text((locate_out[0], locate_out[1] + 10), str(final_result), (255, 0, 0),
                                  font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                        self.detect_img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

                        self.mutex_process.lock()
                        print(image_file)

                        self.sg_addtable.emit(str(car_name), str(final_result))
                        if self.visualize:
                            img_show = self.detect_img[:, :, :]
                            print('发结果')
                        else:
                            img_show = self.ori_image
                            print('发原图')
                        self.sg_show.emit(img_show)
                        self.mutex_process.unlock()


                else:
                    self.mutex_process.lock()
                    print(image_file)
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    cv2.putText(self.detect_img, "Unlocated!",
                                (car_position[car_num][0] + 50, car_position[car_num][1]), font, 1.0, (0, 0, 255), 2)
                    final_result = ['无法定位']
                    self.excelResults.append(final_result)
                    self.excelName.append(car_name)
                    self.sg_addtable.emit(str(car_name), str(final_result))
                    if self.visualize:
                        img_show = self.detect_img[:, :, :]
                        print('发结果')
                    else:
                        img_show = self.ori_image
                        print('发原图')
                    self.sg_show.emit(img_show)
                    self.mutex_process.unlock()

        self.close_session()
        K.clear_session()

        excel = xlwt.Workbook()
        sheet = excel.add_sheet('Results', cell_overwrite_ok=True)  # 添加工作表，可以覆盖
        sheet.write(0, 0, "图片名称")
        sheet.write(0, 1, "识别结果")
        sheet.write(0, 2, "匹配个数")

        for names in range(len(self.excelName)):
            sheet.write(names + 1, 0, self.excelName[names])
            print(self.excelName[names])

        for rec in range(len(self.excelResults)):
            sheet.write(rec + 1, 1, self.excelResults[rec])
            count = 0
            if len(self.excelResults[rec]) >= 5:
                for rec_index in range(6):
                    if self.excelName[rec][rec_index] == self.excelResults[rec][rec_index + 1]:  # 判断同一位置字符是否相同
                        count = count + 1
            else:
                count = 0
            sheet.write(rec + 1, 2, count)
            print(self.excelResults[rec])

        realtime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # 标准化时间
        excel.save(realtime + '.xls')

        for x in locals().keys():  # 清除函数中的临时变量
            del locals()[x]
        gc.collect()

        self.sg_finished.emit(self.image_file)

    def VideoProcess(self):
        Videoname = self.image_file

        self.model_path = os.getcwd() + '\\keras-yolo3\\model_data\\yolo.h5'  # model path or trained weights path
        self.anchors_path = os.getcwd() + '\\keras-yolo3\\model_data\\yolo_anchors.txt'
        self.classes_path = os.getcwd() + '\\keras-yolo3\\model_data\\coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

        image_data = np.ones((416, 416, 3))
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        ob, osc, oc = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [654, 512],
                K.learning_phase(): 0
            })  # 深度玄学
        self.vexcelResults = []
        self.vexcelName = []

        VC = cv2.VideoCapture(Videoname)
        total_frames = int(VC.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
        FPS = 50
        count = 1
        if VC.isOpened():
            flag, img_frame = VC.read()
            print('读到视频了', '一共', str(total_frames), '帧')
            print(flag)
        else:
            flag = False

        rootdir = os.path.abspath('.') + '\\image_output\\CarImage\\'

        shutil.rmtree(rootdir)  # 清空文件夹
        os.mkdir(rootdir)       #建立文件夹
        down = os.path.abspath('.') + '\\image_output\\PlateImage\\'

        shutil.rmtree(down)# 清空文件夹
        os.mkdir(down)
        while flag:
            flag, img_frame = VC.read()
            if count % FPS == 0:
                print('现在是第', str(count), '帧')

                self.mutex_process.lock()
                self.sg_show.emit(img_frame)
                self.mutex_process.unlock()
                self.detect_img = img_frame

                pil_frame = Image.fromarray(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
                car_position = self.detect_image(pil_frame)

                for car_num in range(np.shape(car_position)[0]):  # 存储车辆图像
                    self.mutex_process.lock()
                    print(Videoname)
                    if car_position[car_num][0] < 0 or car_position[car_num][1] < 0 or car_position[car_num][2] < 0 or \
                            car_position[car_num][3] < 0:
                        continue
                    img_car = img_frame[car_position[car_num][1]:car_position[car_num][3],
                              car_position[car_num][0]: car_position[car_num][2], :]  # y1:y2,x1:x2
                    carFrame = str(count)
                    self.vexcelName.append(carFrame)
                    carFrame = carFrame.zfill(5)
                    car_name = carFrame + '_' + str(car_num) + '.jpg'
                    carpath = os.path.join(self.car_save_path, car_name)
                    cv2.imwrite(carpath, img_car)  # 定位的车辆

                    cv2.rectangle(self.detect_img, (car_position[car_num][0], car_position[car_num][1]),
                                  (car_position[car_num][2], car_position[car_num][3]), (0, 255, 0), 2)
                    img_show = self.detect_img[:, :, :]
                    self.sg_show.emit(img_show)
                    self.mutex_process.unlock()

                    # exe隐藏cmd窗口
                    st = subprocess.STARTUPINFO()
                    st.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
                    st.wShowWindow = subprocess.SW_HIDE

                    rootdir = os.path.abspath('.') + '\\image_output\\CarImage\\'
                    down = os.path.abspath('.') + '\\image_output\\PlateImage\\'
                    path = os.path.join(rootdir, car_name)
                    path = self.to_bytes(path)  # 转为byte型的字符
                    down = self.to_bytes(down)

                    dll = cdll.LoadLibrary("PlateDetectDll.dll")
                    dll.fnPlateDetectDll.restype = POINTER(c_int)  # 重载API的返回值类型  C++中返回的是int*
                    locate_out = dll.fnPlateDetectDll(path, down)  # 输出了车牌在车辆图像中车牌的坐标信息

                    if bool(locate_out) == True:
                        for j in range(0, 4):  # 算车牌在原图的坐标
                            locate_out[2 * j] = locate_out[2 * j] + car_position[car_num][0]  # car_pos这个场景中第k辆车的坐标信息
                            locate_out[2 * j + 1] = locate_out[2 * j + 1] + car_position[car_num][1]
                        cv2.line(self.detect_img, (locate_out[0], locate_out[1]), (locate_out[2], locate_out[3]),
                                 (0, 0, 255), 2)  # 在原图里的画出车牌位置
                        cv2.line(self.detect_img, (locate_out[2], locate_out[3]), (locate_out[4], locate_out[5]),
                                 (0, 0, 255), 2)
                        cv2.line(self.detect_img, (locate_out[4], locate_out[5]), (locate_out[6], locate_out[7]),
                                 (0, 0, 255), 2)
                        cv2.line(self.detect_img, (locate_out[6], locate_out[7]), (locate_out[0], locate_out[1]),
                                 (0, 0, 255), 2)

                        self.mutex_process.lock()
                        print(Videoname)
                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # opencv的默认字体
                        cv2.putText(self.detect_img, "Located!",
                                    (car_position[car_num][0] + 50, car_position[car_num][1]), font, 1.0, (0, 255, 0),
                                    2)
                        img_show = self.detect_img[:, :, :]
                        self.sg_show.emit(img_show)
                        self.mutex_process.unlock()

                        platepath = (os.getcwd() + '\\image_output\\PlateImage\\')  # 车牌图像路径
                        chin_path = (os.getcwd() + '\\image_output\\Chin/')  # 中文字符路径

                        path = os.path.join(platepath, car_name)
                        plate = cv2.imread(path)  # 车牌图像
                        if self.seg_4_plate(plate) == 0:
                            print(path, "Can Not Segment!")

                            self.mutex_process.lock()
                            print(Videoname)
                            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                            cv2.putText(self.detect_img, "Segment Error!",
                                        (car_position[car_num][0] + 50, car_position[car_num][1]), font, 1.0,
                                        (0, 0, 255), 2)
                            final_result = ['无法分割']
                            final_result = ''.join(final_result)
                            self.vexcelResults.append(final_result)
                            self.vexcelName.append(car_name)
                            self.sg_addtable.emit(str(car_name), str(final_result))
                            img_show = self.detect_img[:, :, :]
                            self.sg_show.emit(img_show)
                            self.mutex_process.unlock()

                        else:
                            print(path, "Can Segment")
                            p1, p2, p3, p4, p5, p6, p7 = self.seg_4_plate(plate)  # 分割字符

                            # -------------------------字符识别模块---------------------#
                            cv2.imwrite(chin_path + 'chin.jpg', p1)  # 存储汉字字符
                            chinexe = "chin_rec_build.exe"
                            subprocess.call(chinexe + " " + '"' + chin_path + '"', startupinfo=st)  # 路径中有空格 需要加""
                            chin_data = sio.loadmat('result.mat')
                            chin_rec = chin_data['result']  # 汉字识别结果写在list里面
                            char_result, recog_prob = self.char_recog(p2, p3, p4, p5, p6, p7)  # recog_result要显示在list里面
                            chin_rec = np.array(chin_rec).tolist()  # 数组变为列表
                            final_result = np.append(chin_rec, char_result)
                            final_result = np.array(final_result).tolist()  # array转为list
                            final_result = ''.join(final_result)  # list转为str
                            self.vexcelResults.append(final_result)
                            self.vexcelName.append(car_name)

                            cv2img = cv2.cvtColor(self.detect_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                            pilimg = Image.fromarray(cv2img)
                            draw = ImageDraw.Draw(pilimg)  # 图片上打印
                            font = ImageFont.truetype("simsun.ttc", 20,
                                                      index=1)  # 参数1：字体文件路径，参数2：字体大小"simhei.ttf",20, encoding="utf-8"
                            draw.text((locate_out[0], locate_out[1] + 10), str(final_result), (255, 0, 0),
                                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                            self.detect_img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

                            self.mutex_process.lock()
                            print(Videoname)
                            img_show = self.detect_img[:, :, :]
                            self.sg_show.emit(img_show)
                            self.sg_addtable.emit(str(car_name), str(final_result))
                            self.mutex_process.unlock()

                    else:
                        self.mutex_process.lock()
                        print(Videoname)
                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                        cv2.putText(self.detect_img, "Unlocated!",
                                    (car_position[car_num][0] + 50, car_position[car_num][1]), font, 1.0, (0, 0, 255),
                                    2)
                        final_result = ['无法定位']
                        self.vexcelResults.append(final_result)
                        self.vexcelName.append(car_name)
                        self.sg_addtable.emit(str(car_name), str(final_result))
                        img_show = self.detect_img[:, :, :]
                        self.sg_show.emit(img_show)
                        self.mutex_process.unlock()
            count = count + 1
        # K.clear_session()
        self.close_session()

        excel = xlwt.Workbook()
        sheet = excel.add_sheet('Results', cell_overwrite_ok=True)  # 添加工作表，可以覆盖
        sheet.write(0, 0, "帧号")
        sheet.write(0, 1, "识别结果")

        for names in range(len(self.vexcelName)):
            sheet.write(names + 1, 0, self.vexcelName[names])
            print(self.vexcelName[names])

        for rec in range(len(self.vexcelResults)):
            sheet.write(rec + 1, 1, self.vexcelResults[rec])
            print(self.vexcelResults[rec])

        realtime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # 标准化时间
        excel.save(realtime + 'Video.xls')

        self.sg_finished.emit(self.image_file)

        for x in locals().keys():  # 清除函数中的临时变量
            del locals()[x]
        gc.collect()


class Ui_carSearchSystem(qt.QObject):
    def setupUi(self, carSearchSystem):
        carSearchSystem.setObjectName("carSearchSystem")
        carSearchSystem.resize(920, 800)
        # carSearchSystem.setStyleSheet("background-image:url(background1.jpg)")
        self.startButton = QtWidgets.QPushButton(carSearchSystem)
        self.startButton.setGeometry(QtCore.QRect(30, 450, 100, 30))
        self.startButton.setObjectName("startButton")
        self.Area1 = QtWidgets.QLabel(carSearchSystem)
        self.Area1.setGeometry(QtCore.QRect(15, 8, 400, 20))
        self.Area1.setObjectName("Area1")

        self.Area2 = QtWidgets.QLabel(carSearchSystem)
        self.Area2.setGeometry(QtCore.QRect(440, 9, 54, 16))
        self.Area2.setObjectName("Area2")
        self.PlateInfoButton = QtWidgets.QPushButton(carSearchSystem)
        self.PlateInfoButton.setGeometry(QtCore.QRect(30, 380, 100, 30))
        self.PlateInfoButton.setAccessibleName("")
        self.PlateInfoButton.setObjectName("PlateInfoButton")
        self.lineEdit = QtWidgets.QLineEdit(carSearchSystem)
        self.lineEdit.setGeometry(QtCore.QRect(150, 380, 120, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.ColorInfoButton = QtWidgets.QPushButton(carSearchSystem)
        self.ColorInfoButton.setGeometry(QtCore.QRect(30, 340, 100, 30))
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
        self.ComboBox.setGeometry(QtCore.QRect(150, 340, 100, 30))
        self.radioButton = QtWidgets.QRadioButton(carSearchSystem)
        self.radioButton.setGeometry(QtCore.QRect(160, 410, 100, 30))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(carSearchSystem)
        self.radioButton_2.setGeometry(QtCore.QRect(30, 410, 100, 30))
        self.radioButton_2.setObjectName("radioButton_2")
        self.textEdit = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit.setGeometry(QtCore.QRect(350, 210, 310, 40))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_2.setGeometry(QtCore.QRect(680, 210, 310, 40))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_3.setGeometry(QtCore.QRect(350, 440, 310, 40))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_4 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_4.setGeometry(QtCore.QRect(680, 440, 310, 40))
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_5 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_5.setGeometry(QtCore.QRect(350, 670, 310, 40))
        self.textEdit_5.setObjectName("textEdit_5")
        self.textEdit_6 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_6.setGeometry(QtCore.QRect(680, 670, 310, 40))
        self.textEdit_6.setObjectName("textEdit_6")
        self.textEdit_7 = QtWidgets.QTextEdit(carSearchSystem)
        self.textEdit_7.setGeometry(QtCore.QRect(150, 450, 120, 30))
        self.textEdit_7.setObjectName("textEdit_7")
        self.selectImg = QtWidgets.QPushButton(carSearchSystem)
        self.selectImg.setGeometry(QtCore.QRect(30, 300, 100, 30))
        self.selectImg.setObjectName("selectImg")
        self.selectVideo = QtWidgets.QPushButton(carSearchSystem)
        self.selectVideo.setGeometry(QtCore.QRect(150, 300, 100, 30))
        self.selectVideo.setObjectName("selectImg_2")
        # self.label = QtWidgets.QLabel(carSearchSystem)
        # self.label.setGeometry(QtCore.QRect(180, 350, 60, 20))
        # self.label.setObjectName("label")
        self.showImg = QtWidgets.QLabel(carSearchSystem)
        self.showImg.setGeometry(QtCore.QRect(15, 30, 300, 240))
        self.showImg.setMaximumSize(QtCore.QSize(400, 340))
        self.showImg.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(255, 255, 255);")
        self.showImg.setAlignment(QtCore.Qt.AlignCenter)
        self.showImg.setObjectName("showImg")
        self.label_2 = QtWidgets.QLabel(carSearchSystem)
        self.label_2.setGeometry(QtCore.QRect(350, 30, 310, 180))
        self.label_2.setMaximumSize(QtCore.QSize(350, 180))
        self.label_2.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(255, 255, 255);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(carSearchSystem)
        self.label_3.setGeometry(QtCore.QRect(680, 30, 310, 180))
        self.label_3.setMaximumSize(QtCore.QSize(350, 180))
        self.label_3.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(255, 255, 255);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(carSearchSystem)
        self.label_4.setGeometry(QtCore.QRect(350, 260, 310, 180))
        self.label_4.setMaximumSize(QtCore.QSize(310, 180))
        self.label_4.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(255, 255, 255);")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(carSearchSystem)
        self.label_5.setGeometry(QtCore.QRect(680, 260, 310, 180))
        self.label_5.setMaximumSize(QtCore.QSize(310, 180))
        self.label_5.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(255, 255, 255);")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(carSearchSystem)
        self.label_6.setGeometry(QtCore.QRect(350, 490, 310, 180))
        self.label_6.setMaximumSize(QtCore.QSize(310, 180))
        self.label_6.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(255, 255, 255);")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(carSearchSystem)
        self.label_7.setGeometry(QtCore.QRect(680, 490, 310, 180))
        self.label_7.setMaximumSize(QtCore.QSize(310, 180))
        self.label_7.setStyleSheet("border-color: rgb(0, 0, 0);\n"
                                   "background-color: rgb(255, 255, 255);")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        # self.label_8 = QtWidgets.QLabel(carSearchSystem)
        # self.label_8.setGeometry(QtCore.QRect(180,380, 54, 16))
        # self.label_8.setObjectName("carAreaName")

        self.selectImg.clicked.connect(self.selectImgs)  # 选择文件夹和函数连接
        self.selectVideo.clicked.connect(self.selectVideos)
        self.startButton.clicked.connect(self.startCarSearch)

        self.retranslateUi(carSearchSystem)
        # self.exitButton.clicked.connect(carSearchSystem.close)
        QtCore.QMetaObject.connectSlotsByName(carSearchSystem)

    def retranslateUi(self, carSearchSystem):
        _translate = QtCore.QCoreApplication.translate
        carSearchSystem.setWindowTitle(_translate("carSearchSystem", "车辆检索系统"))
        self.startButton.setText(_translate("carSearchSystem", "开始"))
        # self.exitButton.setText(_translate("carSearchSystem", "退出"))
        self.Area1.setText(_translate("carSearchSystem", "采集中心"))
        self.Area2.setText(_translate("carSearchSystem", "检索结果"))
        # self.label_8.setText(_translate("carSearchSystem", "目标车辆"))
        self.PlateInfoButton.setText(_translate("carSearchSystem", "请输入车辆号牌"))
        self.ColorInfoButton.setText(_translate("carSearchSystem", "请选择车辆颜色"))
        # self.selectColorButton.setText(_translate("carSearchSystem", "black"))
        self.radioButton.setText(_translate("carSearchSystem", "按车牌检索"))
        self.radioButton_2.setText(_translate("carSearchSystem", "按特征检索"))
        self.selectImg.setText(_translate("carSearchSystem", "选择目标车辆"))
        self.selectVideo.setText(_translate("carSearchSystem", "选择目标视频"))
        # self.label.setText(_translate("carSearchSystem", "目标车辆"))
        self.showImg.setText(_translate("carSearchSystem", "显示目标车辆"))
        self.label_2.setText(_translate("carSearchSystem", "rank1"))
        self.label_3.setText(_translate("carSearchSystem", "rank2"))
        self.label_4.setText(_translate("carSearchSystem", "rank3"))
        self.label_5.setText(_translate("carSearchSystem", "rank4"))
        self.label_6.setText(_translate("carSearchSystem", "rank5"))
        self.label_7.setText(_translate("carSearchSystem", "rank6"))

    def selectImgs(self):
        print("load--Img")
        self.ImgName, _ = qt.QFileDialog.getOpenFileName(None, '选择图片', os.getcwd(), 'Image files(*.jpg *.gif *.png)')
        print(self.ImgName)
        pixmap = QtGui.QPixmap(self.ImgName)
        self.showImg.setPixmap(pixmap)  # 在label上显示图片
        self.showImg.setScaledContents(True)
        return self.ImgName

    def selectVideos(self):
        print("load--Video")
        self.VideoName, _ = qt.QFileDialog.getOpenFileName(None, '选择视频', os.getcwd(), 'Image files(*.MP4 *.avi)')
        print(self.VideoName)
        return self.VideoName

    def resultShow(self):
        ImgPath = os.getcwd() + '/rankImage/'
        txtResult = open('rankResult.txt', 'r')
        imgs = os.listdir(ImgPath)
        Num = len(imgs)
        if Num == 1:
            ImgName1 = ImgPath + '1.jpg'
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

        if Num >= 6:
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

            ImgName6 = ImgPath + '6.jpg'
            pixmap6 = QtGui.QPixmap(ImgName6)
            self.label_7.setPixmap(pixmap6)  # 在label上显示图片
            self.label_7.setScaledContents(True)
            result6 = txtResult.readline()
            self.textEdit_6.setText(result6)

    def startCarSearch(self):  # 开始处理函数
        VideoPath = self.VideoName
        ObjectImg = self.ImgName

        self.carsearchcontrol = CarSearchControl(VideoPath, ObjectImg)  # Controller的实例化对象
        self.textEdit_7.setText("任务执行")  # 正在进行的状态
        thread = qt.QThread()  # 开启一个线程
        self.VideoPath = thread

        # self.sg_continue.connect(self.controller.work_continue)  # 信号链接的是线程的阻塞函数
        # self.sg_pause.connect(self.controller.work_pause)

        self.carsearchcontrol.sg_finished.connect(self.finished)
        self.carsearchcontrol.sg_setTaskStatus.connect(self.setTaskStatus)
        self.carsearchcontrol.moveToThread(thread)
        thread.started.connect(self.carsearchcontrol.CarSearchProcess)
        thread.start()

    def finished(self, VideoName):
        print('任务结束')
        del self.VideoName
        self.textEdit_7.setText("任务结束")
        self.resultShow()

    def setTaskStatus(self, TaskStatus):
        self.textEdit_7.setText(TaskStatus)


class CarSearchControl(qt.QObject):
    sg_finished = qt.pyqtSignal(str)
    sg_setTaskStatus = qt.pyqtSignal(str)

    def __init__(self, OpenFiles, SelectedImg):
        super(CarSearchControl, self).__init__()
        self.videoName = OpenFiles
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

    def _get_class(self):  # YOLO初始化类别，下面几个都是yolo-keras的初始化
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):  # 获取检测的尺度 9个
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):  # 加载模型
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):  # 物体检测
        start = timer()
        car_position = list()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        car_area_threshold = image.size[0] * image.size[1] / 120
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            height = box[2] - box[0]
            width = box[3] - box[1]
            if (predicted_class == 'car' or predicted_class == 'truck' or predicted_class == 'bus') and score > 0.5 and height * width > car_area_threshold \
                    and 0.5 <= (height / width) <= 2:  # 限制条件：检测类别，概率，面积，宽高比

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                car_position.append([left + i, top + i, right - i, bottom - i])

        end = timer()
        print(end - start)
        return car_position

    def close_session(self):
        self.sess.close()

    def to_bytes(self, bytes_or_str):  # 转为byte类型字符串 C++dll不能识别utf-8编码的字符串
        if isinstance(bytes_or_str, str):  # 判断是否是字符串 isinstance判断类型
            return bytes_or_str.encode('utf-8')  # utf-8编码
        return bytes_or_str

    def avg(self, array):
        sum = 0
        n = len(array)
        for num in array:
            sum = sum + num
        avge = 1.0 * sum / (n + 0.1)  # 改过
        return avge

    def seg_4_plate(self, img):
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

        char_result = []
        char_result.append(str(table[num2]))
        char_result.append(str(table[num3]))
        char_result.append(str(table[num4]))
        char_result.append(str(table[num5]))
        char_result.append(str(table[num6]))
        char_result.append(str(table[num7]))
        prob = str([prob2[0][num2], prob3[0][num3], prob4[0][num4], prob5[0][num5], prob6[0][num6], prob7[0][num7]])

        return char_result, prob

    def CarSearchProcess(self):
        Videoname = self.videoName
        self.sg_setTaskStatus.emit("正在进行车辆检测")


        self.model_path = os.getcwd() + '\\keras-yolo3\\model_data\\yolo.h5'  # model path or trained weights path
        self.anchors_path = os.getcwd() + '\\keras-yolo3\\model_data\\yolo_anchors.txt'
        self.classes_path = os.getcwd() + '\\keras-yolo3\\model_data\\coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()


        image_data = np.ones((416, 416, 3))
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        ob, osc, oc = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [654, 512],
                K.learning_phase(): 0
            })  # 深度玄学

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
        FPS = 20
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
                pil_frame = Image.fromarray(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))
                car_position = self.detect_image(pil_frame)

                for car_num in range(np.shape(car_position)[0]):  # 存储车辆图像

                    if car_position[car_num][0] < 0 or car_position[car_num][1] < 0 or car_position[car_num][2] < 0 or \
                            car_position[car_num][3] < 0:
                        continue
                    img_car = img_frame[car_position[car_num][1]:car_position[car_num][3],
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
                            # exe隐藏cmd窗口
                            st = subprocess.STARTUPINFO()
                            st.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
                            st.wShowWindow = subprocess.SW_HIDE

                            subprocess.call(chinexe + " " + '"' + chin_path + '"', startupinfo=st)  # 路径中有空格 需要加""
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
        xlsName = 'carSearchResult.xls'
        # excel.save(realtime + '.xls')
        excel.save(xlsName)

        # 车辆检索
        self.sg_setTaskStatus.emit("正在进行车辆检索")
        srcImg_path = os.getcwd() + '\\image_output\\CarImage\\'  # './image_output/CarImage/'#这个可能需要改一下
        templateImg = self.ObjectCar

        # exe隐藏cmd窗口
        st = subprocess.STARTUPINFO()
        st.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
        st.wShowWindow = subprocess.SW_HIDE
        vehicleReIDPath = "vehicleSearch.exe"
        subprocess.call(vehicleReIDPath + " " + '"' + srcImg_path + '"' + " " + '"' + templateImg + '"',startupinfo=st)

        vehicleSearchResult = sio.loadmat('vehicleSearch.mat')
        searchResult = vehicleSearchResult['FeatureMatchResult']
        # rankcarPath = (os.getcwd() + '\\image_output\\rankImage\\')  # 车牌图像路径 'F:\\CompleteVehicleReID\\ImageUI\\rankImage\\'
        rank = np.array(searchResult)

        imgs = os.listdir(srcImg_path)
        carsNum = len(imgs)
        print("CarsNum:",carsNum)
        data = xlrd.open_workbook(xlsName)  # 读了车牌检测的xls
        table = data.sheets()[0]
        t = os.path.getctime(Videoname)
        timeStruct = time.localtime(t)

        year = time.strftime('%Y', timeStruct)
        month = time.strftime('%m', timeStruct)
        day = (time.strftime('%d', timeStruct))
        hour = int(time.strftime('%H', timeStruct))
        minute = int(time.strftime('%M', timeStruct))
        second = int((time.strftime('%S', timeStruct)))
        print(carsNum)
        f = open('rankResult.txt', 'w')

        rankcarPath = os.path.abspath('.') + '\\rankImage\\'
        shutil.rmtree(rankcarPath)
        os.mkdir(rankcarPath)

        for i in range(0, carsNum):
            print('第' + str(i + 1) + '幅')
            rankCar = rank[:, i]
            rankCar = np.int(rankCar)
            row = table.row_values(rankCar)
            row = np.array(row)
            frameNum = row[0]
            carlocation = row[1]
            PlateNum = row[2]

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
            rankInfo = '排名第' + str(i + 1) + '的车辆出现在：' + frameTime
            print(rankInfo + '  ' + '车牌号为：' + PlateNum)
            f.write(rankInfo + ' ' + '车牌号为：' + PlateNum + '\n')
            VC.set(cv2.CAP_PROP_POS_FRAMES, int(frameNum))
            a, rankImg = VC.read()
            # rankImg = self.format_img_size(rankImg, C)
            b1 = carlocation.find(',')
            x1 = int(carlocation[1:b1])
            # print(x1)
            b2 = carlocation.find(',', b1 + 1)
            y1 = int(carlocation[b1 + 2:b2])
            # print(y1)
            b3 = carlocation.find(',', b2 + 1)
            x2 = int(carlocation[b2 + 2:b3])
            # print(x2)
            b4 = carlocation.find(',', b3 + 1)
            y2 = int(carlocation[b3 + 2:b4])
            # print(y2)
            cv2.rectangle(rankImg, (x1, y1), (x2, y2), (0, 255, 0), 3)

            rankcarName = str(i + 1) + '.jpg'
            rankpath = os.path.join(rankcarPath, rankcarName)
            cv2.imwrite(rankpath, rankImg)

        f.close()
        VC.release()

        for x in locals().keys():  # 清除函数中的临时变量
            del locals()[x]
        gc.collect()

        self.sg_finished.emit(self.videoName)



app = qt.QApplication(sys.argv)
tool = Ui_ImageSystem()
sys.exit(app.exec_())
