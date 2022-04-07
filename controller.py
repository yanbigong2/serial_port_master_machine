import platform
import sys

import schedule

if platform.system() == 'Windows':
    pass

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import numpy as np
import sounddevice as sd

from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudio, QCameraInfo

import soundfile as sf
import queue
import cv2, imutils
import datetime

import serial
import serial.tools.list_ports
import time
import os
import re
from threading import Timer

import pandas as pd
import numpy as np

input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)
input_video_deviceInfos = QCameraInfo.availableCameras()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()


class CONTROL_SYSTEM(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.i = 1
        self.logic = None
        self.ui = uic.loadUi('main.ui', self)

        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(5)  # 3 or 4?
        self.CHUNK = 1024
        self.q = queue.Queue(maxsize=self.CHUNK)  # use to buffer audio data

        # set devices

        self.audio_devices_list = []
        self.video_devices_list = []
        for device in input_audio_deviceInfos:
            self.audio_devices_list.append(device.deviceName())
        for device in input_video_deviceInfos:
            self.video_devices_list.append(device.description())

        self.port_list = list(serial.tools.list_ports.comports())
        self.IMU_devices_list = []
        self.IMU_devices = []
        if len(self.port_list) == 0:
            print('no port for use')
        else:
            for i in range(0, len(self.port_list)):
                self.IMU_devices_list.append(self.port_list[i].description)
                self.IMU_devices.append(self.port_list[i].device)

        self.comboBox.addItems(self.IMU_devices_list)
        # print(self.IMU_devices_list)
        # print(self.IMU_devices)
        self.comboBox.currentIndexChanged['QString'].connect(self.update_IMU_now)
        self.comboBox_2.addItems(self.audio_devices_list)
        self.comboBox_2.currentIndexChanged['QString'].connect(self.update_audio_now)
        self.comboBox_2.setCurrentIndex(0)
        self.comboBox_3.addItems(self.video_devices_list)
        self.comboBox_3.currentIndexChanged['QString'].connect(self.update_video_now)
        self.comboBox_3.setCurrentIndex(0)

        # self.widget_4 = QtWidgets.QWidget(self.groupBox_4)
        # self.widget_4.setMinimumSize(QtCore.QSize(450, 190))
        # self.widget_4.setStyleSheet("background-color: rgb(0, 0, 0);")
        # self.widget_4.setObjectName("widget_4")
        # self.gridLayout.addWidget(self.widget_4, 0, 2, 2, 1)
        #
        # self.widget_5 = QtWidgets.QWidget(self.groupBox_5)
        # self.widget_5.setMinimumSize(QtCore.QSize(450, 190))
        # self.widget_5.setStyleSheet("background-color: rgb(0, 0, 0);")
        # self.widget_5.setObjectName("widget_5")
        # self.gridLayout_2.addWidget(self.widget_5, 0, 2, 2, 1)
        self.audio_canvas = MplCanvas(self, width=4, height=2, dpi=100)
        self.ui.gridLayout.addWidget(self.audio_canvas, 0, 2, 2, 1)
        # self.video_canvas = MplCanvas(self, width=4, height=2, dpi=100)
        # self.ui.gridLayout_2.addWidget(self.video_canvas, 0, 2, 2, 1)

        self.audio_reference_plot = None  # what's this?
        self.video_reference_plot = None

        # initialize devices
        self.audio_device = self.audio_devices_list[0]
        self.audio_window_length = 1000
        self.audio_downsample = 1
        self.audio_channels = [1]
        self.audio_interval = 100  # update in short time, msec
        self.audio_samplerate = 44100
        length = int(self.audio_window_length * self.audio_samplerate / (1000 * self.audio_downsample))
        sd.default.samplerate = self.audio_samplerate

        self.audio_plotdata = np.zeros((length, len(self.audio_channels)))
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)  # msec 每这么长时间update一次plot
        # self.timer.timeout.connect(self.audio_update_plot)
        self.timer.timeout.connect(self.swallow_and_cough)
        # self.timer.start()
        self.audio_data = [0]
        self.date = datetime.datetime.now()

        self.video_device = self.video_devices_list[1]

        self.swallow_num = False  # 如果发生咳嗽或者吞咽，10次记录为1
        self.cough_num = False

        # connect the functions

        self.pushButton.clicked.connect(self.start_worker)
        # self.pushButton.clicked.connect(self.start_record_audio)
        # self.pushButton.clicked.connect(self.start_record_video)
        self.pushButton_2.clicked.connect(self.stop_record)
        self.pushButton_3.clicked.connect(self.swallow_func)
        self.pushButton_4.clicked.connect(self.cough_func)

        self.audio_worker = None  # a thread manged by the pool, assigned to a worker
        self.video_worker = None
        self.IMU_worker = None
        self.sc_worker = None
        self.go_on = False  # a flag used to break the loop and release the thread (stop the thread)

        self.swallow_df = pd.DataFrame([0])
        self.cough_df = pd.DataFrame([0])

    def start_worker(self):
        self.go_on = False
        print('Start worker')

        file_path = os.getcwd() + '/' + 'data'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        self.IMU_worker = Worker(self.start_record_IMU)
        self.threadpool.start(self.IMU_worker)

        self.audio_canvas.axes.clear()
        self.audio_worker = Worker(self.start_record_audio)
        self.threadpool.start(self.audio_worker)
        self.audio_reference_plot = None  # what's this?

        self.video_worker = Worker(self.start_record_video)
        self.threadpool.start(self.video_worker)
        self.video_reference_plot = None

        self.date = datetime.datetime.now()
        self.timer.start()
        self.swallow_num = False  # 如果发生咳嗽或者吞咽，10次记录为1
        self.cough_num = False
        # self.sc_worker = Worker(self.swallow_and_cough)
        # self.threadpool.start(self.sc_worker)

    def stop_record(self):
        self.logic = 0
        self.go_on = True

    def start_record_audio(self):
        print('Hello Audio')
        try:
            QtWidgets.QApplication.processEvents()

            date = datetime.datetime.now()
            f_path = ('data/audio_%s_%s_%s_%s_%s_%s.wav' %
                      (date.year, date.month, date.day, date.hour, date.minute, date.second))
            file = sf.SoundFile(f_path, mode='w', samplerate=self.audio_samplerate, channels=1)

            def audio_callback(indata, frames, time, status):
                self.q.put(indata[::self.audio_downsample, [0]])

            stream = sd.InputStream(device=self.audio_device, channels=max(self.audio_channels),
                                    samplerate=self.audio_samplerate, callback=audio_callback)

            with file:
                with stream:
                    while True:
                        file.write(self.q.get())
                        QtWidgets.QApplication.processEvents()
                        if self.go_on:
                            break
                stream.stop()
            file.close()
        except Exception as e:
            print("Error", e)
            pass

    def audio_update_plot(self):
        try:
            # print(self.threadpool.activeThreadCount())
            # print('ACTIVE THREADS:', self.threadpool.activeThreadCount(), end=" \r")
            # print('Happy')
            while self.go_on is False:
                QtWidgets.QApplication.processEvents()
                try:
                    self.audio_data = self.q.get_nowait()
                except Exception as e:
                    # print('Error, ', e) # queue空着的时候不打印很正常，直接break就行了
                    break
                shift = len(self.audio_data)
                self.audio_plotdata = np.roll(self.audio_plotdata, -shift, axis=0)
                self.audio_plotdata = self.audio_data
                self.audio_ydata = self.audio_plotdata[:]
                self.audio_canvas.axes.set_facecolor((0, 0, 0))

                if self.audio_reference_plot is None:
                    plot_refs = self.audio_canvas.axes.plot(self.audio_ydata, color=(0, 1, 0.29))
                    self.audio_reference_plot = plot_refs[0]
                else:
                    self.audio_reference_plot.set_ydata(self.audio_ydata)

            self.audio_canvas.axes.yaxis.grid(True, linestyle='--')
            start, end = self.audio_canvas.axes.get_ylim()
            self.audio_canvas.axes.yaxis.set_ticks(np.arange(start, end, 0.5))
            self.audio_canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            self.audio_canvas.axes.set_ylim(ymin=-1, ymax=1)

            self.audio_canvas.draw()
        except Exception as e:
            print('Error, ', e)
            pass

    def update_audio_now(self, value):
        self.audio_device = self.audio_devices_list.index(value)
        print(value)
        print(self.audio_devices_list.index(value))

    def start_record_video(self):
        # video
        print('Hellow Video')
        self.logic = 1
        cap = cv2.VideoCapture(self.video_device, cv2.CAP_DSHOW)
        date = datetime.datetime.now()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('data/video_%s_%s_%s_%s_%s_%s.avi' %
                              (date.year, date.month, date.day, date.hour, date.minute, date.second),
                              fourcc, 30, (width, height), True)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:  # ret==True
                # self.display image(frame,1)
                # cv2.imshow('frame', frame)
                resized = imutils.resize(frame.copy(), height=450)
                self.setImage(self.label_11, resized)
                out.write(frame)
                cv2.waitKey()
                if self.logic == 0:
                    break
        cap.release()
        out.release()

    def setImage(self, box, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888
        h, w = img.shape[:2]
        img = QImage(img, w, h, qformat)
        img = img.rgbSwapped()
        box.setPixmap(QPixmap.fromImage(img))
        box.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    def update_video_now(self, value):
        self.video_device = self.video_devices_list.index(value)
        print(value)
        print(self.video_devices_list.index(value))

    def start_record_IMU(self):
        print('Hello IMU')
        ser = serial.Serial()
        for i in range(len(self.IMU_devices_list)):
            if 'CH340' in self.IMU_devices_list[i]:
                ser = serial.Serial(self.IMU_devices[i], timeout=1)
        if ser.isOpen():
            # print("open IMU success")
            # 向端口些数据 字符串必须译码
            ser.write("AT+INIT\r\n".encode())
            # print('Initialize')
            ser.write("AT+PRATE=100\r\n".encode())  # 采样率，每x毫秒采样一次
            # print('Set sample rate')
            line = ser.readline()

            date = datetime.datetime.now()
            df = pd.DataFrame(columns=['X', 'Y', 'Z'])

            # if b'OK' in line:
            #     print('yes')
            # cur_time = time.time()
            # print(cur_time)
            try:
                while True:
                    line = ser.readline()
                    if self.go_on:
                        break
                    if line:
                        # print(time.asctime(time.localtime(time.time())))
                        str_line = line.decode("ISO-8859-1")[:-2]
                        # print(str_line)
                        # str_line = 'AccX:-264 mG,AccY:929 mG,AccZ:168 mG'
                        result_x = re.search('X:(-?\d*) mG', str_line)
                        result_y = re.search('Y:(-?\d*) mG', str_line)
                        result_z = re.search('Z:(-?\d*) mG', str_line)
                        if result_x is not None and result_y is not None and result_z is not None:
                            # print(result_x.group(1), result_y.group(1), result_z.group(1))
                            # print('X: %s, Y: %s, Z: %s' % (result_x.group(1), result_y.group(1), result_z.group(1)))
                            self.lineEdit_8.setText(str_line)
                            # print(type(result_z.group(1)))
                            df.loc[len(df.index)] = (
                                [int(result_x.group(1)), int(result_y.group(1)), int(result_z.group(1))])
                        else:
                            print('Not find')
                    else:
                        print('bug???')
                print('end record')
                # print(df)
                df.to_csv('data/IMU_%s_%s_%s_%s_%s_%s.csv' %
                          (date.year, date.month, date.day, date.hour, date.minute, date.second))
            except Exception as e:
                print(e)

        else:
            print("open failed")

    def update_IMU_now(self, value):
        pass

    def swallow_and_cough(self):
        # print('Hello Swallow and Cough')
        # date = datetime.datetime.now()
        #
        #
        #
        # while True:
        #     print('enter loop')
        #     time.sleep(0.1)
        print(self.i)
        self.i += 1
        if self.swallow_num:
            self.swallow_df.loc[len(self.swallow_df.index)] = [1]
            self.swallow_num = False
        else:
            self.swallow_df.loc[len(self.swallow_df.index)] = [0]
        if self.cough_num:
            self.cough_df.loc[len(self.cough_df.index)] = [1]
            self.cough_num = False
        else:
            self.cough_df.loc[len(self.cough_df.index)] = [0]

        # if self.go_on:
        #     break
        # print(cough_df)
        # print(swallow_df)
        if self.go_on:
            # print(self.swallow_df)
            # print(self.cough_df)
            self.cough_df.to_csv('data/cough_%s_%s_%s_%s_%s_%s.csv' %
                                 (self.date.year, self.date.month, self.date.day, self.date.hour, self.date.minute,
                                  self.date.second))
            self.swallow_df.to_csv('data/swallow_%s_%s_%s_%s_%s_%s.csv' %
                                   (self.date.year, self.date.month, self.date.day, self.date.hour, self.date.minute,
                                    self.date.second))
            self.timer.stop()

    def insert_data(self):
        print('bug here?')
        pass

    def swallow_func(self):
        self.swallow_num = True

    def cough_func(self):
        self.cough_num = True


class Worker(QtCore.QRunnable):

    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = CONTROL_SYSTEM()
    mainWindow.show()
    sys.exit(app.exec_())
