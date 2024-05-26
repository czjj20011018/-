import os
import sys
import threading

import cv2
import numpy as np
import time

from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QImage
from PyQt5.QtWidgets import *
from template_ui import Ui_MainWindow
import cv2


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def run(self):
        video = "rtsp://admin:admin@192.168.1.224:8554/live"
        cap = cv2.VideoCapture(video)
        while True:
            ret, cv_img = cap.read()
            if ret:
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img)


class MyMainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # 设备背景图片
        palette = QPalette()
        pix = QPixmap('background.jpg')
        pix = pix.scaled(pix.width(), pix.height())
        palette.setBrush(QPalette.Background, QBrush(pix))
        self.setPalette(palette)
        try:
            self.thread = VideoThread(self)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
        except Exception as e:
            print(e)
            pass

        # Button 的颜色修改
        button_color = [self.pushButton, self.pushButton_2, self.pushButton_3, self.pushButton_4]
        for i in range(4):
            button_color[i].setStyleSheet("QPushButton{color:rgb(0,0,0)}"
                                          "QPushButton{background-color:rgb(255,255,255)}"
                                          "QPushButton{border-radius:10px}"
                                          "QPushButton{padding:15px 4px}")

        self.pushButton.clicked.connect(self.Select_template_image)
        self.pushButton_2.clicked.connect(self.Select_image)
        self.pushButton_3.clicked.connect(self.closeEvent)
        self.pushButton_4.clicked.connect(self.capture)
        self.comboBox.currentIndexChanged.connect(self.on_comboBox_activated)

    def update_image(self, qt_img):
        self.label_3.setScaledContents(True)
        self.label_3.setPixmap(QPixmap.fromImage(qt_img))

    def on_comboBox_activated(self):
        global model
        model = self.comboBox.currentText()

    def Select_template_image(self):
        global template_path
        template_path, _ = QFileDialog.getOpenFileName(self, '选择模板', r'./template_img')
        print(template_path)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def Select_image(self):
        global img_path
        img_path, _ = QFileDialog.getOpenFileName(self, '选择图片', r'./test_img')
        self.label.setPixmap(QPixmap(img_path))
        # self.label.setScaledContents(True)  # 自适应大小
        # 开线程识别
        t = threading.Thread(target=self.detect)
        t.setDaemon(True)
        t.start()

    def capture(self):
        # 获取照片
        try:
            # 获取一张照片，并显示
            pixmap = self.label_3.pixmap()
            if pixmap is not None:
                # 保存图片
                global img_path
                img_path = 'test_img/capture.jpg'
                pixmap.save(img_path)
                self.label.setScaledContents(True)
                self.label.setPixmap(pixmap)
                # 开线程识别
                t = threading.Thread(target=self.detect)
                t.setDaemon(True)
                t.start()

        except Exception as e:
            print('相机获取照片错误' + str(e))
            pass

    def detect(self):
        try:
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            cropped_template_bgr = cv2.imread(template_path)
            cropped_template_rgb = cv2.cvtColor(cropped_template_bgr, cv2.COLOR_BGR2RGB)
            cropped_template_rgb = np.array(cropped_template_rgb)
            cropped_template_gray = cv2.cvtColor(cropped_template_rgb, cv2.COLOR_RGB2GRAY)
            height, width = cropped_template_gray.shape
            points_list = self.invariantMatchTemplate(img_rgb, cropped_template_rgb, model, 0.5, 500, [0, 360], 10,
                                                 [100, 150], 10, True, True)
            fig, ax = plt.subplots(1)
            ax.imshow(img_rgb)
            centers_list = []
            for point_info in points_list:
                point = point_info[0]
                angle = point_info[1]
                print("Corresponding angle:", angle)
                scale = point_info[2]
                print("Corresponding scale:", scale)
                print("Centers_Point:", point[0] + width / 2 * scale / 100, point[1] + height / 2 * scale / 100)
                centers_list.append([point, scale])
                plt.scatter(point[0] + (width / 2) * scale / 100, point[1] + (height / 2) * scale / 100, s=20, color="red")
                plt.scatter(point[0], point[1], s=20, color="green")
                rectangle = patches.Rectangle((point[0], point[1]), width * scale / 100, height * scale / 100, color="red",
                                              alpha=0.50, label='Matched box')
                box = patches.Rectangle((point[0], point[1]), width * scale / 100, height * scale / 100, color="green",
                                        alpha=0.50, label='Bounding box')
                transform = mpl.transforms.Affine2D().rotate_deg_around(point[0] + width / 2 * scale / 100,
                                                                        point[1] + height / 2 * scale / 100,
                                                                        angle) + ax.transData
                rectangle.set_transform(transform)
                ax.add_patch(rectangle)
                ax.add_patch(box)
                plt.legend(handles=[rectangle, box])
            # 保存图形
            plt.savefig('template_matching_results.png')
            # plt.show()
            fig2, ax2 = plt.subplots(1)
            ax2.imshow(img_rgb)
            for point_info in centers_list:
                point = point_info[0]
                scale = point_info[1]
                plt.scatter(point[0] + width / 2 * scale / 100, point[1] + height / 2 * scale / 100, s=20, color="red")
            # # 保存图形
            # plt.savefig('template_matching_results1.png')

            # 显示结果图形
            self.label.setPixmap(QPixmap('template_matching_results.png'))

        except Exception as e:
            QMessageBox.about(self, "警告", "未选择模型！")
            print(e)

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def scale_image(self, image, percent, maxwh):
        max_width = maxwh[1]
        max_height = maxwh[0]
        max_percent_width = max_width / image.shape[1] * 100
        max_percent_height = max_height / image.shape[0] * 100
        max_percent = 0
        if max_percent_width < max_percent_height:
            max_percent = max_percent_width
        else:
            max_percent = max_percent_height
        if percent > max_percent:
            percent = max_percent
        width = int(image.shape[1] * percent / 100)
        height = int(image.shape[0] * percent / 100)
        result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return result, percent


    def invariantMatchTemplate(self, rgbimage, rgbtemplate, method, matched_thresh, rgbdiff_thresh, rot_range,
                               rot_interval, scale_range, scale_interval, rm_redundant, minmax):
        """
        rgbimage: RGB图像，用于搜索。
        rgbtemplate: RGB搜索模板。它的大小不能大于源图像，并且必须具有相同的数据类型。
        method: [字符串] 指定比较方法的参数。
        matched_thresh: [浮点数] 设置匹配结果的阈值（0~1）。
        rgbdiff_thresh: [浮点数] 设置模板与源图像之间的平均RGB差异的阈值。
        rot_range: [整数] 旋转角度范围的数组，单位为度。示例：[0,360]。
        rot_interval: [整数] 在旋转角度范围内遍历的间隔，单位为度。
        scale_range: [整数] 缩放百分比范围的数组。示例：[50,200]。
        scale_interval: [整数] 在缩放百分比范围内遍历的间隔。
        rm_redundant: [布尔值] 基于模板的宽度和高度移除冗余匹配结果的选项。
        minmax: [布尔值] 查找具有最小/最大值的点的选项。

        返回: 符合匹配点的列表，格式为[[点的x坐标，点的y坐标]，角度，缩放比]。

        """
        image_maxwh = rgbimage.shape
        height, width, numchannel = rgbtemplate.shape
        all_points = []
        if minmax == False:
            for next_angle in range(rot_range[0], rot_range[1], rot_interval):
                for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                    scaled_template, actual_scale = self.scale_image(rgbtemplate, next_scale, image_maxwh)
                    if next_angle == 0:
                        rotated_template = scaled_template
                    else:
                        rotated_template = self.rotate_image(scaled_template, next_angle)
                    if method == "TM_CCOEFF":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_CCOEFF)
                        satisfied_points = np.where(matched_points >= matched_thresh)
                    elif method == "TM_CCOEFF_NORMED":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_CCOEFF_NORMED)
                        satisfied_points = np.where(matched_points >= matched_thresh)
                    elif method == "TM_CCORR":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_CCORR)
                        satisfied_points = np.where(matched_points >= matched_thresh)
                    elif method == "TM_CCORR_NORMED":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_CCORR_NORMED)
                        satisfied_points = np.where(matched_points >= matched_thresh)
                    elif method == "TM_SQDIFF":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_SQDIFF)
                        satisfied_points = np.where(matched_points <= matched_thresh)
                    elif method == "TM_SQDIFF_NORMED":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_SQDIFF_NORMED)
                        satisfied_points = np.where(matched_points <= matched_thresh)
                    else:
                        raise print("There's no such comparison method for template matching.")
                    for pt in zip(*satisfied_points[::-1]):
                        all_points.append([pt, next_angle, actual_scale])
        else:
            for next_angle in range(rot_range[0], rot_range[1], rot_interval):
                for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                    scaled_template, actual_scale = self.scale_image(rgbtemplate, next_scale, image_maxwh)
                    if next_angle == 0:
                        rotated_template = scaled_template
                    else:
                        rotated_template = self.rotate_image(scaled_template, next_angle)
                    if method == "TM_CCOEFF":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_CCOEFF)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if max_val >= matched_thresh:
                            all_points.append([max_loc, next_angle, actual_scale, max_val])
                    elif method == "TM_CCOEFF_NORMED":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if max_val >= matched_thresh:
                            all_points.append([max_loc, next_angle, actual_scale, max_val])
                    elif method == "TM_CCORR":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_CCORR)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if max_val >= matched_thresh:
                            all_points.append([max_loc, next_angle, actual_scale, max_val])
                    elif method == "TM_CCORR_NORMED":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_CCORR_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if max_val >= matched_thresh:
                            all_points.append([max_loc, next_angle, actual_scale, max_val])
                    elif method == "TM_SQDIFF":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_SQDIFF)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if min_val <= matched_thresh:
                            all_points.append([min_loc, next_angle, actual_scale, min_val])
                    elif method == "TM_SQDIFF_NORMED":
                        matched_points = cv2.matchTemplate(rgbimage, rotated_template, cv2.TM_SQDIFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                        if min_val <= matched_thresh:
                            all_points.append([min_loc, next_angle, actual_scale, min_val])
                    else:
                        raise print("There's no such comparison method for template matching.")
            if method == "TM_CCOEFF":
                all_points = sorted(all_points, key=lambda x: -x[3])
            elif method == "TM_CCOEFF_NORMED":
                all_points = sorted(all_points, key=lambda x: -x[3])
            elif method == "TM_CCORR":
                all_points = sorted(all_points, key=lambda x: -x[3])
            elif method == "TM_CCORR_NORMED":
                all_points = sorted(all_points, key=lambda x: -x[3])
            elif method == "TM_SQDIFF":
                all_points = sorted(all_points, key=lambda x: x[3])
            elif method == "TM_SQDIFF_NORMED":
                all_points = sorted(all_points, key=lambda x: x[3])
        if rm_redundant == True:
            lone_points_list = []
            visited_points_list = []
            for point_info in all_points:
                point = point_info[0]
                scale = point_info[2]
                all_visited_points_not_close = True
                if len(visited_points_list) != 0:
                    for visited_point in visited_points_list:
                        if (abs(visited_point[0] - point[0]) < (width * scale / 100)) and (
                                abs(visited_point[1] - point[1]) < (height * scale / 100)):
                            all_visited_points_not_close = False
                    if all_visited_points_not_close:
                        lone_points_list.append(point_info)
                        visited_points_list.append(point)
                else:
                    lone_points_list.append(point_info)
                    visited_points_list.append(point)
            points_list = lone_points_list
        else:
            points_list = all_points
        return points_list


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
