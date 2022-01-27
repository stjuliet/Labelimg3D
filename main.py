import os
import numpy as np
import cv2 as cv
import sys
from interface import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow
from YoloDetect import YoloDetect
from copy import deepcopy
import re
import xml.etree.ElementTree as ET
from tools import ReadCalibParam, ParamToMatrix, cal_3dbbox, save3dbbox_result, dashLine


dict_map_order_str = {'car': 1, 'truck': 2, 'bus': 3}
classes = ["Car", "Truck", "Bus"]


# support key-point annotation
class MyLabel(QLabel):
    def __init__(self, parent=None):
        super(MyLabel, self).__init__((parent))
        self.points = []
        self.paint_flag = False
        self.scaleX = 0.0
        self.scaleY = 0.0
        self.q_points = []  # QPoint for show

    def mousePressEvent(self, event):
        if self.scaleX != 0 and self.scaleY != 0:
            if event.button() == Qt.LeftButton:
                self.paint_flag = True
                pt = event.pos()  # QPoint
                pt_x = event.x() * self.scaleX
                pt_y = event.y() * self.scaleY
                self.points.append([pt_x, pt_y])
                self.q_points.append(pt)
            elif event.button() == Qt.RightButton and self.points:
                self.points.pop(-1)
                self.q_points.pop(-1)
            if self.paint_flag:
                self.update()

    def mouseReleaseEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        pass

    def paintEvent(self, event):
        super().paintEvent(event)  # parent for show background
        if self.scaleX != 0 and self.scaleY != 0:
            painter_brush = QPainter()
            painter_brush.begin(self)
            painter_brush.setPen(QPen(Qt.green, 4))
            if self.paint_flag:
                for i in range(len(self.q_points)):
                    painter_brush.drawPoints(self.q_points[i])
                    point_coordination = "(" + str(int(self.points[i][0])) + "," + str(int(self.points[i][1])) + ")"
                    painter_brush.drawText(self.q_points[i], point_coordination)
            else:
                return


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self)
        self.center()  # center

        self.frame = None
        self.pushButton_ClearAnnotations.setEnabled(False)

        self.perspective = "right"
        if self.perspective == "right":
            self.radioButton_BasePointRight.setChecked(True)
        else:
            self.radioButton_BasePointLeft.setChecked(True)
        self.l = self.doubleSpinBox_Bbox3D_Length.value() * 1000.0
        self.w = self.doubleSpinBox_Bbox3D_Width.value() * 1000.0
        self.h = self.doubleSpinBox_Bbox3D_Height.value() * 1000.0
        self.key_point_nums = 4

        self.veh_box = []
        self.key_points = []

        self.all_veh_2dbbox = []  # save annotations
        self.all_3dbbox_2dvertex = []
        self.all_vehicle_type = []
        self.all_vehicle_size = []
        self.all_perspective = []
        self.all_base_point = []
        self.all_3dbbox_3dvertex =[]
        self.all_vehicle_location = []
        self.all_veh_conf = []
        self.all_key_points = []

        # load redefined Mylabel
        self.label_ImageDisplay = MyLabel(self.groupBox_ImageDisplay)
        self.label_ImageDisplay.setGeometry(QtCore.QRect(10, 20, 971, 701))

        self.label_ImageDisplay.setScaledContents(True)
        self.label_ImageDisplay.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")  # style sheet

    def closeEvent(self, event):
        reply = QMessageBox.information(self, "info", "Are you sure to exit?",
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit()
        else:
            event.ignore()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def show_img_in_label(self, label_name, img):
        """ show img in label """
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        qimage = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], rgb_img.shape[1] * 3, QImage.Format_RGB888)
        label_name.setPixmap(QPixmap.fromImage(qimage))

    def draw_3dbox(self, frame):
        """ calculate and draw 3d box """
        self.list_3dbbox_2dvertex, self.list_3dbbox_3dvertex, self.centroid_2d = cal_3dbbox(self.perspective, self.m_trans, self.veh_base_point, self.veh_turple_vp, self.l, self.w, self.h)
        drawbox_img = self.paint(frame.copy(), self.list_3dbbox_2dvertex, self.centroid_2d)
        if self.label_ImageDisplay.points:
            self.key_points = np.array(self.label_ImageDisplay.points).flatten().tolist()  # 每次绘制3d box时，保存拉平的关键点对
            self.paint_key_points(drawbox_img, self.key_points)
        return drawbox_img

    def paint(self, imgcopy, list_vertex, centroid_2d):
        """ paint 3d box """
        # width
        cv.line(imgcopy, list_vertex[0], (list_vertex[1][0], list_vertex[1][1]), (0, 0, 255), 2)
        # cv.line(imgcopy, list_vertex[2], list_vertex[3], (0, 0, 255), 2)
        dashLine(imgcopy, list_vertex[2], list_vertex[3], (0, 0, 255), 2, 5)
        cv.line(imgcopy, list_vertex[4], list_vertex[5], (0, 0, 255), 2)
        cv.line(imgcopy, list_vertex[6], list_vertex[7], (0, 0, 255), 2)

        # length
        cv.line(imgcopy, list_vertex[0], list_vertex[3], (255, 0, 0), 2)
        # cv.line(imgcopy, list_vertex[1], list_vertex[2], (255, 0, 0), 2)
        dashLine(imgcopy, (list_vertex[1][0], list_vertex[1][1]), list_vertex[2], (255, 0, 0), 2, 5)
        cv.line(imgcopy, list_vertex[4], list_vertex[7], (255, 0, 0), 2)
        cv.line(imgcopy, list_vertex[5], list_vertex[6], (255, 0, 0), 2)

        # height
        cv.line(imgcopy, list_vertex[0], list_vertex[4], (0, 255, 0), 2)
        cv.line(imgcopy, (list_vertex[1][0], list_vertex[1][1]), list_vertex[5], (0, 255, 0), 2)
        cv.line(imgcopy, list_vertex[3], list_vertex[7], (0, 255, 0), 2)
        # cv.line(imgcopy, list_vertex[2], list_vertex[6], (0, 255, 0), 2)
        dashLine(imgcopy, list_vertex[2], list_vertex[6], (0, 255, 0), 2, 5)

        # centroid
        cv.circle(imgcopy, centroid_2d, 5, (0, 255, 255), 3)

        self.show_img_in_label(self.label_ImageDisplay, imgcopy)

        return imgcopy

    def paint_key_points(self, imgcopy, list_key_point):
        """ paint key points """
        for i in range(len(list_key_point)//2):
            pt = (int(list_key_point[2*i]), int(list_key_point[2*i+1]))
            cv.circle(imgcopy, pt, 5, (0, 255, 0), 3)

        self.show_img_in_label(self.label_ImageDisplay, imgcopy)
        return imgcopy

    def image_label_display(self, object):
        """ object detection callback  """
        # object[0]: detected frame (QPixmap)
        # object[1]: listbox
        # object[2]: listtype
        self.label_ImageDisplay.setPixmap(object[0])
        self.textEdit_ObjNums.setText(str(len(object[1])))
        self.obj_num = int(self.textEdit_ObjNums.toPlainText())
        self.list_box = object[1]
        self.list_type = object[2]
        self.list_conf = object[3]

    def choose_img_folder(self):
        """ choose annotation folder """
        self.list_file_path = []
        self.str_folder_path = QFileDialog.getExistingDirectory(self, "Choose Folder", os.getcwd())
        # support chinese path
        self.str_folder_path = self.str_folder_path.encode("utf-8").decode("utf-8")

        parent = os.path.abspath(os.path.join(self.str_folder_path, os.pardir))
        # yolov4 model files and thread initialize
        self.classesFile = os.path.join(parent, "model_us_yolov4/obj.names")
        self.modelConfiguration = os.path.join(parent , "model_us_yolov4/yolov4.cfg")
        self.modelWeights = os.path.join(parent, "model_us_yolov4/yolov4_best.weights")
        if os.path.exists(self.classesFile) and os.path.exists(self.modelConfiguration) and os.path.exists(self.modelWeights):
            self.thread = YoloDetect(self.classesFile, self.modelConfiguration, self.modelWeights)
        else:
            return
        try:
            all_file_names = os.listdir(self.str_folder_path)  # get all file names in the directory
            for single_filename in all_file_names:
                if single_filename == 'calib':
                    self.calib_file_path = os.path.join(self.str_folder_path, "calib", os.listdir(os.path.join(self.str_folder_path, "calib"))[0])
                elif os.path.splitext(single_filename)[1] == ".bmp" or ".jpg" or "jpeg" or ".png":
                    self.list_file_path.append(os.path.join(self.str_folder_path, single_filename))  # get all suitable files
            # show in listview
            listview_model = QStringListModel()
            listview_model.setStringList(self.list_file_path)
            self.listView_FileList.setModel(listview_model)
        except Exception as e:  # if not choose file dir
            return

    def listview_doubleclick_slot(self, QModelIndex):
        """ double-click to do object detection, and load calib information"""

        self.all_veh_2dbbox.clear()
        self.all_3dbbox_2dvertex.clear()
        self.all_vehicle_type.clear()
        self.all_vehicle_size.clear()
        self.all_perspective.clear()
        self.all_base_point.clear()
        self.all_3dbbox_3dvertex.clear()
        self.all_vehicle_location.clear()
        self.all_veh_conf.clear()
        self.all_key_points.clear()
        self.select_file = self.list_file_path[QModelIndex.row()]
        self.spinBox_CurAnnNum.setValue(-1)

        if self.select_file.split('.')[-1] == "jpg" or self.select_file.split('.')[-1] == "png":
            # support chinese path
            buf_data = np.fromfile(self.select_file, dtype=np.uint8)
            self.frame = cv.imdecode(buf_data, 1)  # color
            self.frame_copy = deepcopy(self.frame)

            # calculate scale_x, scale_y of label_display
            self.label_ImageDisplay.scaleX = float(self.frame.shape[1]) / float(self.label_ImageDisplay.width())
            self.label_ImageDisplay.scaleY = float(self.frame.shape[0]) / float(self.label_ImageDisplay.height())
        else:
            return

        # load calib parameters
        self.focal, self.fi, self.theta, self.cam_height, self.turple_vp, self.vpline = \
            ReadCalibParam(self.calib_file_path)
        # self.theta = - self.theta
        self.m_trans = ParamToMatrix(self.focal, self.fi, self.theta, self.cam_height, self.frame.shape[1]/2, self.frame.shape[0]/2)
        self.veh_turple_vp = self.turple_vp

        # load annotation files
        # if not exists, load img to detection
        self.select_file_xml = self.select_file.split('.')[0] + '.xml'
        if os.path.exists(self.select_file_xml):
            tree = ET.parse(self.select_file_xml)
            root = tree.getroot()
            # box_nums
            box_nums = len(tree.findall("object"))
            tp_veh_key_point_data = np.zeros((box_nums, 2*self.key_point_nums))
            # box
            for idx, obj in enumerate(root.iter('object')):
                # 1、2d box [left, top, width, height]
                bbox2d_data = obj.find('bbox2d').text.split()
                bbox2d_data = [int(box) for box in bbox2d_data]
                self.all_veh_2dbbox.append(bbox2d_data)
                # 2、vehicle type (0，1，2)
                veh_type_data = obj.find('type').text
                if veh_type_data not in classes:
                    continue
                veh_cls_id = classes.index(veh_type_data)
                self.all_vehicle_type.append(veh_type_data)
                # 3、centroid 2d (int)
                veh_centre_data = obj.find('veh_loc_2d').text.split()
                veh_centre_data = [int(loc) for loc in veh_centre_data]
                self.all_vehicle_location.append(veh_centre_data)
                # 4、vertex 2d (int)
                veh_vertex_data = []  # for each box (8 vertex)
                box_2dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex2d').text)
                for x in box_2dvertex:
                    veh_vertex_data += [int(item) for item in x.split(", ")]  # [x1,y1,x2,y2,...,x8,y8]
                self.all_3dbbox_2dvertex.append(veh_vertex_data)
                # vertex 3d (float)
                veh_3dvertex_data = []  # for each box (8 vertex)
                box_3dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex3d').text)
                for x in box_3dvertex:
                    veh_3dvertex_data += [float(item) for item in x.split(", ")]  # [x1,y1,x2,y2,...,x8,y8]
                self.all_3dbbox_3dvertex.append(veh_3dvertex_data)

                # 5、vehicle size (float, m)
                veh_size_data = obj.find('veh_size').text.split()
                veh_size_data = [float(size) for size in veh_size_data]
                self.all_vehicle_size.append(veh_size_data)
                # 6、view (left, right)
                veh_view_data = obj.find('perspective').text
                self.all_perspective.append(veh_view_data)
                # 7、vehicle base point (int)
                veh_base_point_data = obj.find('base_point').text.split()
                veh_base_point_data = [int(base_point) for base_point in veh_base_point_data]
                self.all_base_point.append(veh_base_point_data)

                # veh_vertex_data -> (tuple)
                tp_veh_vertex_data = []
                for i in range(0, len(veh_vertex_data)-1, 2):
                    tp_veh_vertex_data.append((veh_vertex_data[i], veh_vertex_data[i+1]))
                self.all_3dbbox_2dvertex[idx] = tp_veh_vertex_data

                tp_veh_3dvertex_data = []
                for i in range(0, len(veh_3dvertex_data)-1, 2):
                    tp_veh_3dvertex_data.append((veh_3dvertex_data[i], veh_3dvertex_data[i+1]))
                self.all_3dbbox_3dvertex[idx] = tp_veh_3dvertex_data

                # veh_centre_data -> (tuple)
                tp_veh_centre_data = (int(veh_centre_data[0]), int(veh_centre_data[1]))

                # 8、key_point
                if obj.find('key_points') is not None:
                    veh_key_point_data = obj.find('key_points')
                    if veh_key_point_data:
                        veh_key_point_data = veh_key_point_data.text.split()
                        veh_key_point_data = [int(float(key_point)) for key_point in veh_key_point_data]
                        tp_veh_key_point_data[idx] = veh_key_point_data

                    # show 2d keypoint
                    if np.array(veh_key_point_data).all():
                        self.frame = self.paint_key_points(self.frame, veh_key_point_data)

                # draw 2D box
                left, top, right, bottom = int(bbox2d_data[0]), int(bbox2d_data[1]), int(bbox2d_data[0]) + int(bbox2d_data[2]), int(bbox2d_data[1]) + int(bbox2d_data[3])
                cv.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 3)
                label = '%s:%.2f-%s' % (veh_type_data.lower(), 1.00, str(idx + 1))

                # Display the label at the top of the bounding box
                # display the idx of each object
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(top, labelSize[1])
                cv.rectangle(self.frame, (left, top - round(1.5 * labelSize[1])),
                             (left + round(1.5 * labelSize[0]), top + baseLine),
                             (255, 255, 255), cv.FILLED)
                cv.putText(self.frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

                # draw 3D box
                drawbox_img = self.paint(self.frame, tp_veh_vertex_data, tp_veh_centre_data)

            self.all_key_points = tp_veh_key_point_data.tolist()
            self.show_img_in_label(self.label_ImageDisplay, drawbox_img)

        else:
            # object detection
            self.thread.Init(self.frame)
            self.thread.send_detect_result.connect(self.image_label_display)
            self.thread.start()

    def spin_cur_anno_order(self):
        """ choose annotation object """
        # if annotation file exists, revise can be done.
        if self.frame is not None:
            if os.path.exists(self.select_file_xml):
                self.obj_num = len(self.all_vehicle_size)
                if (self.spinBox_CurAnnNum.value() + 1) <= self.obj_num and self.spinBox_CurAnnNum.value() >= 0:
                    self.spinBox_CurAnnNum.setMaximum(self.obj_num - 1)  # max obj num
                    self.veh_box = self.all_veh_2dbbox[self.spinBox_CurAnnNum.value()]
                    self.veh_type = self.all_vehicle_type[self.spinBox_CurAnnNum.value()].lower()
                    self.comboBox_CurAnnType.setCurrentIndex(dict_map_order_str[self.veh_type])

                    self.perspective = self.all_perspective[self.spinBox_CurAnnNum.value()]
                    self.base_point = self.all_base_point[self.spinBox_CurAnnNum.value()]
                    self.veh_base_point = self.base_point

                    self.l = self.all_vehicle_size[self.spinBox_CurAnnNum.value()][0] * 1000
                    self.w = self.all_vehicle_size[self.spinBox_CurAnnNum.value()][1] * 1000
                    self.h = self.all_vehicle_size[self.spinBox_CurAnnNum.value()][2] * 1000

                    self.drawbox_img = self.draw_3dbox(self.frame)

            else:  # make annotation from zero
                try:
                    if (self.spinBox_CurAnnNum.value() + 1) <= self.obj_num and self.spinBox_CurAnnNum.value() >= 0:
                        self.spinBox_CurAnnNum.setMaximum(self.obj_num - 1)  # max obj num
                        self.veh_box = self.list_box[self.spinBox_CurAnnNum.value()]
                        self.veh_type = self.list_type[self.spinBox_CurAnnNum.value()]
                        self.veh_conf = self.list_conf[self.spinBox_CurAnnNum.value()]
                        self.comboBox_CurAnnType.setCurrentIndex(dict_map_order_str[self.veh_type])
                        # reset slider
                        self.horizontalSlider_BasePointAdj_LR.setValue(0)
                        self.verticalSlider_BasePointAdj_UD.setValue(0)
                        self.horizontalSlider_VPAdj_LR.setValue(0)
                        if self.perspective == 'left':
                            self.base_point = (self.veh_box[0], self.veh_box[1] + self.veh_box[3])
                            self.veh_base_point = self.base_point
                        elif self.perspective == 'right':  # right
                            self.base_point = (self.veh_box[0] + self.veh_box[2], self.veh_box[1] + self.veh_box[3])
                            self.veh_base_point = self.base_point

                        # key-point mode
                        if self.actionkeypoint_only.isChecked():
                            if self.label_ImageDisplay.points:
                                self.drawbox_img = self.draw_3dbox(self.frame)
                            else:
                                QMessageBox.information(self, "Information", "Please make key points annotation first! ",
                                                        QMessageBox.Yes | QMessageBox.No)
                                self.spinBox_CurAnnNum.setValue(-1)
                        else:
                            self.drawbox_img = self.draw_3dbox(self.frame)
                except:
                    QMessageBox.information(self, "Information", "Please choose one vehicle! ",
                                            QMessageBox.Yes | QMessageBox.No)

    def radio_bp_left(self):
        """ choose base point: left bottom """
        try:
            self.perspective = "left"
            self.base_point = (self.veh_box[0], self.veh_box[1] + self.veh_box[3])
            self.veh_base_point = self.base_point
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def radio_bp_right(self):
        """ choose base point: right bottom """
        try:
            self.perspective = "right"
            self.base_point = (self.veh_box[0] + self.veh_box[2], self.veh_box[1] + self.veh_box[3])
            self.veh_base_point = self.base_point
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_bp_adjust_lr(self):
        """ adjust base point: left right """
        try:
            self.veh_base_point = (self.base_point[0] + self.horizontalSlider_BasePointAdj_LR.value(), self.veh_base_point[1])
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_bp_adjust_ud(self):
        """ choose base point: top bottom """
        try:
            self.veh_base_point = (self.veh_base_point[0], self.base_point[1] + self.verticalSlider_BasePointAdj_UD.value())
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_vp_adjust_lr(self):
        """ move vp along horizon line """
        try:
            self.veh_vpx = self.turple_vp[0] + self.horizontalSlider_VPAdj_LR.value()
            self.veh_vpy = self.vpline[0] * (self.veh_vpx - self.vpline[1]) + self.vpline[2]
            self.veh_turple_vp = (self.veh_vpx, self.veh_vpy)
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_vp_adjust_ud(self):
        pass

    def save_temp_annotation_results(self):
        """ save single obj annotation """
        # if annotation file exists, revise can be done.
        if self.frame is not None:
            if os.path.exists(self.select_file_xml):
                self.frame = self.drawbox_img
                if self.actionkeypoint_only.isChecked():
                    self.all_key_points[self.spinBox_CurAnnNum.value()] = self.key_points
                    self.label_ImageDisplay.points.clear()
                    self.label_ImageDisplay.q_points.clear()
                else:
                    self.all_3dbbox_2dvertex[self.spinBox_CurAnnNum.value()] = self.list_3dbbox_2dvertex
                    self.all_vehicle_type[self.spinBox_CurAnnNum.value()] = self.comboBox_CurAnnType.currentText()
                    self.all_vehicle_size[self.spinBox_CurAnnNum.value()] = [self.doubleSpinBox_Bbox3D_Length.value(), self.doubleSpinBox_Bbox3D_Width.value(),self.doubleSpinBox_Bbox3D_Height.value()]
                    self.all_perspective[self.spinBox_CurAnnNum.value()] = self.perspective
                    self.all_base_point[self.spinBox_CurAnnNum.value()] = [self.list_3dbbox_2dvertex[1][0], self.list_3dbbox_2dvertex[1][1]]  # 基准点p1
                    self.all_3dbbox_3dvertex[self.spinBox_CurAnnNum.value()] = self.list_3dbbox_3dvertex
                    self.all_vehicle_location[self.spinBox_CurAnnNum.value()] = [self.centroid_2d[0], self.centroid_2d[1]]
            else:
                # if not clear, not duplicate, add can be done.
                # if clear, all_list empty, save can not be done.
                if self.veh_box:
                    try:  # find duplicate
                        index = self.all_veh_2dbbox.index(self.veh_box)
                        QMessageBox.information(self, "提示", "repeated element!", QMessageBox.Yes)
                    except:  # not duplicate
                        self.frame = self.drawbox_img
                        self.all_veh_2dbbox.append(self.veh_box)
                        self.all_3dbbox_2dvertex.append(self.list_3dbbox_2dvertex)
                        self.all_vehicle_type.append(self.comboBox_CurAnnType.currentText())
                        self.all_vehicle_size.append([self.doubleSpinBox_Bbox3D_Length.value(), self.doubleSpinBox_Bbox3D_Width.value(), self.doubleSpinBox_Bbox3D_Height.value()])
                        self.all_perspective.append(self.perspective)
                        self.all_base_point.append(self.list_3dbbox_2dvertex[1])  # base point p1
                        self.all_3dbbox_3dvertex.append(self.list_3dbbox_3dvertex)
                        self.all_vehicle_location.append(self.centroid_2d)
                        self.all_veh_conf.append(self.veh_conf)
                        self.all_key_points.append(self.key_points)
                        self.label_ImageDisplay.points.clear()
                        self.label_ImageDisplay.q_points.clear()
                        self.pushButton_ClearAnnotations.setEnabled(True)
                else:
                    return

    def save_annotation_results(self):
        """ save single img annotation """
        if self.frame is not None:
            if self.all_3dbbox_2dvertex:
                # save annotation img
                cv.imwrite(self.select_file[0:len(self.select_file)-4] + "_drawbbox_result.bmp", self.frame)
                xml_path = self.select_file[0:len(self.select_file)-4] + ".xml"
                save3dbbox_result(xml_path, self.select_file, self.calib_file_path, self.frame, self.all_veh_2dbbox, self.all_vehicle_type, self.all_3dbbox_2dvertex,
                self.all_vehicle_size, self.all_perspective, self.all_base_point, self.all_3dbbox_3dvertex, self.all_vehicle_location, self.all_key_points, self.actionkeypoint_only.isChecked())
            else:
                return

    def clear_single_annotation(self):
        """
        1\ 未保存时清空
        2\ 保存后清空
        :return:
        """
        frame_copy_ = self.frame_copy.copy()
        if self.frame is not None:
            self.pushButton_ClearAnnotations.setEnabled(False)
            # if not save, but clear, information still can be saved, so parameters should be cleared.
            # if save, and clear, pop the last information without save.
            if len(self.all_vehicle_size) > 0:
                self.veh_box = []  # fix
                self.all_veh_2dbbox.pop(-1)  # fix
                self.all_3dbbox_2dvertex.pop(-1)
                self.all_vehicle_type.pop(-1)  # fix
                self.all_vehicle_size.pop(-1)
                self.all_perspective.pop(-1)
                self.all_base_point.pop(-1)  # base point p1
                self.all_3dbbox_3dvertex.pop(-1)
                self.all_vehicle_location.pop(-1)
                self.all_veh_conf.pop(-1)  # fix
                self.all_key_points.pop(-1)
                for i in range(len(self.list_box)):
                    # draw 2D box
                    left, top, right, bottom = int(self.list_box[i][0]), int(self.list_box[i][1]), int(
                        self.list_box[i][0]) + int(
                        self.list_box[i][2]), int(self.list_box[i][1]) + int(self.list_box[i][3])
                    cv.rectangle(frame_copy_, (left, top), (right, bottom), (0, 0, 255), 3)
                    label = '%s:%.2f' % (self.list_type[i].lower(), self.list_conf[i])

                    # Display the label at the top of the bounding box
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    top = max(top, labelSize[1])
                    cv.rectangle(frame_copy_, (left, top - round(1.5 * labelSize[1])),
                                 (left + round(1.5 * labelSize[0]), top + baseLine),
                                 (255, 255, 255), cv.FILLED)
                    cv.putText(frame_copy_, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
                for i in range(len(self.all_vehicle_size)):  # list after clear
                    imgcopy = self.paint(frame_copy_, self.all_3dbbox_2dvertex[i], (self.all_vehicle_location[i][0], self.all_vehicle_location[i][1]))
                self.frame = frame_copy_
                self.spinBox_CurAnnNum.setValue(self.spinBox_CurAnnNum.value() - 1)
                self.show_img_in_label(self.label_ImageDisplay, frame_copy_)
            else:
                for i in range(len(self.list_box)):
                    # draw 2D box
                    left, top, right, bottom = int(self.list_box[i][0]), int(self.list_box[i][1]), int(
                        self.list_box[i][0]) + int(
                        self.list_box[i][2]), int(self.list_box[i][1]) + int(self.list_box[i][3])
                    cv.rectangle(self.frame_copy, (left, top), (right, bottom), (0, 0, 255), 3)
                    label = '%s:%.2f' % (self.list_type[i].lower(), self.list_conf[i])

                    # Display the label at the top of the bounding box
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    top = max(top, labelSize[1])
                    cv.rectangle(self.frame_copy, (left, top - round(1.5 * labelSize[1])),
                                 (left + round(1.5 * labelSize[0]), top + baseLine),
                                 (255, 255, 255), cv.FILLED)
                    cv.putText(self.frame_copy, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
                self.frame = self.frame_copy
                self.show_img_in_label(self.label_ImageDisplay, self.frame_copy)

    def combo_cur_anno_type(self):
        pass

    def spind_3dbbox_length(self):
        """ adjust vehicle length """
        try:
            self.l = self.doubleSpinBox_Bbox3D_Length.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def spind_3dbbox_width(self):
        """ adjust vehicle width """
        try:
            self.w = self.doubleSpinBox_Bbox3D_Width.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def spind_3dbbox_height(self):
        """ adjust vehicle height """
        try:
            self.h = self.doubleSpinBox_Bbox3D_Height.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())
