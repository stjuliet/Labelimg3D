import os
import numpy as np
import cv2 as cv
import sys
from interface import *
from dialog_vehicle_size import Ui_Dialog as dialog_vehsize
from sub_dialogs.bbox2d_anno import Ui_Dialog as dialog_bbox2d
# from sub_dialogs.window_pretrain_model_3d import Main_Pretrain
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow
from YoloDetect import YoloDetect
from copy import deepcopy
import re
import xml.etree.ElementTree as ET
from tools import ReadCalibParam, ParamToMatrix, cal_3dbbox, cal_3dbbox_dairv2x, save3dbbox_result, dashLine


dict_map_order_str = {'Car': 1, 'Truck': 2, 'Bus': 3, 'Vehicle': 4, 'Non-motor': 5, 'Pedestrian': 6}
classes = ["Car", "Truck", "Bus", "Vehicle", "Non-motor", "Pedestrian"]
veh_size_dict = {"Car": [4.5, 1.8, 1.5],
                 "Truck": [6.0, 2.2, 2.4],
                 "Bus": [10.0, 2.5, 2.6],
                 "Vehicle": [4.5, 1.8, 1.5],
                 "Non-motor": [1.8, 0.8, 1.5],
                 "Pedestrian": [0.6, 0.7, 1.5]}


# support key-point annotation, and wheel scale
class MyLabel(QLabel):
    SCALE_MIN_VALUE = 0.1
    SCALE_MAX_VALUE = 10.0

    def __init__(self, parent=None, window_width=1920, window_height=1080):
        super(MyLabel, self).__init__((parent))
        # for resize screen
        self.window_width = window_width
        self.window_height = window_height
        self.setMouseTracking(True)
        self.setPixmap(QPixmap("imgs/background.png"))
        self.m_scaleValue = 1.0
        self.m_rectPixmap = QRectF(self.pixmap().rect())
        self.m_drawPoint = QPointF(0.0, 0.0)
        self.m_pressed = False
        self.m_lastPos = QPoint()

        # for keypoint
        self.points = []
        self.paint_flag = False
        self.keypoint_flag = False
        self.scaleX = 0.0
        self.scaleY = 0.0
        self.q_points = []  # QPoint for show

        # for add new bbox2d
        self.add_new_bbox2d_flag = False
        self.bbox2d = []
        self.types = []
        self.base_point = []
        self.centroid = []
        self.veh_size = []
        self.q_bbox2d = []
        self.q_start_point = None  # QPoint for show
        self.q_end_point = None  # QPoint for show
        self.flag = False
        self.show_cursor = False

    def mousePressEvent(self, event):
        if self.add_new_bbox2d_flag:
            if self.scaleX != 0 and self.scaleY != 0:
                if event.button() == Qt.LeftButton:
                    self.flag = True
                    self.q_start_point = event.pos()
                    self.q_end_point = event.pos()
                elif event.button() == Qt.RightButton:
                    if len(self.q_bbox2d) > 0:
                        self.bbox2d.pop(-1)
                        self.q_bbox2d.pop(-1)
                        self.types.pop(-1)
                        self.base_point.pop(-1)
                        self.centroid.pop(-1)
                        self.veh_size.pop(-1)
                        self.q_start_point = None
                        self.q_end_point = None
                        self.update()
                # if self.flag:
                #     self.update()
        elif self.keypoint_flag:
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
        else:
            if event.button() == Qt.LeftButton:
                self.m_pressed = True
                self.m_lastPos = event.pos()

    def mouseDoubleClickEvent(self, event):
        if self.add_new_bbox2d_flag:
            pass
        else:  # 双击屏幕复位
            self.m_scaleValue = 1.0
            self.m_drawPoint = QPointF(0.0, 0.0)
            self.update()

    def mouseMoveEvent(self, event):
        if self.add_new_bbox2d_flag:
            if self.show_cursor:
                self.pos = event.pos()
                self.update()
            if self.flag:
                self.q_end_point = event.pos()
                self.update()
        else:
            if self.m_pressed:
                delta = event.pos() - self.m_lastPos
                self.m_lastPos = event.pos()
                self.m_drawPoint += delta
                self.update()

    def mouseReleaseEvent(self, event):
        if self.add_new_bbox2d_flag:
            if event.button() == Qt.LeftButton:
                self.flag = False
                self.bbox2d.append([int(self.q_start_point.x() * self.scaleX), int(self.q_start_point.y() * self.scaleY),
                                    int(abs(self.q_end_point.x() - self.q_start_point.x()) * self.scaleX),
                                    int(abs(self.q_end_point.y() - self.q_start_point.y()) * self.scaleY)])
                self.q_bbox2d.append(QRect(self.q_start_point.x(), self.q_start_point.y(),
                                    abs(self.q_end_point.x() - self.q_start_point.x()),
                                    abs(self.q_end_point.y() - self.q_start_point.y())))
                self.base_point.append([int(self.q_start_point.x() * self.scaleX),
                                        int(self.q_end_point.y() * self.scaleY)])  # fixed bugs
                self.centroid.append([int(((self.q_end_point.x() + self.q_start_point.x()) / 2) * self.scaleX),
                                      int(((self.q_end_point.y() + self.q_start_point.y()) / 2) * self.scaleY)])

                q_dialog, dialog = QDialog(), dialog_bbox2d()
                # add classes
                dialog.setupUi(q_dialog)
                listview_model = QStringListModel()
                listview_model.setStringList(classes)
                dialog.listView_type.setModel(listview_model)

                q_dialog.show()
                if q_dialog.exec() == QDialog.Accepted:
                    index = dialog.listView_type.currentIndex().row()
                    self.types.append(str(classes[index]))
                    self.veh_size.append(veh_size_dict[str(classes[index])])
                else:
                    self.bbox2d.pop(-1)
                    self.q_bbox2d.pop(-1)
                    # self.base_point.pop(-1)
                    # self.veh_size.pop(-1)
                    # self.centroid.pop(-1)
                    self.q_start_point = None
                    self.q_end_point = None
                    self.update()
        else:
            if event.button() == Qt.LeftButton:
                self.m_pressed = False

    def wheelEvent(self, event):
        if self.add_new_bbox2d_flag:
            pass
        else:
            oldScale = self.m_scaleValue
            if event.angleDelta().y() > 0:
                self.m_scaleValue *= 1.1
            else:
                self.m_scaleValue *= 0.9

            if self.m_scaleValue > self.SCALE_MAX_VALUE:
                self.m_scaleValue = self.SCALE_MAX_VALUE

            if self.m_scaleValue < self.SCALE_MIN_VALUE:
                self.m_scaleValue = self.SCALE_MIN_VALUE

            if self.m_rectPixmap.contains(event.pos()):
                x = self.m_drawPoint.x() - (event.pos().x() - self.m_drawPoint.x()) / self.m_rectPixmap.width() * (
                            self.width() * (self.m_scaleValue - oldScale))
                y = self.m_drawPoint.y() - (event.pos().y() - self.m_drawPoint.y()) / self.m_rectPixmap.height() * (
                            self.height() * (self.m_scaleValue - oldScale))
                self.m_drawPoint = QPointF(x, y)
            else:
                x = self.m_drawPoint.x() - (self.width() * (self.m_scaleValue - oldScale)) / 2
                y = self.m_drawPoint.y() - (self.height() * (self.m_scaleValue - oldScale)) / 2
                self.m_drawPoint = QPointF(x, y)
            self.update()

    def paintEvent(self, event):
        if self.add_new_bbox2d_flag:
            super().paintEvent(event)
            if self.scaleX != 0 and self.scaleY != 0:
                painter_brush = QPainter()
                painter_brush.begin(self)
                painter_brush.setPen(QPen(Qt.red, 2, Qt.SolidLine))
                # point_lt_coordination = "(" + str(int(self.q_strat_point[i].x())) + "," + str(int(self.q_strat_point[i].y())) + ")"
                # painter_brush.drawText(self.q_strat_point[i], point_lt_coordination)
                # point_br_coordination = "(" + str(int(self.q_end_point[i].x())) + "," + str(int(self.q_end_point[i].y())) + ")"
                # painter_brush.drawText(self.q_end_point[i], point_br_coordination)
                # rect = QRect(self.q_start_point, self.q_end_point)
                # painter_brush.drawRect(rect)
                if self.q_start_point is not None and self.q_end_point is not None:
                    painter_brush.drawRect(self.q_start_point.x(), self.q_start_point.y(),
                                           self.q_end_point.x() - self.q_start_point.x(),
                                           self.q_end_point.y() - self.q_start_point.y())
                else:
                    pass
                for i in range(len(self.q_bbox2d)):
                    painter_brush.drawRect(self.q_bbox2d[i])
                if self.show_cursor:
                    painter_brush.setPen(QPen(Qt.white, 1, Qt.SolidLine))
                    painter_brush.drawLine(self.pos.x(), 0, self.pos.x(), self.window_height)
                    painter_brush.drawLine(0, self.pos.y(), self.window_width, self.pos.y())
        elif self.keypoint_flag:
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
        else:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.scale(self.window_width / self.pixmap().width() * self.m_scaleValue,
                          self.window_height / self.pixmap().height() * self.m_scaleValue)
            painter.drawPixmap(self.m_drawPoint, self.pixmap())

    def resizeEvent(self, event):
        if self.add_new_bbox2d_flag:
            pass
        else:
            super(MyLabel, self).resizeEvent(event)
            self.m_rectPixmap = QRectF(self.pixmap().rect())
            self.m_rectPixmap.setWidth(self.window_width / self.m_rectPixmap.width() * self.m_scaleValue)
            self.m_rectPixmap.setHeight(self.window_height / self.m_rectPixmap.height() * self.m_scaleValue)
            self.m_drawPoint = QPointF(0, 0)


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self)
        self.center()  # center

        self.frame = None
        self.pushButton_ClearAnnotations.setEnabled(False)

        self.mode = None
        self.pre_anno_dir = None
        self.pedes_mode = False

        self.add_new_bbox2d_flag = False

        self.perspective = "right"
        if self.perspective == "right":
            self.radioButton_BasePointRight.setChecked(True)
        else:
            self.radioButton_BasePointLeft.setChecked(True)
        self.l = self.doubleSpinBox_Bbox3D_Length.value() * 1000.0
        self.w = self.doubleSpinBox_Bbox3D_Width.value() * 1000.0
        self.h = self.doubleSpinBox_Bbox3D_Height.value() * 1000.0
        self.rot = self.doubleSpinBox_Bbox3D_Rot.value()
        self.key_point_nums = 4

        self.list_box = []
        self.list_type = []
        self.list_conf = []

        self.veh_box = []
        self.key_points = []

        self.listview_model_vehsize = QStringListModel()

        self.all_veh_2dbbox = []  # save annotations
        self.all_3dbbox_2dvertex = []
        self.all_vehicle_type = []
        self.all_vehicle_size = []
        self.all_vehicle_rots = []  # add vehicle rotations
        self.all_perspective = []
        self.all_base_point = []
        self.all_3dbbox_3dvertex = []
        self.all_vehicle_location = []
        self.all_vehicle_location_3d = []  # add vehicle center (m)
        self.all_veh_conf = []
        self.all_key_points = []

        # load redefined Mylabel
        self.label_ImageDisplay = MyLabel(self.groupBox_ImageDisplay,
                                          self.label_ImageDisplay.width(),
                                          self.label_ImageDisplay.height())
        self.label_ImageDisplay.setGeometry(QtCore.QRect(10, 20, 1091, 791))

        self.label_ImageDisplay.setScaledContents(True)
        self.label_ImageDisplay.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")  # style sheet

        self.actionpretrain_model_3d.triggered.connect(self.config_pretrain_model_3d)
        # self.window_pretrain_model_3d = Main_Pretrain()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "info", "Are you sure to exit?",
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

    def draw_3dbox(self, frame, mode):
        """ calculate and draw 3d box """
        if mode == "d":
            self.list_3dbbox_2dvertex, self.list_3dbbox_3dvertex, self.centroid_2d = cal_3dbbox_dairv2x(self.m_trans, self.veh_centroid_3d, self.l, self.w, self.h, self.rot)
            drawbox_img = self.paint(frame.copy(), self.list_3dbbox_2dvertex, self.centroid_2d)
        else:
            self.list_3dbbox_2dvertex, self.list_3dbbox_3dvertex, self.centroid_2d = cal_3dbbox(self.perspective, self.m_trans, self.veh_centroid, self.l, self.w, self.h, self.rot)
            # frame = cv.circle(frame.copy(), self.veh_base_point, 50, (255, 0, 0), 5, -1)
            drawbox_img = self.paint(frame.copy(), self.list_3dbbox_2dvertex, self.centroid_2d)
        if self.label_ImageDisplay.points:
            self.key_points = np.array(self.label_ImageDisplay.points).flatten().tolist()  # 每次绘制3d box时，保存拉平的关键点对
            self.paint_key_points(drawbox_img, self.key_points)
        return drawbox_img

    def paint(self, imgcopy, list_vertex, centroid_2d):
        """ paint 3d box """
        # width
        cv.line(imgcopy, list_vertex[0], (list_vertex[1][0], list_vertex[1][1]), (0, 0, 255), 3)
        # cv.line(imgcopy, list_vertex[2], list_vertex[3], (0, 0, 255), 2)
        dashLine(imgcopy, list_vertex[2], list_vertex[3], (255, 0, 255), 4, 5)
        cv.line(imgcopy, list_vertex[4], list_vertex[5], (0, 0, 255), 3)
        cv.line(imgcopy, list_vertex[6], list_vertex[7], (255, 0, 255), 4)

        # length
        cv.line(imgcopy, list_vertex[0], list_vertex[3], (255, 0, 0), 3)
        # cv.line(imgcopy, list_vertex[1], list_vertex[2], (255, 0, 0), 2)
        dashLine(imgcopy, (list_vertex[1][0], list_vertex[1][1]), list_vertex[2], (255, 0, 0), 3, 5)
        cv.line(imgcopy, list_vertex[4], list_vertex[7], (255, 0, 0), 3)
        cv.line(imgcopy, list_vertex[5], list_vertex[6], (255, 0, 0), 3)

        # height
        cv.line(imgcopy, list_vertex[0], list_vertex[4], (0, 255, 0), 3)
        cv.line(imgcopy, (list_vertex[1][0], list_vertex[1][1]), list_vertex[5], (0, 255, 0), 3)
        cv.line(imgcopy, list_vertex[3], list_vertex[7], (0, 255, 0), 3)
        # cv.line(imgcopy, list_vertex[2], list_vertex[6], (0, 255, 0), 2)
        dashLine(imgcopy, list_vertex[2], list_vertex[6], (0, 255, 0), 3, 5)

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
        self.pedes_mode = self.actionpedes.isChecked()
        self.label_ImageDisplay.keypoint_flag = self.actionkeypoint_only.isChecked()
        self.list_file_path = []
        self.str_folder_path = QFileDialog.getExistingDirectory(self, "Choose Folder", os.getcwd())
        # support chinese path
        self.str_folder_path = self.str_folder_path.encode("utf-8").decode("utf-8")

        if self.str_folder_path.split("/")[-1].startswith("dairv2x"):
            self.mode = "d"  # "dairv2x
        else:
            self.mode = "o"  # others

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
                    self.calib_file_path = self.str_folder_path + "/calib/" + os.listdir(os.path.join(self.str_folder_path, "calib"))[0]
                elif single_filename == "PreAnnos2D" and self.pedes_mode:
                    self.pre_anno_dir = os.path.join(self.str_folder_path, "PreAnnos2D")
                elif os.path.splitext(single_filename)[1] == ".bmp" or ".jpg" or "jpeg" or ".png":
                    self.list_file_path.append(self.str_folder_path + "/" + single_filename)  # get all suitable files
            # show in listview
            listview_model = QStringListModel()
            listview_model.setStringList(self.list_file_path)
            self.listView_FileList.setModel(listview_model)
        except Exception as e:  # if not choose file dir
            return

    def config_vehicle_size(self):
        """ config vehicle size """
        q_dialog, dialog = QDialog(), dialog_vehsize()
        dialog.setupUi(q_dialog)
        q_dialog.show()
        if q_dialog.exec() == QDialog.Accepted:
            veh_l = float(dialog.doubleSpinBox_l.text())
            veh_w = float(dialog.doubleSpinBox_w.text())
            veh_h = float(dialog.doubleSpinBox_h.text())
            line = str(veh_l) + "," + str(veh_w) + "," + str(veh_h)
            current_liststr = self.listview_model_vehsize.stringList()
            # put into list view
            if line not in current_liststr:
                current_liststr.append(line)
                self.listview_model_vehsize.setStringList(current_liststr)
                self.listView_VehSize.setModel(self.listview_model_vehsize)
            else:
                QMessageBox.information(self, "Information", "Repeated element! ",
                                        QMessageBox.Yes | QMessageBox.No)
        else:
            return

    def config_pretrain_model_3d(self):
        """ config pretrain model 3d """
        pass
        # self.window_pretrain_model_3d.show()

    def transfer_anno_vehicle_size(self, QModelIndex):
        row = self.listview_model_vehsize.stringList()[QModelIndex.row()]
        l, w, h = row.split(",")
        self.doubleSpinBox_Bbox3D_Length.setValue(float(l))
        self.doubleSpinBox_Bbox3D_Width.setValue(float(w))
        self.doubleSpinBox_Bbox3D_Height.setValue(float(h))

    def remove_listview_vehsize_item(self, QModelIndex):
        button = QMessageBox.question(self, "Question", "deleted? ",
                                      QMessageBox.Ok, QMessageBox.No)
        if button == QMessageBox.Ok:
            if self.listView_VehSize.selectedIndexes():
                self.listview_model_vehsize.removeRow(QModelIndex.row())
        else:
            return

    def listview_doubleclick_slot(self, QModelIndex):
        """ double-click to do object detection, and load calib information"""
        idx = 0

        self.pedes_mode = self.actionpedes.isChecked()
        self.label_ImageDisplay.keypoint_flag = self.actionkeypoint_only.isChecked()
        self.pushButton_add_new_bbox2d.setEnabled(True)
        self.pushButton_update_new_bbox2d.setEnabled(False)

        self.all_veh_2dbbox.clear()
        self.all_3dbbox_2dvertex.clear()
        self.all_vehicle_type.clear()
        self.all_vehicle_size.clear()
        self.all_vehicle_rots.clear()
        self.all_perspective.clear()
        self.all_base_point.clear()
        self.all_3dbbox_3dvertex.clear()
        self.all_vehicle_location.clear()
        self.all_vehicle_location_3d.clear()
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
        if self.mode == "d":
            self.calib_file_path = self.str_folder_path + "/calib/" + self.select_file.split("/")[-1][:-4] + "_calib.xml"
            self.m_trans = ReadCalibParam(self.calib_file_path)
        else:
            self.focal, self.fi, self.theta, self.cam_height, self.turple_vp, self.vpline = ReadCalibParam(self.calib_file_path)
            # self.theta = - self.theta
            self.m_trans = ParamToMatrix(self.focal, self.fi, self.theta, self.cam_height, self.frame.shape[1]/2, self.frame.shape[0]/2)
            self.veh_turple_vp = self.turple_vp

        # load annotation files
        # if not exists, load img to detection
        # if self.pedes_mode:
        #     self.select_file_xml = self.select_file.split('.')[0] + '_sup.xml'
        # else:
        self.select_file_xml = self.select_file.split('.')[0] + '.xml'
        vehsize_lines = []  # for listview
        if os.path.exists(self.select_file_xml):
            tree = ET.parse(self.select_file_xml)
            root = tree.getroot()
            # box_nums
            box_nums = len(tree.findall("object"))
            tp_veh_key_point_data = np.zeros((box_nums, 2*self.key_point_nums))
            # box
            for id, obj in enumerate(root.iter('object')):
                # 2、vehicle type (0，1，2)
                veh_type_data = obj.find('type').text
                if veh_type_data not in classes:
                    continue
                else:
                    if self.mode == "d":
                        dtype = int
                    else:
                        dtype = int
                    # 1、2d box [left, top, width, height]
                    bbox2d_data = obj.find('bbox2d').text.split()
                    bbox2d_data = [dtype(float(box)) for box in bbox2d_data]
                    self.all_veh_2dbbox.append(bbox2d_data)

                    veh_cls_id = classes.index(veh_type_data)
                    self.all_vehicle_type.append(veh_type_data)
                    # 3、centroid 2d (int)
                    veh_centre_data = obj.find('veh_loc_2d').text.split()
                    veh_centre_data = [dtype(float(loc)) for loc in veh_centre_data]
                    self.all_vehicle_location.append(veh_centre_data)
                    # 4、vertex 2d (int)
                    veh_vertex_data = []  # for each box (8 vertex)
                    box_2dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex2d').text)
                    for x in box_2dvertex:
                        veh_vertex_data += [dtype(float(item)) for item in x.split(", ")]  # [x1,y1,x2,y2,...,x8,y8]
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
                    # listview
                    veh_l, veh_w, veh_h = veh_size_data
                    vehsize_line = str(veh_l) + "," + str(veh_w) + "," + str(veh_h)
                    vehsize_lines.append(vehsize_line)

                    if self.mode == "o":
                        # vehicle rot
                        if obj.find('veh_angle') is not None:
                            if obj.find('veh_angle').text.startswith("["):
                                veh_rot_data = float(obj.find('veh_angle').text[1:-1])
                            else:
                                veh_rot_data = float(obj.find('veh_angle').text)
                            self.all_vehicle_rots.append(veh_rot_data)
                        else:
                            self.all_vehicle_rots.append(180.0)

                    # 6、view (left, right)
                    veh_view_data = obj.find('perspective').text
                    self.all_perspective.append(veh_view_data)
                    # 7、vehicle base point (int)
                    veh_base_point_data = obj.find('base_point').text.split()
                    veh_base_point_data = [dtype(float(base_point)) for base_point in veh_base_point_data]
                    # load annotations
                    # veh_base_point_data = [veh_vertex_data[2], veh_vertex_data[3]]
                    self.all_base_point.append(veh_base_point_data)

                    # veh_vertex_data -> (tuple)
                    tp_veh_vertex_data = []
                    for i in range(0, len(veh_vertex_data)-1, 2):
                        tp_veh_vertex_data.append((veh_vertex_data[i], veh_vertex_data[i+1]))
                    self.all_3dbbox_2dvertex[idx] = tp_veh_vertex_data

                    tp_veh_3dvertex_data = []
                    for i in range(0, len(veh_3dvertex_data)-2, 3):
                        tp_veh_3dvertex_data.append((veh_3dvertex_data[i], veh_3dvertex_data[i+1], veh_3dvertex_data[i+2]))
                    self.all_3dbbox_3dvertex[idx] = tp_veh_3dvertex_data

                    self.all_veh_conf.append(1.0)

                    # veh_centre_data -> (tuple)
                    tp_veh_centre_data = (dtype(veh_centre_data[0]), dtype(veh_centre_data[1]))

                    if self.mode == "d":
                        veh_centre_3d_data = obj.find('veh_loc_3d').text.split()
                        veh_centre_3d_data = [float(loc) for loc in veh_centre_3d_data]
                        self.all_vehicle_location_3d.append(veh_centre_3d_data)
                        veh_angle_data = obj.find('veh_angle').text
                        veh_angle_data = np.rad2deg(float(veh_angle_data))
                        self.all_vehicle_rots.append(veh_angle_data)

                    # 8、key_point
                    if obj.find('key_points') is not None:
                        veh_key_point_data = obj.find('key_points')
                        if veh_key_point_data:
                            veh_key_point_data = veh_key_point_data.text.split()
                            veh_key_point_data = [dtype(float(key_point)) for key_point in veh_key_point_data]
                            tp_veh_key_point_data[idx] = veh_key_point_data

                        # show 2d keypoint
                        if np.array(veh_key_point_data).all():
                            self.frame = self.paint_key_points(self.frame, veh_key_point_data)

                    # draw 2D box
                    if self.mode == "d":
                        left, top, right, bottom = int(bbox2d_data[0]), int(bbox2d_data[1]), int(bbox2d_data[2]), int(bbox2d_data[3])
                    else:
                        left, top, right, bottom = int(bbox2d_data[0]), int(bbox2d_data[1]), int(bbox2d_data[0]) + int(bbox2d_data[2]), int(bbox2d_data[1]) + int(bbox2d_data[3])

                    cv.rectangle(self.frame, (left, top), (right, bottom), (0, 128, 255), 3)
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
                    self.frame = self.paint(self.frame, tp_veh_vertex_data, tp_veh_centre_data)
                    idx += 1
            self.textEdit_ObjNums.setText(str(idx))
            self.obj_num = int(self.textEdit_ObjNums.toPlainText())
            # put into list view
            current_liststr = self.listview_model_vehsize.stringList()
            for v in set(vehsize_lines):
                if v not in current_liststr:
                    current_liststr += [v]
            self.listview_model_vehsize.setStringList(current_liststr)
            self.listView_VehSize.setModel(self.listview_model_vehsize)

            self.all_key_points = tp_veh_key_point_data.tolist()
            self.show_img_in_label(self.label_ImageDisplay, self.frame)
        if not os.path.exists(self.select_file_xml) and self.pre_anno_dir is not None and self.pedes_mode:
            pre_anno_path = os.path.join(self.pre_anno_dir, self.select_file.split("/")[-1][:-4] + "_bbox2d.txt")
            with open(pre_anno_path, "r", encoding="utf-8") as f:
                pre_annos = f.readlines()
            for pre_anno in pre_annos:
                pre_anno = pre_anno.strip("\n").split(" ")
                cls_name = str(pre_anno[0])
                if cls_name == "Pedestrian":
                    score = float(pre_anno[1])
                    left, top, right, bottom = np.array(pre_anno[2:], dtype=np.float)
                    left, top, right, bottom = int(left), int(top), int(right), int(bottom)

                    self.list_box.append([left, top, right - left, bottom - top])
                    self.list_type.append(cls_name)
                    self.list_conf.append(score)

                    self.all_veh_2dbbox.append([left, top, right - left, bottom - top])
                    self.all_vehicle_type.append(cls_name)
                    self.all_perspective.append("left")
                    self.all_base_point.append([left, bottom])
                    self.all_vehicle_size.append(veh_size_dict[cls_name])
                    self.all_vehicle_rots.append(180.0)
                    self.all_3dbbox_2dvertex.append([(0, 0) for _ in range(8)])
                    self.all_3dbbox_3dvertex.append([(0, 0, 0) for _ in range(8)])
                    self.all_vehicle_location_3d.append(np.zeros((1, 3)))
                    self.all_vehicle_location.append([(left + right) / 2, (top + bottom) / 2])
                    self.all_veh_conf.append(score)

                    # draw 2D box
                    cv.rectangle(self.frame, (left, top), (right, bottom), (0, 128, 255), 3)
                    label = '%s:%.2f-%s' % (cls_name, score, str(idx + 1))

                    # Display the label at the top of the bounding box
                    # display the idx of each object
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    top = max(top, labelSize[1])
                    cv.rectangle(self.frame, (left, top - round(1.5 * labelSize[1])),
                                 (left + round(1.5 * labelSize[0]), top + baseLine),
                                 (255, 255, 255), cv.FILLED)
                    cv.putText(self.frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

                    idx += 1
                else:
                    continue
            self.textEdit_ObjNums.setText(str(idx))
            self.obj_num = int(self.textEdit_ObjNums.toPlainText())

            self.show_img_in_label(self.label_ImageDisplay, self.frame)
        if not os.path.exists(self.select_file_xml) and self.pre_anno_dir is None:
            # object detection
            self.thread.Init(self.frame)
            self.thread.send_detect_result.connect(self.image_label_display)
            self.thread.start()

    def limit_spin_box_value(self, value):
        if value > self.obj_num:
            value = 1
        elif value < 1:
            value = self.obj_num
        else:
            pass
        return value

    def spin_cur_anno_order(self):
        """ choose annotation object """
        # if annotation file exists, revise can be done.
        if self.frame is not None:
            self.spinBox_CurAnnNum.setValue(self.limit_spin_box_value(self.spinBox_CurAnnNum.value()))  # roll
            if os.path.exists(self.select_file_xml) or self.pedes_mode:
                self.obj_num = len(self.all_vehicle_type)
                if (self.spinBox_CurAnnNum.value()) <= self.obj_num and self.spinBox_CurAnnNum.value() >= 1:
                    # self.spinBox_CurAnnNum.setMaximum(self.obj_num)  # max obj num
                    self.veh_box = self.all_veh_2dbbox[self.spinBox_CurAnnNum.value() - 1]
                    self.veh_type = self.all_vehicle_type[self.spinBox_CurAnnNum.value() - 1]
                    self.comboBox_CurAnnType.setCurrentIndex(dict_map_order_str[self.veh_type])
                    self.veh_conf = self.all_veh_conf[self.spinBox_CurAnnNum.value() - 1]

                    self.perspective = self.all_perspective[self.spinBox_CurAnnNum.value() - 1]
                    if self.perspective == "left":
                        self.radioButton_BasePointLeft.setChecked(True)
                    else:
                        self.radioButton_BasePointRight.setChecked(True)
                    self.centroid = self.all_vehicle_location[self.spinBox_CurAnnNum.value() - 1]
                    self.veh_centroid = self.centroid
                    self.base_point = self.all_base_point[self.spinBox_CurAnnNum.value() - 1]
                    self.veh_base_point = self.base_point

                    if self.mode == "d":
                        self.l = self.all_vehicle_size[self.spinBox_CurAnnNum.value() - 1][0]
                        self.w = self.all_vehicle_size[self.spinBox_CurAnnNum.value() - 1][1]
                        self.h = self.all_vehicle_size[self.spinBox_CurAnnNum.value() - 1][2]
                        self.doubleSpinBox_Bbox3D_Length.setValue(self.l)
                        self.doubleSpinBox_Bbox3D_Width.setValue(self.w)
                        self.doubleSpinBox_Bbox3D_Height.setValue(self.h)
                    else:
                        self.l = self.all_vehicle_size[self.spinBox_CurAnnNum.value() - 1][0] * 1000
                        self.w = self.all_vehicle_size[self.spinBox_CurAnnNum.value() - 1][1] * 1000
                        self.h = self.all_vehicle_size[self.spinBox_CurAnnNum.value() - 1][2] * 1000
                        self.doubleSpinBox_Bbox3D_Length.setValue(self.l / 1000)
                        self.doubleSpinBox_Bbox3D_Width.setValue(self.w / 1000)
                        self.doubleSpinBox_Bbox3D_Height.setValue(self.h / 1000)

                    if self.mode == "d":
                        self.centriod_3d = self.all_vehicle_location_3d[self.spinBox_CurAnnNum.value() - 1]
                        self.veh_centroid_3d = self.centriod_3d
                        self.rot = self.all_vehicle_rots[self.spinBox_CurAnnNum.value() - 1]
                    else:
                        self.rot = self.all_vehicle_rots[self.spinBox_CurAnnNum.value() - 1]
                        self.dial_Bbox3D_Rot.setValue(int(self.rot))
                    self.dial_Bbox3D_Rot.setValue(int(self.rot))
                    self.doubleSpinBox_Bbox3D_Rot.setValue(self.rot)
                    self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
                    self.all_3dbbox_2dvertex[self.spinBox_CurAnnNum.value() - 1] = self.list_3dbbox_2dvertex
                    self.all_3dbbox_3dvertex[self.spinBox_CurAnnNum.value() - 1] = self.list_3dbbox_3dvertex
                    self.all_vehicle_location[self.spinBox_CurAnnNum.value() - 1] = self.centroid_2d

            else:  # make annotation from zero
                try:
                    if (self.spinBox_CurAnnNum.value()) <= self.obj_num and self.spinBox_CurAnnNum.value() >= 1:
                        # self.spinBox_CurAnnNum.setMaximum(self.obj_num)  # max obj num
                        self.veh_box = self.list_box[self.spinBox_CurAnnNum.value() - 1]
                        self.veh_type = self.list_type[self.spinBox_CurAnnNum.value() - 1]
                        self.veh_conf = self.list_conf[self.spinBox_CurAnnNum.value() - 1]
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

                        self.l = self.doubleSpinBox_Bbox3D_Length.value() * 1000
                        self.w = self.doubleSpinBox_Bbox3D_Width.value() * 1000
                        self.h = self.doubleSpinBox_Bbox3D_Height.value() * 1000
                        self.rot = self.dial_Bbox3D_Rot.value()

                        self.centroid = (self.veh_box[0] + self.veh_box[2] / 2, self.veh_box[1] + self.veh_box[3] / 2)
                        self.veh_centroid = self.centroid

                        # key-point mode
                        if self.actionkeypoint_only.isChecked():
                            if self.label_ImageDisplay.points:
                                self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
                            else:
                                QMessageBox.information(self, "Information", "Please make key points annotation first! ",
                                                        QMessageBox.Yes | QMessageBox.No)
                                self.spinBox_CurAnnNum.setValue(-1)
                        else:
                            self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
                except:
                    QMessageBox.information(self, "Information", "Please choose one vehicle! ",
                                            QMessageBox.Yes | QMessageBox.No)

    def add_new_bbox2d(self):
        if self.frame is not None:
            self.add_new_bbox2d_flag = True
            self.label_ImageDisplay.add_new_bbox2d_flag = True
            self.label_ImageDisplay.setCursor(Qt.CrossCursor)
            self.pushButton_add_new_bbox2d.setEnabled(False)
            self.label_ImageDisplay.show_cursor = True
            self.pushButton_update_new_bbox2d.setEnabled(True)

    def cancel_add_new_bbox2d(self):
        self.pushButton_add_new_bbox2d.setEnabled(True)
        self.add_new_bbox2d_flag = False
        self.label_ImageDisplay.add_new_bbox2d_flag = False
        self.label_ImageDisplay.unsetCursor()
        self.label_ImageDisplay.show_cursor = False

    def update_new_bbox2d(self):
        if self.frame is not None:
            base_num = len(self.all_vehicle_type)
            if len(self.label_ImageDisplay.bbox2d) > 0:
                self.all_veh_2dbbox.extend(self.label_ImageDisplay.bbox2d)
                self.all_vehicle_type.extend(self.label_ImageDisplay.types)
                self.all_perspective.extend(["left"] * len(self.label_ImageDisplay.bbox2d))
                self.all_base_point.extend(self.label_ImageDisplay.base_point)
                self.all_vehicle_size.extend(self.label_ImageDisplay.veh_size)
                self.all_vehicle_rots.extend([180.0] * len(self.label_ImageDisplay.bbox2d))
                self.all_3dbbox_2dvertex.extend([[(0, 0) for _ in range(8)]] * len(self.label_ImageDisplay.bbox2d))
                self.all_3dbbox_3dvertex.extend([[(0, 0, 0) for _ in range(8)]] * len(self.label_ImageDisplay.bbox2d))
                self.all_vehicle_location_3d.extend(np.zeros((len(self.label_ImageDisplay.bbox2d), 3)))
                self.all_vehicle_location.extend(self.label_ImageDisplay.centroid)  # update centroid for draw 3d bbox
                self.all_veh_conf.extend([1.0] * len(self.label_ImageDisplay.bbox2d))
                for idx, box in enumerate(self.label_ImageDisplay.bbox2d):
                    left, top, right, bottom = int(box[0]), int(box[1]), \
                                               int(box[0]) + int(box[2]), \
                                               int(box[1]) + int(box[3])

                    cv.rectangle(self.frame, (left, top), (right, bottom), (0, 128, 255), 3)
                    label = '%s:%.2f-%s' % (self.label_ImageDisplay.types[idx], 1.00, str(base_num + idx + 1))

                    # Display the label at the top of the bounding box
                    # display the idx of each object
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    top = max(top, labelSize[1])
                    cv.rectangle(self.frame, (left, top - round(1.5 * labelSize[1])),
                                 (left + round(1.5 * labelSize[0]), top + baseLine),
                                 (255, 255, 255), cv.FILLED)
                    cv.putText(self.frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
                self.show_img_in_label(self.label_ImageDisplay, self.frame)
                self.label_ImageDisplay.q_bbox2d.clear()
                self.label_ImageDisplay.bbox2d.clear()
                self.label_ImageDisplay.veh_size.clear()
                self.label_ImageDisplay.types.clear()
                self.label_ImageDisplay.q_bbox2d.clear()
                self.label_ImageDisplay.base_point.clear()
                self.label_ImageDisplay.centroid.clear()
                self.label_ImageDisplay.q_start_point = None
                self.label_ImageDisplay.q_end_point = None
                self.label_ImageDisplay.update()
                self.textEdit_ObjNums.setText(str(len(self.all_vehicle_type)))
                self.obj_num = len(self.all_vehicle_type)
                self.label_ImageDisplay.add_new_bbox2d_flag = False
                self.pushButton_add_new_bbox2d.setEnabled(True)
                self.label_ImageDisplay.unsetCursor()
                self.label_ImageDisplay.show_cursor = False

    def radio_bp_left(self):
        """ choose base point: left bottom """
        try:
            if self.mode == "d":
                QMessageBox.information(self, "Information", "perspective ignored! ",
                                        QMessageBox.Yes | QMessageBox.No)
            else:
                self.perspective = "left"
                self.base_point = (self.veh_box[0], self.veh_box[1] + self.veh_box[3])
                self.veh_base_point = self.base_point
                self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def radio_bp_right(self):
        """ choose base point: right bottom """
        try:
            if self.mode == "d":
                QMessageBox.information(self, "Information", "perspective ignored! ",
                                        QMessageBox.Yes | QMessageBox.No)
            else:
                self.perspective = "right"
                self.base_point = (self.veh_box[0] + self.veh_box[2], self.veh_box[1] + self.veh_box[3])
                self.veh_base_point = self.base_point
                self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_bp_adjust_lr(self):
        """ adjust base point: left right """
        try:
            if self.mode == "d":
                self.veh_centroid_3d = (self.veh_centroid_3d[0], self.centriod_3d[1] - self.horizontalSlider_BasePointAdj_LR.value() / 100.0, self.veh_centroid_3d[2])
                self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
            else:
                self.veh_centroid = (self.centroid[0] + self.horizontalSlider_BasePointAdj_LR.value(), self.veh_centroid[1])
                self.veh_base_point = (self.base_point[0] + self.horizontalSlider_BasePointAdj_LR.value(), self.veh_base_point[1])
                self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_bp_adjust_ud(self):
        """ choose base point: top bottom """
        try:
            if self.mode == "d":
                self.veh_centroid_3d = (self.centriod_3d[0] - self.verticalSlider_BasePointAdj_UD.value() / 30.0, self.veh_centroid_3d[1], self.veh_centroid_3d[2])
                self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
            else:
                self.veh_centroid = (self.veh_centroid[0], self.centroid[1] + self.verticalSlider_BasePointAdj_UD.value())
                self.veh_base_point = (self.veh_base_point[0], self.base_point[1] + self.verticalSlider_BasePointAdj_UD.value())
                self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_vp_adjust_lr(self):
        """ move vp along horizon line """
        try:
            if self.mode == "d":
                QMessageBox.information(self, "Information", "vp ignored! ",
                                        QMessageBox.Yes | QMessageBox.No)
            else:
                self.veh_vpx = self.turple_vp[0] + self.horizontalSlider_VPAdj_LR.value()
                self.veh_vpy = self.vpline[0] * (self.veh_vpx - self.vpline[1]) + self.vpline[2]
                self.veh_turple_vp = (self.veh_vpx, self.veh_vpy)
                self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def save_temp_annotation_results(self):
        """ save single obj annotation """
        # if annotation file exists, revise can be done.
        if self.frame is not None and self.drawbox_img is not None:
            if os.path.exists(self.select_file_xml) or self.pedes_mode:
                self.frame = self.drawbox_img
                if self.actionkeypoint_only.isChecked():
                    self.all_key_points[self.spinBox_CurAnnNum.value() - 1] = self.key_points
                    self.label_ImageDisplay.points.clear()
                    self.label_ImageDisplay.q_points.clear()
                else:
                    self.all_3dbbox_2dvertex[self.spinBox_CurAnnNum.value() - 1] = self.list_3dbbox_2dvertex
                    self.all_vehicle_type[self.spinBox_CurAnnNum.value() - 1] = self.comboBox_CurAnnType.currentText()
                    self.all_vehicle_size[self.spinBox_CurAnnNum.value() - 1] = [self.doubleSpinBox_Bbox3D_Length.value(), self.doubleSpinBox_Bbox3D_Width.value(), self.doubleSpinBox_Bbox3D_Height.value()]
                    self.all_vehicle_rots[self.spinBox_CurAnnNum.value() - 1] = self.doubleSpinBox_Bbox3D_Rot.value()
                    self.all_perspective[self.spinBox_CurAnnNum.value() - 1] = self.perspective
                    self.all_base_point[self.spinBox_CurAnnNum.value() - 1] = [self.list_3dbbox_2dvertex[1][0], self.list_3dbbox_2dvertex[1][1]]  # 基准点p1
                    self.all_3dbbox_3dvertex[self.spinBox_CurAnnNum.value() - 1] = self.list_3dbbox_3dvertex
                    self.all_vehicle_location[self.spinBox_CurAnnNum.value() - 1] = [self.centroid_2d[0], self.centroid_2d[1]]
                    if self.mode == "d":
                        self.all_vehicle_location_3d[self.spinBox_CurAnnNum.value() - 1] = self.veh_centroid_3d
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
                        self.all_vehicle_rots.append(self.doubleSpinBox_Bbox3D_Rot.value())
                        self.all_perspective.append(self.perspective)
                        self.all_base_point.append(self.list_3dbbox_2dvertex[1])  # base point p1
                        self.all_3dbbox_3dvertex.append(self.list_3dbbox_3dvertex)
                        self.all_vehicle_location.append(self.centroid_2d)
                        if self.mode == "d":
                            self.all_vehicle_location_3d.append(self.veh_centroid_3d)
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
                cv.imwrite(self.select_file[0:len(self.select_file)-4] + "_drawbbox_result.png", self.frame)
                # if self.pre_anno_dir is not None:
                #     xml_path = self.select_file[0:len(self.select_file)-4] + "_sup.xml"
                # else:
                xml_path = self.select_file[0:len(self.select_file) - 4] + ".xml"
                save3dbbox_result(self.mode, xml_path, self.select_file, self.calib_file_path, self.frame, self.all_veh_2dbbox, self.all_vehicle_type, self.all_3dbbox_2dvertex,
                self.all_vehicle_size, self.all_vehicle_rots, self.all_vehicle_location_3d, self.all_perspective, self.all_base_point, self.all_3dbbox_3dvertex, self.all_vehicle_location, self.all_key_points, self.actionkeypoint_only.isChecked())
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
                self.all_vehicle_rots.pop(-1)
                self.all_perspective.pop(-1)
                self.all_base_point.pop(-1)  # base point p1
                self.all_3dbbox_3dvertex.pop(-1)
                self.all_vehicle_location.pop(-1)
                self.all_vehicle_location_3d.pop(-1)
                self.all_veh_conf.pop(-1)  # fix
                self.all_key_points.pop(-1)
                for i in range(len(self.list_box)):
                    # draw 2D box
                    left, top, right, bottom = int(self.list_box[i][0]), int(self.list_box[i][1]), int(
                        self.list_box[i][0]) + int(
                        self.list_box[i][2]), int(self.list_box[i][1]) + int(self.list_box[i][3])
                    cv.rectangle(frame_copy_, (left, top), (right, bottom), (0, 128, 255), 3)
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
                self.spinBox_CurAnnNum.setValue(self.spinBox_CurAnnNum.value())
                self.show_img_in_label(self.label_ImageDisplay, frame_copy_)
            else:
                for i in range(len(self.list_box)):
                    # draw 2D box
                    left, top, right, bottom = int(self.list_box[i][0]), int(self.list_box[i][1]), int(
                        self.list_box[i][0]) + int(
                        self.list_box[i][2]), int(self.list_box[i][1]) + int(self.list_box[i][3])
                    cv.rectangle(self.frame_copy, (left, top), (right, bottom), (0, 128, 255), 3)
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
            if self.mode == "d":
                self.l = self.doubleSpinBox_Bbox3D_Length.value()
            else:
                self.l = self.doubleSpinBox_Bbox3D_Length.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def spind_3dbbox_width(self):
        """ adjust vehicle width """
        try:
            if self.mode == "d":
                self.w = self.doubleSpinBox_Bbox3D_Width.value()
            else:
                self.w = self.doubleSpinBox_Bbox3D_Width.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def spind_3dbbox_height(self):
        """ adjust vehicle height """
        try:
            if self.mode == "d":
                self.h = self.doubleSpinBox_Bbox3D_Height.value()
            else:
                self.h = self.doubleSpinBox_Bbox3D_Height.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def dial_box_rot_adjust(self):
        """ adjust vehicle rot by dial """
        try:
            self.rot = self.dial_Bbox3D_Rot.value()
            self.doubleSpinBox_Bbox3D_Rot.setValue(float(self.rot))
            self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def spind_3dbbox_rot(self):
        """ adjust vehicle rot by double spin"""
        try:
            self.rot = self.doubleSpinBox_Bbox3D_Rot.value()
            self.dial_Bbox3D_Rot.setValue(int(self.rot))
            self.drawbox_img = self.draw_3dbox(self.frame, self.mode)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())
