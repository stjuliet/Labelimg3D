import os
import cv2 as cv
import sys
from interface import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from YoloDetect import YoloDetect
from copy import deepcopy
import re
import xml.etree.ElementTree as ET
from tools import ReadCalibParam, ParamToMatrix, cal_3dbbox, save3dbbox_result, dashLine


# 初始化读取车辆类型字典
dict_map_order_str = {'car': 1, 'truck': 2, 'bus': 3}
classes = ["Car", "Truck", "Bus"]


# 重写主窗体类
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        # 初始化窗体显示
        self.setupUi(self)
        self.center()  # 居中窗口显示

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

        self.veh_box = []

        self.all_veh_2dbbox = []  # 保存标注信息
        self.all_3dbbox_2dvertex = []
        self.all_vehicle_type = []
        self.all_vehicle_size = []
        self.all_perspective = []
        self.all_base_point = []
        self.all_3dbbox_3dvertex =[]
        self.all_vehicle_location = []
        self.all_veh_conf = []

        # 设置在label中自适应显示图片
        self.label_ImageDisplay.setScaledContents(True)
        # 初始化label背景色为全黑
        self.label_ImageDisplay.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")  # 设置样式表

    def closeEvent(self, event):
        reply = QMessageBox.information(self, "info", "Are you sure to exit?",
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit()
        else:
            event.ignore()

    def center(self):  # 定义一个函数使得窗口居中显示
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def show_img_in_label(self, label_name, img):
        """ 显示图片至label控件 """
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        qimage = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], rgb_img.shape[1] * 3, QImage.Format_RGB888)
        label_name.setPixmap(QPixmap.fromImage(qimage))

    def draw_3dbox(self, frame):
        '''计算并绘制3d bbox'''
        self.list_3dbbox_2dvertex, self.list_3dbbox_3dvertex, self.centroid_2d = cal_3dbbox(self.perspective, self.m_trans, self.veh_base_point, self.veh_turple_vp, self.l, self.w, self.h)
        drawbox_img = self.paint(frame.copy(), self.list_3dbbox_2dvertex, self.centroid_2d)
        return drawbox_img

    def paint(self, imgcopy, list_vertex, centroid_2d):
        '''由8个顶点绘制3d bbox'''
        # 宽度
        cv.line(imgcopy, list_vertex[0], (list_vertex[1][0], list_vertex[1][1]), (0, 0, 255), 2)
        # cv.line(imgcopy, list_vertex[2], list_vertex[3], (0, 0, 255), 2)
        dashLine(imgcopy, list_vertex[2], list_vertex[3], (0, 0, 255), 2, 5)
        cv.line(imgcopy, list_vertex[4], list_vertex[5], (0, 0, 255), 2)
        cv.line(imgcopy, list_vertex[6], list_vertex[7], (0, 0, 255), 2)

        # 长度
        cv.line(imgcopy, list_vertex[0], list_vertex[3], (255, 0, 0), 2)
        # cv.line(imgcopy, list_vertex[1], list_vertex[2], (255, 0, 0), 2)
        dashLine(imgcopy, (list_vertex[1][0], list_vertex[1][1]), list_vertex[2], (255, 0, 0), 2, 5)
        cv.line(imgcopy, list_vertex[4], list_vertex[7], (255, 0, 0), 2)
        cv.line(imgcopy, list_vertex[5], list_vertex[6], (255, 0, 0), 2)

        # 高度
        cv.line(imgcopy, list_vertex[0], list_vertex[4], (0, 255, 0), 2)
        cv.line(imgcopy, (list_vertex[1][0], list_vertex[1][1]), list_vertex[5], (0, 255, 0), 2)
        cv.line(imgcopy, list_vertex[3], list_vertex[7], (0, 255, 0), 2)
        # cv.line(imgcopy, list_vertex[2], list_vertex[6], (0, 255, 0), 2)
        dashLine(imgcopy, list_vertex[2], list_vertex[6], (0, 255, 0), 2, 5)

        # 质心
        cv.circle(imgcopy, centroid_2d, 5, (0, 255, 255), 3)

        self.show_img_in_label(self.label_ImageDisplay, imgcopy)

        return imgcopy

    def image_label_display(self, object):
        '''目标检测传回信号的槽函数'''
        # object[0]: detected frame(QPixmap)
        # object[1]: listbox
        # object[2]: listtype
        self.label_ImageDisplay.setPixmap(object[0])
        self.textEdit_ObjNums.setText(str(len(object[1])))
        self.obj_num = int(self.textEdit_ObjNums.toPlainText())
        self.list_box = object[1]
        self.list_type = object[2]
        self.list_conf = object[3]

    def choose_img_folder(self):
        '''选择标注文件夹'''
        self.list_file_path = []
        self.str_folder_path = QFileDialog.getExistingDirectory(self, "Choose Folder", os.getcwd())

        parent = os.path.abspath(os.path.join(self.str_folder_path, os.pardir))
        # yolov4 model files and thread initialize
        self.classesFile = parent + "\\model_us_yolov4\\obj.names"
        self.modelConfiguration = parent + "\\model_us_yolov4\\yolov4.cfg"
        self.modelWeights = parent + "\\model_us_yolov4\\yolov4_best.weights"
        if os.path.exists(self.classesFile) and os.path.exists(self.modelConfiguration) and os.path.exists(self.modelWeights):
            self.thread = YoloDetect(self.classesFile, self.modelConfiguration, self.modelWeights)
        else:
            return
        try:
            all_file_names = os.listdir(self.str_folder_path)  # 获得文件夹下所有文件名
            for single_filename in all_file_names:
                if single_filename == 'calib':
                    self.calib_file_path = self.str_folder_path + '/calib/' + os.listdir(self.str_folder_path + '/calib')[0]
                elif os.path.splitext(single_filename)[1] == '.bmp' or '.jpg' or 'jpeg' or '.png':
                    self.list_file_path.append(self.str_folder_path + '/' + single_filename)  # 获得所有jpg文件名
            # 显示到listview
            listview_model = QStringListModel()  # 创建mode
            listview_model.setStringList(self.list_file_path)  # 将数据设置到model
            self.listView_FileList.setModel(listview_model)  # 绑定 listView 和 model
        except Exception as e:  # 如果没有选择到文件夹路径则返回
            return

    def listview_doubleclick_slot(self, QModelIndex):
        '''双击listview中行显示目标检测结果,并读取标定相关参数及消失点'''
        # 每次换新图片将暂存的上次结果全部清除
        self.all_veh_2dbbox.clear()
        self.all_3dbbox_2dvertex.clear()
        self.all_vehicle_type.clear()
        self.all_vehicle_size.clear()
        self.all_perspective.clear()
        self.all_base_point.clear()
        self.all_3dbbox_3dvertex.clear()
        self.all_vehicle_location.clear()
        self.all_veh_conf.clear()
        self.select_file = self.list_file_path[QModelIndex.row()]
        self.spinBox_CurAnnNum.setValue(-1)

        # 读入图像
        if self.select_file.split('.')[-1] == 'jpg':
            self.frame = cv.imread(self.select_file)  # 读取选中行
            self.frame_copy = deepcopy(self.frame)
        else:
            return

        # 读取标定参数
        self.focal, self.fi, self.theta, self.cam_height, self.turple_vp, self.vpline = \
            ReadCalibParam(self.calib_file_path)
        # self.theta = - self.theta
        self.m_trans = ParamToMatrix(self.focal, self.fi, self.theta, self.cam_height, self.frame.shape[1]/2, self.frame.shape[0]/2)
        self.veh_turple_vp = self.turple_vp

        # 读入标注文件, 如果无标注文件再进行目标检测
        self.select_file_xml = self.select_file.split('.')[0] + '.xml'
        if os.path.exists(self.select_file_xml):
            # 读入标注文件, 并绘制于原图
            tree = ET.parse(self.select_file_xml)
            root = tree.getroot()
            # box
            for idx, obj in enumerate(root.iter('object')):
                # 1、二维框[left, top, width, height]
                bbox2d_data = obj.find('bbox2d').text.split()
                bbox2d_data = [int(box) for box in bbox2d_data]
                self.all_veh_2dbbox.append(bbox2d_data)
                # 2、车辆类型(0，1，2)
                veh_type_data = obj.find('type').text
                if veh_type_data not in classes:
                    continue
                veh_cls_id = classes.index(veh_type_data)
                self.all_vehicle_type.append(veh_type_data)
                # 3、车辆三维中心点坐标(int)
                veh_centre_data = obj.find('veh_loc_2d').text.split()
                veh_centre_data = [int(loc) for loc in veh_centre_data]
                self.all_vehicle_location.append(veh_centre_data)
                # 4、车辆三维框顶点坐标(int)
                veh_vertex_data = []  # for each box (8 vertex)
                box_2dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex2d').text)
                for x in box_2dvertex:
                    veh_vertex_data += [int(item) for item in x.split(", ")]  # 合并为[x1,y1,x2,y2,...,x8,y8]的形式
                self.all_3dbbox_2dvertex.append(veh_vertex_data)
                # 车辆三维框3D顶点坐标(float)
                veh_3dvertex_data = []  # for each box (8 vertex)
                box_3dvertex = re.findall(r'[(](.*?)[)]', obj.find('vertex3d').text)
                for x in box_3dvertex:
                    veh_3dvertex_data += [float(item) for item in x.split(", ")]  # 合并为[x1,y1,x2,y2,...,x8,y8]的形式
                self.all_3dbbox_3dvertex.append(veh_3dvertex_data)

                # 5、车辆三维尺寸(float, m)
                veh_size_data = obj.find('veh_size').text.split()
                veh_size_data = [float(size) for size in veh_size_data]
                self.all_vehicle_size.append(veh_size_data)
                # 6、视角(left, right)
                veh_view_data = obj.find('perspective').text
                self.all_perspective.append(veh_view_data)
                # 7、车辆基准点坐标(int)
                veh_base_point_data = obj.find('base_point').text.split()
                veh_base_point_data = [int(base_point) for base_point in veh_base_point_data]
                self.all_base_point.append(veh_base_point_data)

                # veh_vertex_data -> list(元组)
                tp_veh_vertex_data = []
                for i in range(0, len(veh_vertex_data)-1, 2):
                    tp_veh_vertex_data.append((veh_vertex_data[i], veh_vertex_data[i+1]))
                self.all_3dbbox_2dvertex[idx] = tp_veh_vertex_data

                tp_veh_3dvertex_data = []
                for i in range(0, len(veh_3dvertex_data)-1, 2):
                    tp_veh_3dvertex_data.append((veh_3dvertex_data[i], veh_3dvertex_data[i+1]))
                self.all_3dbbox_3dvertex[idx] = tp_veh_3dvertex_data

                # veh_centre_data -> 元组
                tp_veh_centre_data = (int(veh_centre_data[0]), int(veh_centre_data[1]))

                # 绘制2D box
                left, top, right, bottom = int(bbox2d_data[0]), int(bbox2d_data[1]), int(bbox2d_data[0]) + int(bbox2d_data[2]), int(bbox2d_data[1]) + int(bbox2d_data[3])
                cv.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 3)
                label = '%s:%.2f' % (veh_type_data.lower(), 1.00)

                # Display the label at the top of the bounding box
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(top, labelSize[1])
                cv.rectangle(self.frame, (left, top - round(1.5 * labelSize[1])),
                             (left + round(1.5 * labelSize[0]), top + baseLine),
                             (255, 255, 255), cv.FILLED)
                cv.putText(self.frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

                # 绘制3D box, 并显示至label
                drawbox_img = self.paint(self.frame, tp_veh_vertex_data, tp_veh_centre_data)

            self.show_img_in_label(self.label_ImageDisplay, drawbox_img)

        else:
            # 调用目标检测, 并显示结果
            self.thread.Init(self.frame)
            self.thread.send_detect_result.connect(self.image_label_display)
            self.thread.start()

    def spin_cur_anno_order(self):
        '''选择标注对象'''
        # 如果有标注文件, 可以修改标注过的数据
        if self.frame is not None:
            if os.path.exists(self.select_file_xml):
                self.obj_num = len(self.all_vehicle_size)
                if (self.spinBox_CurAnnNum.value() + 1) <= self.obj_num and self.spinBox_CurAnnNum.value() >= 0:
                    self.spinBox_CurAnnNum.setMaximum(self.obj_num - 1)  # 最大只能设置到目标数
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

            else: # 重新标注
                try:
                    if (self.spinBox_CurAnnNum.value() + 1) <= self.obj_num and self.spinBox_CurAnnNum.value() >= 0:
                        self.spinBox_CurAnnNum.setMaximum(self.obj_num - 1)  # 最大只能设置到目标数
                        self.veh_box = self.list_box[self.spinBox_CurAnnNum.value()]
                        self.veh_type = self.list_type[self.spinBox_CurAnnNum.value()]
                        self.veh_conf = self.list_conf[self.spinBox_CurAnnNum.value()]
                        self.comboBox_CurAnnType.setCurrentIndex(dict_map_order_str[self.veh_type])
                        # 每次开始标注新对象时, 将滚动条复位
                        self.horizontalSlider_BasePointAdj_LR.setValue(0)
                        self.verticalSlider_BasePointAdj_UD.setValue(0)
                        self.horizontalSlider_VPAdj_LR.setValue(0)
                        if self.perspective == 'left':
                            self.base_point = (self.veh_box[0], self.veh_box[1] + self.veh_box[3])
                            self.veh_base_point = self.base_point
                        elif self.perspective == 'right':  # right
                            self.base_point = (self.veh_box[0] + self.veh_box[2], self.veh_box[1] + self.veh_box[3])
                            self.veh_base_point = self.base_point
                        self.drawbox_img = self.draw_3dbox(self.frame)

                except:
                    QMessageBox.information(self, "Information", "Please choose one vehicle! ",
                                            QMessageBox.Yes | QMessageBox.No)



    def radio_bp_left(self):
        '''选择3d bbox 包围基准点方向: 左下角'''
        try:
            self.perspective = "left"
            self.base_point = (self.veh_box[0], self.veh_box[1] + self.veh_box[3])
            self.veh_base_point = self.base_point
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def radio_bp_right(self):
        '''选择3d bbox 包围基准点方向: 右下角'''
        try:
            self.perspective = "right"
            self.base_point = (self.veh_box[0] + self.veh_box[2], self.veh_box[1] + self.veh_box[3])
            self.veh_base_point = self.base_point
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_bp_adjust_lr(self):
        '''手动调整3d bbox 基准点位置: 左右'''
        try:
            self.veh_base_point = (self.base_point[0] + self.horizontalSlider_BasePointAdj_LR.value(), self.veh_base_point[1])
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_bp_adjust_ud(self):
        '''手动调整3d bbox 基准点位置: 上下'''
        try:
            self.veh_base_point = (self.veh_base_point[0], self.base_point[1] + self.verticalSlider_BasePointAdj_UD.value())
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def slider_vp_adjust_lr(self):
        '''移动vpx,由地平线控制vpy的变化'''
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
        '''保存单个车辆标注结果'''
        # 如果有标注文件, 可以修改标注过的数据
        if self.frame is not None:
            if os.path.exists(self.select_file_xml):
                self.frame = self.drawbox_img

                self.all_3dbbox_2dvertex[self.spinBox_CurAnnNum.value()] = self.list_3dbbox_2dvertex
                self.all_vehicle_type[self.spinBox_CurAnnNum.value()] = self.comboBox_CurAnnType.currentText()
                self.all_vehicle_size[self.spinBox_CurAnnNum.value()] = [self.doubleSpinBox_Bbox3D_Length.value(), self.doubleSpinBox_Bbox3D_Width.value(),self.doubleSpinBox_Bbox3D_Height.value()]
                self.all_perspective[self.spinBox_CurAnnNum.value()] = self.perspective
                self.all_base_point[self.spinBox_CurAnnNum.value()] = [self.list_3dbbox_2dvertex[1][0], self.list_3dbbox_2dvertex[1][1]]  # 基准点p1
                self.all_3dbbox_3dvertex[self.spinBox_CurAnnNum.value()] = self.list_3dbbox_3dvertex
                self.all_vehicle_location[self.spinBox_CurAnnNum.value()] = [self.centroid_2d[0], self.centroid_2d[1]]

            else:
                # 如果没有清空过，则只要有信息、并且信息不和上一次的重复即可添加
                # 如果清空过，并且all信息列表为空，则不保存
                if self.veh_box:
                    try:  # 找到则不添加
                        index = self.all_veh_2dbbox.index(self.veh_box)
                        QMessageBox.information(self, "提示", "repeated element!", QMessageBox.Yes)
                    except:  # 找不到再添加
                        self.frame = self.drawbox_img
                        self.all_veh_2dbbox.append(self.veh_box)
                        self.all_3dbbox_2dvertex.append(self.list_3dbbox_2dvertex)
                        self.all_vehicle_type.append(self.comboBox_CurAnnType.currentText())
                        self.all_vehicle_size.append([self.doubleSpinBox_Bbox3D_Length.value(), self.doubleSpinBox_Bbox3D_Width.value(), self.doubleSpinBox_Bbox3D_Height.value()])
                        self.all_perspective.append(self.perspective)
                        self.all_base_point.append(self.list_3dbbox_2dvertex[1])  # 基准点p1
                        self.all_3dbbox_3dvertex.append(self.list_3dbbox_3dvertex)
                        self.all_vehicle_location.append(self.centroid_2d)
                        self.all_veh_conf.append(self.veh_conf)
                        self.pushButton_ClearAnnotations.setEnabled(True)
                else:
                    return

    def save_annotation_results(self):
        '''保存单个图像车辆标注结果'''
        if self.frame is not None:
            if self.all_3dbbox_2dvertex:
                # 保存标注图像
                cv.imwrite(self.select_file[0:len(self.select_file)-4] + "_drawbbox_result.bmp", self.frame)
                xml_path = self.select_file[0:len(self.select_file)-4] + ".xml"  # xml与图像同名
                save3dbbox_result(xml_path, self.select_file, self.calib_file_path, self.frame, self.all_veh_2dbbox, self.all_vehicle_type, self.all_3dbbox_2dvertex,
                self.all_vehicle_size, self.all_perspective, self.all_base_point, self.all_3dbbox_3dvertex, self.all_vehicle_location)
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
            # self.pushButton_ClearAnnotations.setEnabled(False)
            # 如果未保存，但是按了清空，下面条件不满足，但是上面参数还在，按保存仍然会保存进去，因此要清空上面一些变量
            # 如果保存了，同时按了清空，下面条件满足，弹出最后存进去的信息，不作保存
            if len(self.all_vehicle_size) > 0:
                self.veh_box = []  # fix 清空当前帧已在图像中清空的所选目标，防止其加入到list中
                self.all_veh_2dbbox.pop(-1)  # fix
                self.all_3dbbox_2dvertex.pop(-1)
                self.all_vehicle_type.pop(-1)  # fix
                self.all_vehicle_size.pop(-1)
                self.all_perspective.pop(-1)
                self.all_base_point.pop(-1) # 基准点p1
                self.all_3dbbox_3dvertex.pop(-1)
                self.all_vehicle_location.pop(-1)
                self.all_veh_conf.pop(-1)  # fix
                for i in range(len(self.list_box)):
                    # 绘制2D box
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
                for i in range(len(self.all_vehicle_size)):  # 清空当前车辆数后的列表
                    imgcopy = self.paint(frame_copy_, self.all_3dbbox_2dvertex[i], (self.all_vehicle_location[i][0], self.all_vehicle_location[i][1]))
                self.frame = frame_copy_
                self.spinBox_CurAnnNum.setValue(self.spinBox_CurAnnNum.value() - 1)
                self.show_img_in_label(self.label_ImageDisplay, frame_copy_)
            else:
                for i in range(len(self.list_box)):
                    # 绘制2D box
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
        '''调节车辆长度'''
        try:
            self.l = self.doubleSpinBox_Bbox3D_Length.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def spind_3dbbox_width(self):
        '''调节车辆宽度'''
        try:
            self.w = self.doubleSpinBox_Bbox3D_Width.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)

    def spind_3dbbox_height(self):
        '''调节车辆高度'''
        try:
            self.h = self.doubleSpinBox_Bbox3D_Height.value() * 1000.0
            self.drawbox_img = self.draw_3dbox(self.frame)
        except:
            QMessageBox.information(self, "Information", "Please choose one vehicle! ", QMessageBox.Yes | QMessageBox.No)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    print(app.exec_())  # 调试时可以用于打印错误
    sys.exit(app.exec_())
