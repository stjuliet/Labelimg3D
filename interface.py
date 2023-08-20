# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1873, 985)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resource/label3d_3.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 1841, 921))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.groupBox_ImageDisplay = QtWidgets.QGroupBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_ImageDisplay.sizePolicy().hasHeightForWidth())
        self.groupBox_ImageDisplay.setSizePolicy(sizePolicy)
        self.groupBox_ImageDisplay.setObjectName("groupBox_ImageDisplay")
        self.label_ImageDisplay = QtWidgets.QLabel(self.groupBox_ImageDisplay)
        self.label_ImageDisplay.setGeometry(QtCore.QRect(10, 20, 1351, 891))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_ImageDisplay.sizePolicy().hasHeightForWidth())
        self.label_ImageDisplay.setSizePolicy(sizePolicy)
        self.label_ImageDisplay.setText("")
        self.label_ImageDisplay.setObjectName("label_ImageDisplay")
        self.horizontalLayout_9.addWidget(self.groupBox_ImageDisplay)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox_AnnOption = QtWidgets.QGroupBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_AnnOption.sizePolicy().hasHeightForWidth())
        self.groupBox_AnnOption.setSizePolicy(sizePolicy)
        self.groupBox_AnnOption.setObjectName("groupBox_AnnOption")
        self.layoutWidget1 = QtWidgets.QWidget(self.groupBox_AnnOption)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 73, 262, 21))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_BasePointSet = QtWidgets.QLabel(self.layoutWidget1)
        self.label_BasePointSet.setObjectName("label_BasePointSet")
        self.horizontalLayout_2.addWidget(self.label_BasePointSet)
        self.radioButton_BasePointLeft = QtWidgets.QRadioButton(self.layoutWidget1)
        self.radioButton_BasePointLeft.setChecked(True)
        self.radioButton_BasePointLeft.setObjectName("radioButton_BasePointLeft")
        self.horizontalLayout_2.addWidget(self.radioButton_BasePointLeft)
        self.radioButton_BasePointRight = QtWidgets.QRadioButton(self.layoutWidget1)
        self.radioButton_BasePointRight.setEnabled(True)
        self.radioButton_BasePointRight.setChecked(False)
        self.radioButton_BasePointRight.setObjectName("radioButton_BasePointRight")
        self.horizontalLayout_2.addWidget(self.radioButton_BasePointRight)
        self.layoutWidget2 = QtWidgets.QWidget(self.groupBox_AnnOption)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 99, 421, 86))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_BasePointAdjust = QtWidgets.QLabel(self.layoutWidget2)
        self.label_BasePointAdjust.setObjectName("label_BasePointAdjust")
        self.horizontalLayout_3.addWidget(self.label_BasePointAdjust)
        self.horizontalSlider_BasePointAdj_LR = QtWidgets.QSlider(self.layoutWidget2)
        self.horizontalSlider_BasePointAdj_LR.setMouseTracking(False)
        self.horizontalSlider_BasePointAdj_LR.setMinimum(-400)
        self.horizontalSlider_BasePointAdj_LR.setMaximum(400)
        self.horizontalSlider_BasePointAdj_LR.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_BasePointAdj_LR.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_BasePointAdj_LR.setObjectName("horizontalSlider_BasePointAdj_LR")
        self.horizontalLayout_3.addWidget(self.horizontalSlider_BasePointAdj_LR)
        self.verticalSlider_BasePointAdj_UD = QtWidgets.QSlider(self.layoutWidget2)
        self.verticalSlider_BasePointAdj_UD.setMinimum(-200)
        self.verticalSlider_BasePointAdj_UD.setMaximum(200)
        self.verticalSlider_BasePointAdj_UD.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_BasePointAdj_UD.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.verticalSlider_BasePointAdj_UD.setObjectName("verticalSlider_BasePointAdj_UD")
        self.horizontalLayout_3.addWidget(self.verticalSlider_BasePointAdj_UD)
        self.layoutWidget3 = QtWidgets.QWidget(self.groupBox_AnnOption)
        self.layoutWidget3.setGeometry(QtCore.QRect(10, 191, 421, 41))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_VPAdjust = QtWidgets.QLabel(self.layoutWidget3)
        self.label_VPAdjust.setObjectName("label_VPAdjust")
        self.horizontalLayout_4.addWidget(self.label_VPAdjust)
        self.horizontalSlider_VPAdj_LR = QtWidgets.QSlider(self.layoutWidget3)
        self.horizontalSlider_VPAdj_LR.setMinimum(-200)
        self.horizontalSlider_VPAdj_LR.setMaximum(200)
        self.horizontalSlider_VPAdj_LR.setSingleStep(1)
        self.horizontalSlider_VPAdj_LR.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_VPAdj_LR.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_VPAdj_LR.setObjectName("horizontalSlider_VPAdj_LR")
        self.horizontalLayout_4.addWidget(self.horizontalSlider_VPAdj_LR)
        self.layoutWidget4 = QtWidgets.QWidget(self.groupBox_AnnOption)
        self.layoutWidget4.setGeometry(QtCore.QRect(10, 45, 301, 23))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget4)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_CurAnnNum = QtWidgets.QLabel(self.layoutWidget4)
        self.label_CurAnnNum.setObjectName("label_CurAnnNum")
        self.horizontalLayout.addWidget(self.label_CurAnnNum)
        self.spinBox_CurAnnNum = QtWidgets.QSpinBox(self.layoutWidget4)
        self.spinBox_CurAnnNum.setReadOnly(False)
        self.spinBox_CurAnnNum.setMinimum(-1)
        self.spinBox_CurAnnNum.setMaximum(50)
        self.spinBox_CurAnnNum.setProperty("value", -1)
        self.spinBox_CurAnnNum.setObjectName("spinBox_CurAnnNum")
        self.horizontalLayout.addWidget(self.spinBox_CurAnnNum)
        self.label_CurAnnType = QtWidgets.QLabel(self.layoutWidget4)
        self.label_CurAnnType.setObjectName("label_CurAnnType")
        self.horizontalLayout.addWidget(self.label_CurAnnType)
        self.comboBox_CurAnnType = QtWidgets.QComboBox(self.layoutWidget4)
        self.comboBox_CurAnnType.setEditable(False)
        self.comboBox_CurAnnType.setObjectName("comboBox_CurAnnType")
        self.comboBox_CurAnnType.addItem("")
        self.comboBox_CurAnnType.setItemText(0, "")
        self.comboBox_CurAnnType.addItem("")
        self.comboBox_CurAnnType.addItem("")
        self.comboBox_CurAnnType.addItem("")
        self.comboBox_CurAnnType.addItem("")
        self.comboBox_CurAnnType.addItem("")
        self.comboBox_CurAnnType.addItem("")
        self.horizontalLayout.addWidget(self.comboBox_CurAnnType)
        self.layoutWidget5 = QtWidgets.QWidget(self.groupBox_AnnOption)
        self.layoutWidget5.setGeometry(QtCore.QRect(157, 425, 295, 30))
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.layoutWidget5)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pushButton_ClearAnnotations = QtWidgets.QPushButton(self.layoutWidget5)
        self.pushButton_ClearAnnotations.setObjectName("pushButton_ClearAnnotations")
        self.horizontalLayout_6.addWidget(self.pushButton_ClearAnnotations)
        self.pushButton_SaveTempAnnotation = QtWidgets.QPushButton(self.layoutWidget5)
        self.pushButton_SaveTempAnnotation.setObjectName("pushButton_SaveTempAnnotation")
        self.horizontalLayout_6.addWidget(self.pushButton_SaveTempAnnotation)
        self.pushButton_SaveAnnotations = QtWidgets.QPushButton(self.layoutWidget5)
        self.pushButton_SaveAnnotations.setObjectName("pushButton_SaveAnnotations")
        self.horizontalLayout_6.addWidget(self.pushButton_SaveAnnotations)
        self.layoutWidget6 = QtWidgets.QWidget(self.groupBox_AnnOption)
        self.layoutWidget6.setGeometry(QtCore.QRect(10, 20, 271, 21))
        self.layoutWidget6.setObjectName("layoutWidget6")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.layoutWidget6)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_ObjNums = QtWidgets.QLabel(self.layoutWidget6)
        self.label_ObjNums.setObjectName("label_ObjNums")
        self.horizontalLayout_7.addWidget(self.label_ObjNums)
        self.textEdit_ObjNums = QtWidgets.QTextEdit(self.layoutWidget6)
        self.textEdit_ObjNums.setReadOnly(True)
        self.textEdit_ObjNums.setObjectName("textEdit_ObjNums")
        self.horizontalLayout_7.addWidget(self.textEdit_ObjNums)
        self.layoutWidget7 = QtWidgets.QWidget(self.groupBox_AnnOption)
        self.layoutWidget7.setGeometry(QtCore.QRect(10, 373, 421, 47))
        self.layoutWidget7.setObjectName("layoutWidget7")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget7)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_Bbox3DSize = QtWidgets.QLabel(self.layoutWidget7)
        self.label_Bbox3DSize.setObjectName("label_Bbox3DSize")
        self.verticalLayout.addWidget(self.label_Bbox3DSize)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_Bbox3D_Length = QtWidgets.QLabel(self.layoutWidget7)
        self.label_Bbox3D_Length.setObjectName("label_Bbox3D_Length")
        self.horizontalLayout_5.addWidget(self.label_Bbox3D_Length)
        self.doubleSpinBox_Bbox3D_Length = QtWidgets.QDoubleSpinBox(self.layoutWidget7)
        self.doubleSpinBox_Bbox3D_Length.setSingleStep(0.01)
        self.doubleSpinBox_Bbox3D_Length.setProperty("value", 4.5)
        self.doubleSpinBox_Bbox3D_Length.setObjectName("doubleSpinBox_Bbox3D_Length")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_Bbox3D_Length)
        self.label_Bbox3D_Width = QtWidgets.QLabel(self.layoutWidget7)
        self.label_Bbox3D_Width.setObjectName("label_Bbox3D_Width")
        self.horizontalLayout_5.addWidget(self.label_Bbox3D_Width)
        self.doubleSpinBox_Bbox3D_Width = QtWidgets.QDoubleSpinBox(self.layoutWidget7)
        self.doubleSpinBox_Bbox3D_Width.setSingleStep(0.01)
        self.doubleSpinBox_Bbox3D_Width.setProperty("value", 1.8)
        self.doubleSpinBox_Bbox3D_Width.setObjectName("doubleSpinBox_Bbox3D_Width")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_Bbox3D_Width)
        self.label_Bbox3D_Height = QtWidgets.QLabel(self.layoutWidget7)
        self.label_Bbox3D_Height.setObjectName("label_Bbox3D_Height")
        self.horizontalLayout_5.addWidget(self.label_Bbox3D_Height)
        self.doubleSpinBox_Bbox3D_Height = QtWidgets.QDoubleSpinBox(self.layoutWidget7)
        self.doubleSpinBox_Bbox3D_Height.setSingleStep(0.01)
        self.doubleSpinBox_Bbox3D_Height.setProperty("value", 1.5)
        self.doubleSpinBox_Bbox3D_Height.setObjectName("doubleSpinBox_Bbox3D_Height")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_Bbox3D_Height)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.layoutWidget8 = QtWidgets.QWidget(self.groupBox_AnnOption)
        self.layoutWidget8.setGeometry(QtCore.QRect(10, 237, 421, 131))
        self.layoutWidget8.setObjectName("layoutWidget8")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.layoutWidget8)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_RotAdjust = QtWidgets.QLabel(self.layoutWidget8)
        self.label_RotAdjust.setObjectName("label_RotAdjust")
        self.horizontalLayout_8.addWidget(self.label_RotAdjust)
        self.dial_Bbox3D_Rot = QtWidgets.QDial(self.layoutWidget8)
        self.dial_Bbox3D_Rot.setMinimum(0)
        self.dial_Bbox3D_Rot.setMaximum(360)
        self.dial_Bbox3D_Rot.setPageStep(5)
        self.dial_Bbox3D_Rot.setProperty("value", 0)
        self.dial_Bbox3D_Rot.setSliderPosition(0)
        self.dial_Bbox3D_Rot.setOrientation(QtCore.Qt.Horizontal)
        self.dial_Bbox3D_Rot.setInvertedAppearance(False)
        self.dial_Bbox3D_Rot.setInvertedControls(False)
        self.dial_Bbox3D_Rot.setWrapping(True)
        self.dial_Bbox3D_Rot.setNotchTarget(0.0)
        self.dial_Bbox3D_Rot.setNotchesVisible(True)
        self.dial_Bbox3D_Rot.setObjectName("dial_Bbox3D_Rot")
        self.horizontalLayout_8.addWidget(self.dial_Bbox3D_Rot)
        self.doubleSpinBox_Bbox3D_Rot = QtWidgets.QDoubleSpinBox(self.layoutWidget8)
        self.doubleSpinBox_Bbox3D_Rot.setMaximum(360.0)
        self.doubleSpinBox_Bbox3D_Rot.setSingleStep(0.1)
        self.doubleSpinBox_Bbox3D_Rot.setProperty("value", 0.0)
        self.doubleSpinBox_Bbox3D_Rot.setObjectName("doubleSpinBox_Bbox3D_Rot")
        self.horizontalLayout_8.addWidget(self.doubleSpinBox_Bbox3D_Rot)
        self.pushButton_add_new_bbox2d = QtWidgets.QPushButton(self.groupBox_AnnOption)
        self.pushButton_add_new_bbox2d.setGeometry(QtCore.QRect(320, 10, 111, 31))
        self.pushButton_add_new_bbox2d.setObjectName("pushButton_add_new_bbox2d")
        self.pushButton_update_new_bbox2d = QtWidgets.QPushButton(self.groupBox_AnnOption)
        self.pushButton_update_new_bbox2d.setEnabled(False)
        self.pushButton_update_new_bbox2d.setGeometry(QtCore.QRect(320, 70, 111, 31))
        self.pushButton_update_new_bbox2d.setObjectName("pushButton_update_new_bbox2d")
        self.pushButton_cancel_add_new_bbox2d = QtWidgets.QPushButton(self.groupBox_AnnOption)
        self.pushButton_cancel_add_new_bbox2d.setGeometry(QtCore.QRect(320, 43, 111, 23))
        self.pushButton_cancel_add_new_bbox2d.setObjectName("pushButton_cancel_add_new_bbox2d")
        self.verticalLayout_3.addWidget(self.groupBox_AnnOption)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_VehSize = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_VehSize.setObjectName("groupBox_VehSize")
        self.listView_VehSize = QtWidgets.QListView(self.groupBox_VehSize)
        self.listView_VehSize.setGeometry(QtCore.QRect(10, 20, 441, 121))
        self.listView_VehSize.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.listView_VehSize.setObjectName("listView_VehSize")
        self.verticalLayout_2.addWidget(self.groupBox_VehSize)
        self.groupBox_FileList = QtWidgets.QGroupBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_FileList.sizePolicy().hasHeightForWidth())
        self.groupBox_FileList.setSizePolicy(sizePolicy)
        self.groupBox_FileList.setObjectName("groupBox_FileList")
        self.listView_FileList = QtWidgets.QListView(self.groupBox_FileList)
        self.listView_FileList.setGeometry(QtCore.QRect(10, 20, 441, 271))
        self.listView_FileList.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.listView_FileList.setObjectName("listView_FileList")
        self.verticalLayout_2.addWidget(self.groupBox_FileList)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_9.addLayout(self.verticalLayout_3)
        self.horizontalLayout_9.setStretch(0, 3)
        self.horizontalLayout_9.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1873, 23))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuRevise_Mode = QtWidgets.QMenu(self.menubar)
        self.menuRevise_Mode.setObjectName("menuRevise_Mode")
        self.menuConfig = QtWidgets.QMenu(self.menubar)
        self.menuConfig.setObjectName("menuConfig")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_Folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_Folder.setObjectName("actionOpen_Folder")
        self.action_open_folder = QtWidgets.QAction(MainWindow)
        self.action_open_folder.setObjectName("action_open_folder")
        self.actionkeypoint_only = QtWidgets.QAction(MainWindow)
        self.actionkeypoint_only.setCheckable(True)
        self.actionkeypoint_only.setChecked(False)
        self.actionkeypoint_only.setObjectName("actionkeypoint_only")
        self.actionvehicle_size = QtWidgets.QAction(MainWindow)
        self.actionvehicle_size.setCheckable(False)
        self.actionvehicle_size.setEnabled(True)
        self.actionvehicle_size.setObjectName("actionvehicle_size")
        self.actionpretrain_model_3d = QtWidgets.QAction(MainWindow)
        self.actionpretrain_model_3d.setObjectName("actionpretrain_model_3d")
        self.actionpedes = QtWidgets.QAction(MainWindow)
        self.actionpedes.setCheckable(True)
        self.actionpedes.setChecked(False)
        self.actionpedes.setObjectName("actionpedes")
        self.menuMenu.addAction(self.action_open_folder)
        self.menuRevise_Mode.addAction(self.actionkeypoint_only)
        self.menuRevise_Mode.addAction(self.actionpedes)
        self.menuConfig.addAction(self.actionvehicle_size)
        self.menuConfig.addAction(self.actionpretrain_model_3d)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuRevise_Mode.menuAction())
        self.menubar.addAction(self.menuConfig.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButton_SaveAnnotations.clicked.connect(MainWindow.save_annotation_results) # type: ignore
        self.horizontalSlider_BasePointAdj_LR.valueChanged['int'].connect(MainWindow.slider_bp_adjust_lr) # type: ignore
        self.verticalSlider_BasePointAdj_UD.valueChanged['int'].connect(MainWindow.slider_bp_adjust_ud) # type: ignore
        self.horizontalSlider_VPAdj_LR.valueChanged['int'].connect(MainWindow.slider_vp_adjust_lr) # type: ignore
        self.spinBox_CurAnnNum.valueChanged['int'].connect(MainWindow.spin_cur_anno_order) # type: ignore
        self.comboBox_CurAnnType.currentIndexChanged['int'].connect(MainWindow.combo_cur_anno_type) # type: ignore
        self.radioButton_BasePointLeft.clicked.connect(MainWindow.radio_bp_left) # type: ignore
        self.radioButton_BasePointRight.clicked.connect(MainWindow.radio_bp_right) # type: ignore
        self.doubleSpinBox_Bbox3D_Length.valueChanged['double'].connect(MainWindow.spind_3dbbox_length) # type: ignore
        self.doubleSpinBox_Bbox3D_Width.valueChanged['double'].connect(MainWindow.spind_3dbbox_width) # type: ignore
        self.doubleSpinBox_Bbox3D_Height.valueChanged['double'].connect(MainWindow.spind_3dbbox_height) # type: ignore
        self.listView_FileList.doubleClicked['QModelIndex'].connect(MainWindow.listview_doubleclick_slot) # type: ignore
        self.pushButton_SaveTempAnnotation.clicked.connect(MainWindow.save_temp_annotation_results) # type: ignore
        self.pushButton_ClearAnnotations.clicked.connect(MainWindow.clear_single_annotation) # type: ignore
        self.action_open_folder.triggered.connect(MainWindow.choose_img_folder) # type: ignore
        self.actionvehicle_size.triggered.connect(MainWindow.config_vehicle_size) # type: ignore
        self.listView_VehSize.clicked['QModelIndex'].connect(MainWindow.transfer_anno_vehicle_size) # type: ignore
        self.listView_VehSize.doubleClicked['QModelIndex'].connect(MainWindow.remove_listview_vehsize_item) # type: ignore
        self.dial_Bbox3D_Rot.valueChanged['int'].connect(MainWindow.dial_box_rot_adjust) # type: ignore
        self.doubleSpinBox_Bbox3D_Rot.valueChanged['double'].connect(MainWindow.spind_3dbbox_rot) # type: ignore
        self.actionpretrain_model_3d.triggered.connect(MainWindow.config_pretrain_model_3d) # type: ignore
        self.pushButton_add_new_bbox2d.clicked.connect(MainWindow.add_new_bbox2d) # type: ignore
        self.pushButton_update_new_bbox2d.clicked.connect(MainWindow.update_new_bbox2d) # type: ignore
        self.pushButton_cancel_add_new_bbox2d.clicked.connect(MainWindow.cancel_add_new_bbox2d) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Labelimg-3D"))
        self.groupBox_ImageDisplay.setTitle(_translate("MainWindow", "Image Display"))
        self.groupBox_AnnOption.setTitle(_translate("MainWindow", "Annotation Options"))
        self.label_BasePointSet.setText(_translate("MainWindow", "Base Point Set:"))
        self.radioButton_BasePointLeft.setText(_translate("MainWindow", "Left"))
        self.radioButton_BasePointRight.setText(_translate("MainWindow", "Right"))
        self.label_BasePointAdjust.setText(_translate("MainWindow", "Base Point Adjust:"))
        self.label_VPAdjust.setText(_translate("MainWindow", "Vanishing Point Adjust:"))
        self.label_CurAnnNum.setText(_translate("MainWindow", "Current annotation:"))
        self.label_CurAnnType.setText(_translate("MainWindow", "Type:"))
        self.comboBox_CurAnnType.setItemText(1, _translate("MainWindow", "Car"))
        self.comboBox_CurAnnType.setItemText(2, _translate("MainWindow", "Truck"))
        self.comboBox_CurAnnType.setItemText(3, _translate("MainWindow", "Bus"))
        self.comboBox_CurAnnType.setItemText(4, _translate("MainWindow", "Vehicle"))
        self.comboBox_CurAnnType.setItemText(5, _translate("MainWindow", "Non-motor"))
        self.comboBox_CurAnnType.setItemText(6, _translate("MainWindow", "Pedestrian"))
        self.pushButton_ClearAnnotations.setText(_translate("MainWindow", "clear"))
        self.pushButton_SaveTempAnnotation.setText(_translate("MainWindow", "Save"))
        self.pushButton_SaveTempAnnotation.setShortcut(_translate("MainWindow", "Ctrl+A"))
        self.pushButton_SaveAnnotations.setText(_translate("MainWindow", "Save All"))
        self.pushButton_SaveAnnotations.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.label_ObjNums.setText(_translate("MainWindow", "Object numbers in current image:"))
        self.label_Bbox3DSize.setText(_translate("MainWindow", "3D bbox Size(m):"))
        self.label_Bbox3D_Length.setText(_translate("MainWindow", "Length:"))
        self.label_Bbox3D_Width.setText(_translate("MainWindow", "Width:"))
        self.label_Bbox3D_Height.setText(_translate("MainWindow", "Height:"))
        self.label_RotAdjust.setText(_translate("MainWindow", "Rotation Adjust(degree):"))
        self.pushButton_add_new_bbox2d.setText(_translate("MainWindow", "ADD 2D Bbox"))
        self.pushButton_update_new_bbox2d.setText(_translate("MainWindow", "UPDATE 2D Bbox"))
        self.pushButton_cancel_add_new_bbox2d.setText(_translate("MainWindow", "CANCEL ADD"))
        self.groupBox_VehSize.setTitle(_translate("MainWindow", "Vehicle Size"))
        self.groupBox_FileList.setTitle(_translate("MainWindow", "File List"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.menuRevise_Mode.setTitle(_translate("MainWindow", "Mode"))
        self.menuConfig.setTitle(_translate("MainWindow", "Config"))
        self.actionOpen_Folder.setText(_translate("MainWindow", "Open Folder..."))
        self.action_open_folder.setText(_translate("MainWindow", "Open Folder..."))
        self.actionkeypoint_only.setText(_translate("MainWindow", "keypoint"))
        self.actionvehicle_size.setText(_translate("MainWindow", "vehicle_size"))
        self.actionpretrain_model_3d.setText(_translate("MainWindow", "pretrain_model_3d"))
        self.actionpedes.setText(_translate("MainWindow", "pedes"))
