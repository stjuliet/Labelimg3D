# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog_vehicle_size.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(225, 187)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 140, 161, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(50, 30, 124, 80))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_l = QtWidgets.QLabel(self.layoutWidget)
        self.label_l.setObjectName("label_l")
        self.horizontalLayout.addWidget(self.label_l)
        self.doubleSpinBox_l = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.doubleSpinBox_l.setSingleStep(0.01)
        self.doubleSpinBox_l.setProperty("value", 4.5)
        self.doubleSpinBox_l.setObjectName("doubleSpinBox_l")
        self.horizontalLayout.addWidget(self.doubleSpinBox_l)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_w = QtWidgets.QLabel(self.layoutWidget)
        self.label_w.setObjectName("label_w")
        self.horizontalLayout_2.addWidget(self.label_w)
        self.doubleSpinBox_w = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.doubleSpinBox_w.setSingleStep(0.01)
        self.doubleSpinBox_w.setProperty("value", 1.8)
        self.doubleSpinBox_w.setObjectName("doubleSpinBox_w")
        self.horizontalLayout_2.addWidget(self.doubleSpinBox_w)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_h = QtWidgets.QLabel(self.layoutWidget)
        self.label_h.setObjectName("label_h")
        self.horizontalLayout_3.addWidget(self.label_h)
        self.doubleSpinBox_h = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.doubleSpinBox_h.setSingleStep(0.01)
        self.doubleSpinBox_h.setProperty("value", 1.5)
        self.doubleSpinBox_h.setObjectName("doubleSpinBox_h")
        self.horizontalLayout_3.addWidget(self.doubleSpinBox_h)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Config Vehicle Size"))
        self.label_l.setText(_translate("Dialog", "Length(m):"))
        self.label_w.setText(_translate("Dialog", "Width(m):"))
        self.label_h.setText(_translate("Dialog", "Height(m):"))
