# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bbox2d_anno.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(567, 368)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(150, 290, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(130, 70, 310, 194))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_type = QtWidgets.QLabel(self.widget)
        self.label_type.setObjectName("label_type")
        self.horizontalLayout.addWidget(self.label_type)
        self.listView_type = QtWidgets.QListView(self.widget)
        self.listView_type.setEnabled(True)
        self.listView_type.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.listView_type.setTabKeyNavigation(False)
        self.listView_type.setObjectName("listView_type")
        self.horizontalLayout.addWidget(self.listView_type)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_type.setText(_translate("Dialog", "类别："))

