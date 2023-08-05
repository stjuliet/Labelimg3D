from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDesktopWidget, QFileDialog
from sub_dialogs.pretrain_model_3d import Ui_MainWindow
from PIL import Image
import cv2 as cv
import numpy as np
import os


class Main_Pretrain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Main_Pretrain, self).__init__(parent)
        self.setupUi(self)
        self.center()  # center
        self.model_path = None
        self.model = None
        self.calib_path = None
        self.image_dir = None
        self.image_list = []
        self.frame_cnt = 0
        self.annotation_dir = None

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def on_pushbutton_config_model_path(self):
        """ config model path """
        self.model_path, _ = QFileDialog.getOpenFileName(self, "choose model path", os.path.join(os.getcwd(), "model_data"),
                                                         "models (*.pth);;All (*)")
        if not self.model_path:
            QMessageBox.warning(self, "info", "please choose again!", QMessageBox.Ok)
            return
        else:
            str_index = self.model_path.index("model_data")
            relative_model_path = self.model_path[str_index:]
            self.textEdit_model_path.setText(relative_model_path)
            QMessageBox.warning(self, "info", "model path config successfully！", QMessageBox.Ok)

    def on_pushbutton_config_image_dir(self):
        """ config image dir """
        self.image_dir = QFileDialog.getExistingDirectory(self, "choose image dir", os.getcwd())
        if self.image_dir:
            self.textEdit_image_dir.setText(self.image_dir)
            self.image_list = os.listdir(self.image_dir)
            QMessageBox.warning(self, "info", "image dir config successfully！", QMessageBox.Ok)

    def on_pushbutton_config_annotation_dir(self):
        """ config annotation dir """
        self.annotation_dir = QFileDialog.getExistingDirectory(self, "choose annotation dir", os.getcwd())
        if self.annotation_dir:
            self.textEdit_annotation_dir.setText(self.annotation_dir)
            QMessageBox.warning(self, "info", "annotation dir config successfully！", QMessageBox.Ok)

    def on_pushbutton_load_model(self):
        """
        load model
        :return:
        """
        try:
            if self.model_path:
                if self.comboBox_model_name.currentText() == "CenterLoc3D":
                    from pretrain_model_3d.centerloc3d.predict.box_predict import Bbox3dPred
                    self.model = Bbox3dPred(self.model_path)
                elif self.comboBox_model_name.currentText() == "CenterDet3D":
                    pass
            else:
                QMessageBox.warning(self, "info", "please config model path first!", QMessageBox.Ok)
                return
        except:
            QMessageBox.warning(self, "info", "not aligned! please choose model name again!", QMessageBox.Ok)
            return

    def on_pushbutton_generate_annotations(self):
        """ generate annotations """
        # TODO: unfinished
        if len(self.image_list) != 0:
            for image_id in self.image_list:
                if image_id.endswith(".jpg"):
                    img = cv.imdecode(np.fromfile(os.path.join(self.image_dir, self.img_list[self.frame_cnt]), np.uint8), cv.IMREAD_COLOR)
                    image = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    if self.comboBox_dataset.currentText() == "common":
                        self.calib_path = os.path.join(self.image_dir, os.listdir(os.path.join(self.image_dir, "calib"))[0])
                    elif self.comboBox_dataset.currentText() == "cross":
                        self.calib_path = os.path.join(self.image_dir, "calib", self.img_list[self.frame_cnt][:-4] + "_calib.xml")
                    xml_path = os.path.join(self.annotation_dir, self.img_list[self.frame_cnt][:-4] + ".xml")
                    r_image, _, proc_time = self.model.detect_image(image, xml_path,
                                                                    os.path.join(self.image_dir, self.img_list[self.frame_cnt]), img,
                                                                    self.calib_path, True)
                    self.frame_cnt += 1
