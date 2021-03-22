import numpy as np
import cv2 as cv
import time
# 使用pyqt多线程、信号槽机制实现目标检测，防止界面卡死
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage


'''
读入目标检测类型文件，返回类别数组
'''



def read_class(file):
    with open(file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes


'''
目标检测类(继承多线程)
'''


class YoloDetect(QThread):
    send_detect_result = pyqtSignal(object)

    def __init__(self, class_path, modelConfiguration, modelWeights):
        super(YoloDetect, self).__init__()
        self.classes = read_class(class_path)
        self.net_input_width = 416
        self.net_input_height = 416
        self.confThreshold = 0.3
        self.nmsThreshold = 0.5
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)  # CUDA: DNN_BACKEND_CUDA
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)       # CUDA: DNN_TARGET_CUDA


    def Init(self, frame):
        self.frame = frame


    # 获得输出层名
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 绘制bbox
    def drawPred(self, frame, classes, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                         (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # 后处理
    def postprocess(self, frame, classes, outs):
        '''
        Remove the bounding boxes with low confidence using non-maximum suppression
        :param frame:
        :param classes:
        :return: outs:[507*85 =(13*13*3)*(5+80),
                 2028*85=(26*26*3)*(5+80),
                 8112*85=(52*52*3)*(5+80)]
        outs中每一行是一个预测值：[x,y,w,h,confs,class_probs_0,class_probs_1,..,class_probs_78,class_probs_79]
        :return:
        '''
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []    # 保存对应二维框
        list_type = []  # 保存类型
        nms_boxes = []
        nms_list_type = []
        nms_list_conf = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                if confidence > self.confThreshold and (classId == 0 or classId == 1 or classId == 2):
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    classIds.append(classId)
                    list_type.append(classes[classId])
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            nms_boxes.append([left, top, width, height])
            nms_list_type.append(list_type[i])
            nms_list_conf.append(confidences[i])
            self.drawPred(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)
        return nms_boxes, nms_list_type, nms_list_conf

    # 前向传播
    def cv_dnn_forward(self, frame):
        '''
        :param frame:
        :return: outs:[507*85 =13*13*3*(5+80),
                       2028*85=26*26*3*(5+80),
                       8112*85=52*52*3*(5+80)]
        '''
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.net_input_width, self.net_input_height), [0, 0, 0], 1,
                                    crop=False)
        # Sets the input to the network
        self.net.setInput(blob)
        # print(list_net[0])
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames(self.net))
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        # runtime, _ = self.net.getPerfProfile()
        return outs

    # 预测总函数
    def yolov4_predict(self, frame):
        t1 = time.time()
        outs = self.cv_dnn_forward(frame)
        # Remove the bounding boxes with low confidence
        list_box, list_type, list_conf = self.postprocess(frame, self.classes, outs)
        t2 = time.time()

        fps = 'FPS: %.2f' % (1. / (t2 - t1))
        # label = 'Inference time: %.2f ms' % (runtime * 1000.0 / cv.getTickFrequency())
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # cv.putText(frame, fps, (0, 40), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
        return frame, list_box, list_type, list_conf


    # 多线程start调用函数
    def run(self):
        detect_frame, list_box, list_type, list_conf = self.yolov4_predict(self.frame)
        # 耗时转换图像操作放在run函数中运行，传出pixmap信号，可直接显示
        detect_frame = cv.cvtColor(detect_frame, cv.COLOR_BGR2RGB)
        qimage = QImage(detect_frame.data, detect_frame.shape[1], detect_frame.shape[0], detect_frame.shape[1] * 3, QImage.Format_RGB888)
        detect_pixmap = QPixmap.fromImage(qimage)
        if detect_frame is not None:
            self.send_detect_result.emit([detect_pixmap, list_box, list_type, list_conf])  # 检测完成传送信号，触发槽函数接收并显示检测结果
        else:
            return


