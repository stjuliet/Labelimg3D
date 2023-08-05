# predict and decode results
import os
import time
import cv2 as cv
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn
from torch.autograd import Variable

from pretrain_model_3d.centerloc3d.nets.fpn import KeyPointDetection
from pretrain_model_3d.centerloc3d.nets.hourglass_official import HourglassNet, Bottleneck
from pretrain_model_3d.centerloc3d.utils.utils import *
from tools import save3dbbox_result

from PyQt5.QtCore import *


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


# use model to predict
# model_path、classes_path和backbone
class Bbox3dPred(QObject):
    _defaults = {
        "model_path"        : 'model_data/svld_3d/resnet50-Epoch99-ciou-Total_train_Loss1.5854-Val_Loss2.2824.pth',
        "classes_path"      : 'model_data/classes.txt',
        "backbone"          : "resnet50",
        "image_size"        : [512, 512, 3],
        "confidence"        : 0.3,
        # backbone: resnet50 - True
        # backbone: hourglass - False
        "nms"               : True,
        "nms_threhold"      : 0.5,
        "cuda"              : True,
        "letterbox_image"   : True   # suggested True
    }
    # 传送信息信号
    send_det_results = pyqtSignal(str)
    send_loc_results = pyqtSignal(object)

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, model_path):
        super(Bbox3dPred, self).__init__()
        self.__dict__.update(self._defaults)  # dict key -> class attr, use self.** to use
        self.model_path = model_path
        self.class_names = self._get_class()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # load model
    def generate(self):
        self.backbone_resnet_index = {"resnet18": 0, "resnet34": 1, "resnet50": 2, "resnet101": 3, "resnet152": 4}
        self.backbone_efficientnet_index = {"efficientnetb0": 0, "efficientnetb1": 1, "efficientnetb2": 2,
                     "efficientnetb3": 3, "efficientnetb4": 4, "efficientnetb5": 5, "efficientnetb6": 6, "efficientnetb7": 7}

        self.num_classes = len(self.class_names)

        # build model
        if self.backbone[:-2] == "resnet":
            self.model = KeyPointDetection(model_name=self.backbone[:-2], model_index=self.backbone_resnet_index[self.backbone], num_classes=self.num_classes)
        if self.backbone[:-2] == "efficientnet":
            self.model = KeyPointDetection(model_name=self.backbone[:-2], model_index=self.backbone_efficientnet_index[self.backbone], num_classes=self.num_classes)
        if self.backbone == "hourglass":
            self.model = HourglassNet(Bottleneck, num_stacks=8, num_blocks=1, num_classes=self.num_classes)

        # load weights
        # print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict["state_dict"], strict=True)
        # eval
        self.model = self.model.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.model.cuda()

        # print('{} model, classes loaded.'.format(self.model_path))

        # cls colors
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image, xml_path, image_path, np_img, calib_path=None, is_record_result=True, draw_convex=True):
        """
        image: PIL img
        """
        all_veh_2dbbox = []
        all_vehicle_type = []
        all_3dbbox_2dvertex = []
        all_vehicle_size = []
        all_vehicle_rots = []
        all_vehicle_location_3d = []
        all_perspective = []
        all_base_point = []
        all_3dbbox_3dvertex = []
        all_vehicle_location = []
        all_key_points = []

        image_shape = np.array(np.shape(image)[0:2])

        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.image_size[1], self.image_size[0])))
            crop_img = np.array(crop_img, dtype=np.float32)[:, :, ::-1]  # rgb -> bgr
        else:
            crop_img = np.array(image, dtype=np.float32)[:, :, ::-1]  # rgb -> bgr
            crop_img = cv.cvtColor(crop_img, cv.COLOR_RGB2BGR)
            crop_img = cv.resize(crop_img, (self.image_size[1], self.image_size[0]), cv.INTER_CUBIC)

        photo = np.array(crop_img, dtype=np.float32)

        # preprocess, normalization, [1, 3, 512, 512]
        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)), [1, self.image_size[2], self.image_size[0], self.image_size[1]])
        
        with torch.no_grad():
            images = Variable(torch.from_numpy(np.asarray(photo)).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            # [bt, num_classes, 128, 128]
            # [bt, 2, 128, 128]
            # [bt, 16, 128, 128]
            # [bt, 3, 128, 128]

            t1 = time.time()
            output_hm, output_center, output_vertex, output_size = self.model(images)

            outputs = decode_bbox(output_hm, output_center, output_vertex, output_size, self.confidence, self.cuda, 50)

            # nms
            empty_box = []
            try:
                if self.nms:
                    for i in range(len(outputs)):
                        x_min = np.min(outputs[i][:, 2:18:2], axis=1, keepdims=True)
                        x_max = np.max(outputs[i][:, 2:18:2], axis=1, keepdims=True)
                        y_min = np.min(outputs[i][:, 3:18:2], axis=1, keepdims=True)
                        y_max = np.max(outputs[i][:, 3:18:2], axis=1, keepdims=True)
                        cls_id = np.expand_dims(outputs[i][:, 22], 1)
                        bbox_det = np.concatenate([x_min, y_min, x_max, y_max, cls_id], 1)
                        empty_box.append(bbox_det)
                    np_det_results = np.array(empty_box, dtype=np.float32)
                    best_dets, best_det_indices = np.array(nms(np_det_results, self.nms_threhold))
                    if len(best_det_indices) == 1:
                        output = outputs[0]
                    else:
                        output = outputs[0][best_det_indices]
            except:
                output = outputs[0]
            
            if len(output) <= 0:
                return image, None, 1.0
            
            # normalization coordinate [0, 1]
            norm_center, norm_vertex, box_size, det_conf, det_cls = output[:,:2], output[:,2:18], output[:,18:21], output[:,21], output[:,22]
            
            # filter box conf > threshold
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_cls[top_indices].tolist()
            top_norm_center = norm_center[top_indices]
            top_norm_vertex = norm_vertex[top_indices]
            top_box_size = box_size[top_indices]
            
            # coordinate, raw img
            if self.letterbox_image:
                top_norm_center = correct_vertex_norm2raw(top_norm_center, image_shape)
                top_norm_vertex = correct_vertex_norm2raw(top_norm_vertex, image_shape)
            else:
                top_norm_center[:, 0] = top_norm_center[:, 0] * image_shape[1]
                top_norm_center[:, 1] = top_norm_center[:, 1] * image_shape[0]
                top_norm_vertex[:, 0:16:2] = top_norm_vertex[:, 0:16:2] * image_shape[1]
                top_norm_vertex[:, 1:16:2] = top_norm_vertex[:, 1:16:2] * image_shape[0]

            t2 = time.time()

            process_time = t2 - t1

        font = ImageFont.truetype(font="model_data/Times New Roman.ttf", size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32')//2)

        calib_matrix, calib_f, calib_fi = read_calib_params(calib_path, image_shape[1], image_shape[0])

        for i in range(len(top_label_indices)):
            c = int(top_label_indices[i])
            predicted_class = self.class_names[c]
            score = top_conf[i]
            cx, cy = top_norm_center[i].astype(np.int32)
            vertex = top_norm_vertex[i].astype(np.int32)
            l, w, h = top_box_size[i]

            # cls, conf
            label = '{}-{}: {:.2f}'.format(i + 1, predicted_class, score)
            draw = ImageDraw.Draw(image, "RGBA")
            label = label.encode("utf-8")

            # 3D box
            if draw_convex:
                # outline轮廓颜色比fill填充颜色深一些会有立体感
                # 0-1-5-4
                draw.polygon([(vertex[0], vertex[1]), (vertex[2], vertex[3]), (vertex[10], vertex[11]),
                              (vertex[8], vertex[9])], fill=(84, 255, 159, 125), outline=(0, 100, 0))
                # 0-3-7-4
                draw.polygon([(vertex[0], vertex[1]), (vertex[6], vertex[7]), (vertex[14], vertex[15]),
                              (vertex[8], vertex[9])],
                             fill=(84, 255, 159, 125), outline=(0, 100, 0))
                # 4-5-6-7
                draw.polygon([(vertex[8], vertex[9]), (vertex[10], vertex[11]), (vertex[12], vertex[13]),
                              (vertex[14], vertex[15])],
                             fill=(84, 255, 159, 125), outline=(0, 100, 0))
            else:
                draw.line([vertex[0], vertex[1], vertex[2], vertex[3]], fill=(255, 0, 0), width=2)
                draw.line([vertex[4], vertex[5], vertex[6], vertex[7]], fill=(255, 0, 0), width=2)
                draw.line([vertex[8], vertex[9], vertex[10], vertex[11]], fill=(255, 0, 0), width=2)
                draw.line([vertex[12], vertex[13], vertex[14], vertex[15]], fill=(255, 0, 0), width=2)

                # 长度方向
                # 0-3 1-2 4-7 5-6
                draw.line([vertex[0], vertex[1], vertex[6], vertex[7]], fill=(0, 0, 255), width=2)
                draw.line([vertex[2], vertex[3], vertex[4], vertex[5]], fill=(0, 0, 255), width=2)
                draw.line([vertex[8], vertex[9], vertex[14], vertex[15]], fill=(0, 0, 255), width=2)
                draw.line([vertex[10], vertex[11], vertex[12], vertex[13]], fill=(0, 0, 255), width=2)

                # 高度方向
                # 0-4 1-5 2-6 3-7
                draw.line([vertex[0], vertex[1], vertex[8], vertex[9]], fill=(0, 255, 0), width=2)
                draw.line([vertex[2], vertex[3], vertex[10], vertex[11]], fill=(0, 255, 0), width=2)
                draw.line([vertex[4], vertex[5], vertex[12], vertex[13]], fill=(0, 255, 0), width=2)
                draw.line([vertex[6], vertex[7], vertex[14], vertex[15]], fill=(0, 255, 0), width=2)

            draw.text((cx, cy), str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=self.colors[c], width=2)

            # draw vehicle size values
            # draw.text([(vertex[0] + vertex[6]) // 2-25, (vertex[1] + vertex[7]) // 2-25], "{:.2f}m".format(l),
            #           fill=(255, 0, 0), font=font)
            # draw.text([(vertex[0] + vertex[2]) // 2-25, (vertex[1] + vertex[3]) // 2], "{:.2f}m".format(w),
            #           fill=(255, 0, 0), font=font)
            # draw.text([(vertex[2] + vertex[10]) // 2, (vertex[3] + vertex[11]) // 2-20], "{:.2f}m".format(h),
            #           fill=(255, 0, 0), font=font)

            # save record
            if is_record_result:
                # calc 3d centroid
                cx_3d, cy_3d, cz_3d = RDUVtoXYZ(calib_matrix, cx, cy, 1000 * h / 2)
                if vertex[14] < vertex[2]:  # right perspective(7,1)
                    left, top, right, bottom = vertex[14], vertex[15], vertex[2], vertex[3]
                else:  # left perspective  (1x,6y,7x,0y)
                    left, top, right, bottom = vertex[2], vertex[13], vertex[14], vertex[1]

                vertex_3d = cal_pred_3dvertex(vertex, h, calib_matrix)

                all_veh_2dbbox.append([left, top, right - left, bottom - top])
                all_vehicle_type.append(predicted_class)
                vertex_save = np.reshape(vertex, [-1, 2])
                vertex_save = [tuple(v) for v in vertex_save]
                all_3dbbox_2dvertex.append(vertex_save)
                all_vehicle_size.append([l, w, h])
                all_vehicle_rots.append(0.0)
                all_vehicle_location_3d.append([cx_3d, cy_3d, cz_3d])
                all_perspective.append("right")
                all_base_point.append([right, bottom])
                vertex_3d_save = np.reshape(vertex_3d, [-1, 3])
                vertex_3d_save = [tuple(v) for v in vertex_3d_save]
                all_3dbbox_3dvertex.append(vertex_3d_save)
                all_vehicle_location.append([cx, cy])
                all_key_points = []

        del draw
        save3dbbox_result("o", xml_path, image_path, calib_path, np_img, all_veh_2dbbox,
                          all_vehicle_type, all_3dbbox_2dvertex,
                          all_vehicle_size, all_vehicle_rots, all_vehicle_location_3d,
                          all_perspective, all_base_point, all_3dbbox_3dvertex,
                          all_vehicle_location, all_key_points, False)
        return image, None, process_time
