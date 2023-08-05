# functions

import math
import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from xml.etree import ElementTree as ET


def letterbox_image(image, size):
    """ resize without deformation"""
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))  # gray
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def pool_nms(heat, kernel=3):
    """ nms with max pool """
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def nms(results, nms_threshold):
    """ common nms """
    outputs = []
    keep = []
    for i in range(len(results)):  # each img
        detections = results[i]
        unique_class = np.unique(detections[:, -1])

        best_box = []
        if len(unique_class) == 0:
            results.append(best_box)
            continue

        for c in unique_class:  # cls
            cls_mask = detections[:, -1] == c

            detection = detections[cls_mask]
            scores = detection[:, 4]
            # descending order
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]
            while np.shape(detection)[0] > 0:
                i = arg_sort[0]
                best_box.append(detection[0])
                keep.append(i)
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1], detection[1:])
                detection = detection[1:][ious < nms_threshold]  # only keep box with ious < nms_threshold
                inds = np.where(ious < nms_threshold)[0]
                arg_sort = arg_sort[inds + 1]
        outputs.append(best_box)
    return outputs, keep


def iou(b1, b2):
    """
    2d iou
    b1: [lt_x, ly_y, br_x, br_y]
    b2: [num_img, 4], 4: [lt_x, ly_y, br_x, br_y]
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou


def nms_diou(bbox_p, bbox_g):
    """
    bbox_p: [lt_x, ly_y, br_x, br_y]
    bbox_g: [lt_x, ly_y, br_x, br_y]
    """
    # pred 2d box area
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # gt 2d box area
    area_g = abs(bbox_g[:, 2] - bbox_g[:, 0]) * abs(bbox_g[:, 3] - bbox_g[:, 1])

    center_p = np.array([(bbox_p[0] + bbox_p[2]) / 2., (bbox_p[1] + bbox_p[3]) / 2.], dtype=np.float32)
    center_g = np.array([(bbox_g[:, 0] + bbox_g[:, 2]) / 2., (bbox_g[:, 1] + bbox_g[:, 3]) / 2.], dtype=np.float32).T

    # box center distance
    box_center_dis = (center_p[0] - center_g[:, 0]) ** 2 + (center_p[1] - center_g[:, 1]) ** 2

    x_min_inter = np.maximum(bbox_p[0], bbox_g[:, 0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[:, 1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[:, 2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[:, 3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)

    union = area_p + area_g - intersection

    # diagonal of union enclosing rectangle
    x_min_union = np.minimum(bbox_p[0], bbox_g[:, 0])
    y_min_union = np.minimum(bbox_p[1], bbox_g[:, 1])
    x_max_union = np.maximum(bbox_p[2], bbox_g[:, 2])
    y_max_union = np.maximum(bbox_p[3], bbox_g[:, 3])

    external_dis = (x_max_union - x_min_union) ** 2 + (y_max_union - y_min_union) ** 2

    # iou
    iou = intersection / np.maximum(union, 1e-6)

    # diou
    diou = iou - (box_center_dis / np.maximum(external_dis, 1e-6))

    return diou


def basic_iou(bbox_p, bbox_g):
    """
    bbox_p: [lt_x, ly_y, br_x, br_y]
    bbox_g: [lt_x, ly_y, br_x, br_y]
    """
    # pred 2d box area
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # gt 2d box area
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)

    union = area_p + area_g - intersection

    # iou
    iou = intersection / np.maximum(union, 1e-6)

    return iou


def basic_giou(bbox_p, bbox_g):
    """
    bbox_p: [lt_x, ly_y, br_x, br_y]
    bbox_g: [lt_x, ly_y, br_x, br_y]
    """
    # pred 2d box area
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # gt 2d box area
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)

    union = area_p + area_g - intersection

    # union enclosing rectangle
    x_min_union = np.minimum(bbox_p[0], bbox_g[0])
    y_min_union = np.minimum(bbox_p[1], bbox_g[1])
    x_max_union = np.maximum(bbox_p[2], bbox_g[2])
    y_max_union = np.maximum(bbox_p[3], bbox_g[3])
    external_rectangle = np.maximum(abs(x_max_union - x_min_union), 0) * np.maximum(abs(y_max_union - y_min_union), 0)

    # iou
    iou = intersection / np.maximum(union, 1e-6)

    # giou
    giou = iou - ((external_rectangle - union) / np.maximun(external_rectangle, 1e-6))

    return giou


def basic_diou(bbox_p, bbox_g):
    """
    bbox_p: [lt_x, ly_y, br_x, br_y]
    bbox_g: [lt_x, ly_y, br_x, br_y]
    """
    # pred 2d box area
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # gt 2d box area
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    center_p = np.array([(bbox_p[0]+bbox_p[2])/2., (bbox_p[1]+bbox_p[3])/2.], dtype=np.float32)
    center_g = np.array([(bbox_g[0]+bbox_g[2])/2., (bbox_g[1]+bbox_g[3])/2.], dtype=np.float32)

    # box center distance
    box_center_dis = (center_p[0] - center_g[0])**2 + (center_p[1] - center_g[1])**2

    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)

    union = area_p + area_g - intersection

    # diagonal of union enclosing rectangle
    x_min_union = np.minimum(bbox_p[0], bbox_g[0])
    y_min_union = np.minimum(bbox_p[1], bbox_g[1])
    x_max_union = np.maximum(bbox_p[2], bbox_g[2])
    y_max_union = np.maximum(bbox_p[3], bbox_g[3])

    external_dis = (x_max_union - x_min_union)**2 + (y_max_union - y_min_union)**2

    # iou
    iou = intersection / np.maximum(union, 1e-6)

    # diou
    diou = iou - box_center_dis / np.maximum(external_dis, 1e-6)

    return diou


def basic_ciou(bbox_p, bbox_g):
    """
    bbox_p: [lt_x, ly_y, br_x, br_y]
    bbox_g: [lt_x, ly_y, br_x, br_y]
    """
    # pred 2d box area
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # gt 2d box area
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    bbox_h_p = bbox_p[3] - bbox_p[1]
    bbox_w_p = bbox_p[2] - bbox_p[0]
    bbox_h_g = bbox_g[3] - bbox_g[1]
    bbox_w_g = bbox_g[2] - bbox_g[0]

    center_p = np.array([(bbox_p[0]+bbox_p[2])/2., (bbox_p[1]+bbox_p[3])/2.], dtype=np.float32)
    center_g = np.array([(bbox_g[0]+bbox_g[2])/2., (bbox_g[1]+bbox_g[3])/2.], dtype=np.float32)

    # box center distance
    box_center_dis = (center_p[0] - center_g[0])**2 + (center_p[1] - center_g[1])**2

    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)

    union = area_p + area_g - intersection

    # diagonal of union enclosing rectangle
    x_min_union = np.minimum(bbox_p[0], bbox_g[0])
    y_min_union = np.minimum(bbox_p[1], bbox_g[1])
    x_max_union = np.maximum(bbox_p[2], bbox_g[2])
    y_max_union = np.maximum(bbox_p[3], bbox_g[3])

    external_dis = (x_max_union - x_min_union)**2 + (y_max_union - y_min_union)**2

    # iou
    iou = intersection / np.maximum(union, 1e-6)

    # penalty of box ratio
    v = (4 / math.pi**2) * (math.atan(bbox_w_g/bbox_h_g) - math.atan(bbox_w_p/bbox_h_p))**2
    alpha = v / np.maximum((1 - iou + v), 1e-6)

    # ciou
    ciou = iou - (box_center_dis / np.maximum(external_dis, 1e-6)) - v*alpha

    return ciou


def basic_cdiou(vertex_p, vertex_g, bbox_p, bbox_g):
    """
    vertex_p: [1, 16]
    vertex_g: [1, 16]
    bbox_p: [lt_x, ly_y, br_x, br_y]
    bbox_g: [lt_x, ly_y, br_x, br_y]
    """
    v_dis = 0.0
    # 计算顶点距离之和
    for i in range(len(vertex_p)//2):  # [0, 8)
        dis = math.sqrt((vertex_p[2*i] - vertex_g[2*i])**2 + (vertex_p[2*i+1] - vertex_g[2*i+1])**2)
        v_dis += dis

    # pred 2d box area
    area_p = abs(bbox_p[2] - bbox_p[0]) * abs(bbox_p[3] - bbox_p[1])
    # gt 2d box area
    area_g = abs(bbox_g[2] - bbox_g[0]) * abs(bbox_g[3] - bbox_g[1])

    x_min_inter = np.maximum(bbox_p[0], bbox_g[0])
    y_min_inter = np.maximum(bbox_p[1], bbox_g[1])
    x_max_inter = np.minimum(bbox_p[2], bbox_g[2])
    y_max_inter = np.minimum(bbox_p[3], bbox_g[3])
    intersection = np.maximum(abs(x_max_inter - x_min_inter), 0) * np.maximum(abs(y_max_inter - y_min_inter), 0)

    union = area_p + area_g - intersection

    # # diagonal of union enclosing rectangle
    x_min_union = np.minimum(bbox_p[0], bbox_g[0])
    y_min_union = np.minimum(bbox_p[1], bbox_g[1])
    x_max_union = np.maximum(bbox_p[2], bbox_g[2])
    y_max_union = np.maximum(bbox_p[3], bbox_g[3])

    external_dis = math.sqrt((x_max_union - x_min_union)**2 + (y_max_union - y_min_union)**2)

    # iou
    iou = intersection / np.maximum(union, 1e-6)

    # ciou
    cdiou = iou - (v_dis / np.maximum(8*external_dis, 1e-6))

    return cdiou


def basic_3diou(b1, b2):
    """
    3d iou with different view
    left: x3_1 > x1_2
    right: x3_1 < x1_2
    b1, b2: [x0, y0, z0, x1, y1, z1, ... , x7, y7, z7]
    0: 0, 1, 2,
    1: 3, 4, 5,
    2: 6, 7, 8,
    3: 9, 10, 11,
    4: 12, 13, 14,
    5: 15, 16, 17,
    6: 18, 19, 20,
    7: 21, 22, 23,
    """
    if b1[9] < b2[3]:  # right
        min_x = np.maximum(b1[0], b2[0])
        max_x = np.minimum(b1[3], b2[3])
    else:  # left
        min_x = np.maximum(b1[3], b2[3])
        max_x = np.minimum(b1[0], b2[0])

    min_y = np.maximum(b1[1], b2[1])
    max_y = np.minimum(b1[10], b2[10])

    min_z = np.maximum(b1[2], b2[2])
    max_z = np.minimum(b1[14], b2[14])

    x_overlap = max_x - min_x
    y_overlap = max_y - min_y
    z_overlap = max_z - min_z

    if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
        overlap_volumn = x_overlap * y_overlap * z_overlap
        b1_volumn = abs(b1[3]-b1[0])*(b1[10]-b1[1])*(b1[14]-b1[2])
        b2_volumn = abs(b2[3]-b2[0])*(b2[10]-b2[1])*(b2[14]-b2[2])

        union_volumn = b1_volumn + b2_volumn - overlap_volumn
        iou = overlap_volumn / np.maximum(union_volumn, 1e-6)
    else:
        iou = 0

    return iou


def calib_param_to_matrix(focal, fi, theta, h, pcx, pcy):
    """
    world axis - y
    :param focal:
    :param fi: phi
    :param theta:
    :param h: camera height
    :param pcx: principle point u
    :param pcy: principle point v
    :return: world -> image matrix
    """
    K = np.array([focal, 0, pcx, 0, focal, pcy, 0, 0, 1]).reshape(3, 3).astype(np.float)
    Rx = np.array([1, 0, 0, 0, -math.sin(fi), -math.cos(fi), 0, math.cos(fi), -math.sin(fi)]).reshape(3, 3).astype(np.float)
    Rz = np.array([math.cos(theta), -math.sin(theta), 0, math.sin(theta), math.cos(theta), 0, 0, 0, 1]).reshape(3,3).astype(np.float)
    R = np.dot(Rx, Rz)
    T = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -h]).reshape(3, 4).astype(np.float)
    trans = np.dot(R, T)
    H = np.dot(K, trans)
    return H


def read_calib_params(xml_path, width, height):
    """
    :param xml_path: calib xml file path
    :param width: raw img w
    :param height: raw img h
    :return: world -> image matrix
    """
    xml_dir = ET.parse(xml_path)
    root = xml_dir.getroot()
    node_f = xml_dir.find('f')
    node_fi = xml_dir.find('fi')
    node_theta = xml_dir.find('theta')
    node_h = xml_dir.find('h')
    calib_matrix = calib_param_to_matrix(float(node_f.text), float(node_fi.text), float(node_theta.text),
                                float(node_h.text), width / 2, height / 2)
    return calib_matrix, float(node_f.text), float(node_fi.text)


def RDUVtoXYZ(CalibTMatrix, u, v, z):
    """
    func: img ---> world, z (need to specify)
    unit: mm
    """
    h11 = CalibTMatrix[0][0]
    h12 = CalibTMatrix[0][1]
    h13 = CalibTMatrix[0][2]
    h14 = CalibTMatrix[0][3]
    h21 = CalibTMatrix[1][0]
    h22 = CalibTMatrix[1][1]
    h23 = CalibTMatrix[1][2]
    h24 = CalibTMatrix[1][3]
    h31 = CalibTMatrix[2][0]
    h32 = CalibTMatrix[2][1]
    h33 = CalibTMatrix[2][2]
    h34 = CalibTMatrix[2][3]

    a11 = h11 - u * h31
    a12 = h12 - u * h32
    a21 = h21 - v * h31
    a22 = h22 - v * h32
    b1 = u * (h33 * z + h34) - (h13 * z + h14)  # revised
    b2 = v * (h33 * z + h34) - (h23 * z + h24)
    x = (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21)
    y = (a11 * b2 - b1 * a21) / (a11 * a22 - a12 * a21)
    return (x, y, z)


def RDXYZToUV(CalibTMatrix, x, y, z):
    """'
    func: world ---> img
    unit：mm
    """
    h11 = CalibTMatrix[0][0]
    h12 = CalibTMatrix[0][1]
    h13 = CalibTMatrix[0][2]
    h14 = CalibTMatrix[0][3]
    h21 = CalibTMatrix[1][0]
    h22 = CalibTMatrix[1][1]
    h23 = CalibTMatrix[1][2]
    h24 = CalibTMatrix[1][3]
    h31 = CalibTMatrix[2][0]
    h32 = CalibTMatrix[2][1]
    h33 = CalibTMatrix[2][2]
    h34 = CalibTMatrix[2][3]

    u = (h11 * x + h12 * y + h13 * z + h14) / (h31 * x + h32 * y + h33 * z + h34)  # revised
    v = (h21 * x + h22 * y + h23 * z + h24) / (h31 * x + h32 * y + h33 * z + h34)
    return (int(u), int(v))


def dashLine(img, p1, p2, color, thickness, interval):
    """
    draw dashline
    """
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    if p1[0] == p2[0]:
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
    len = math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
    k = (float)(p2[1] - p1[1]) / (float)(p2[0] - p1[0] + 1e-6)
    seg = (int)(len / (float)(2 * interval))
    dev_x = 2 * interval / math.sqrt(1 + k * k)
    dev_y = k * dev_x
    pend1 = (p1[0] + dev_x / 2, p1[1] + dev_y / 2)
    for i in range(seg):
        pbeg = (round(p1[0] + dev_x * i), round(p1[1] + dev_y * i))
        pend = (round(pend1[0] + dev_x * i), round(pend1[1] + dev_y * i))
        cv.line(img, pbeg, pend, color, thickness)
    # last segment
    plastBeg = (round(p1[0] + dev_x * seg), round(p1[1] + dev_y * seg))
    if plastBeg[0] < p2[0]:
        cv.line(img, plastBeg, p2, color, thickness)


def cal_pred_2dvertex(perspective, base_point, l, w, h, m_trans):
    """
    use calib matrix, base point, vehicle size (m), view,
    to calculate vertex of 3d box in img
    """
    w_p1 = RDUVtoXYZ(m_trans, base_point[0], base_point[1], 0)
    if perspective == 1:  # right view
        p0 = RDXYZToUV(m_trans, w_p1[0] - w * 1000, w_p1[1], w_p1[2])
        p2 = RDXYZToUV(m_trans, w_p1[0], w_p1[1] + l * 1000, w_p1[2])
        p3 = RDXYZToUV(m_trans, w_p1[0] - w * 1000, w_p1[1] + l * 1000, w_p1[2])
        p4 = RDXYZToUV(m_trans, w_p1[0] - w * 1000, w_p1[1], w_p1[2] + h * 1000)
        p5 = RDXYZToUV(m_trans, w_p1[0], w_p1[1], w_p1[2] + h * 1000)
        p6 = RDXYZToUV(m_trans, w_p1[0], w_p1[1] + l * 1000, w_p1[2] + h * 1000)
        p7 = RDXYZToUV(m_trans, w_p1[0] - w * 1000, w_p1[1] + l * 1000, w_p1[2] + h * 1000)
    else:
        p0 = RDXYZToUV(m_trans, w_p1[0] + w * 1000, w_p1[1], w_p1[2])
        p2 = RDXYZToUV(m_trans, w_p1[0], w_p1[1] + l * 1000, w_p1[2])
        p3 = RDXYZToUV(m_trans, w_p1[0] + w * 1000, w_p1[1] + l * 1000, w_p1[2])
        p4 = RDXYZToUV(m_trans, w_p1[0] + w * 1000, w_p1[1], w_p1[2] + h * 1000)
        p5 = RDXYZToUV(m_trans, w_p1[0], w_p1[1], w_p1[2] + h * 1000)
        p6 = RDXYZToUV(m_trans, w_p1[0], w_p1[1] + l * 1000, w_p1[2] + h * 1000)
        p7 = RDXYZToUV(m_trans, w_p1[0] + w * 1000, w_p1[1] + l * 1000, w_p1[2] + h * 1000)
    return np.array([p0[0], p0[1], base_point[0], base_point[1], p2[0], p2[1], p3[0], p3[1],
            p4[0], p4[1], p5[0], p5[1], p6[0], p6[1], p7[0], p7[1]], dtype=np.float32)


def cal_pred_3dvertex(vertex_2d, h, m_trans):
    """
    use vertex of 3d box in img, vehicle height (m), calib matrix,
    to calculate vertex of 3d box in world
    """
    w_p0 = RDUVtoXYZ(m_trans, vertex_2d[0], vertex_2d[1], 0)
    w_p1 = RDUVtoXYZ(m_trans, vertex_2d[2], vertex_2d[3], 0)
    w_p2 = RDUVtoXYZ(m_trans, vertex_2d[4], vertex_2d[5], 0)
    w_p3 = RDUVtoXYZ(m_trans, vertex_2d[6], vertex_2d[7], 0)
    w_p4 = RDUVtoXYZ(m_trans, vertex_2d[8], vertex_2d[9], h * 1000)
    w_p5 = RDUVtoXYZ(m_trans, vertex_2d[10], vertex_2d[11], h * 1000)
    w_p6 = RDUVtoXYZ(m_trans, vertex_2d[12], vertex_2d[13], h * 1000)
    w_p7 = RDUVtoXYZ(m_trans, vertex_2d[14], vertex_2d[15], h * 1000)
    return np.array([w_p0[0], w_p0[1], w_p0[2], w_p1[0], w_p1[1], w_p1[2],w_p2[0], w_p2[1], w_p2[2],
    w_p3[0], w_p3[1], w_p3[2],w_p4[0], w_p4[1], w_p4[2],w_p5[0], w_p5[1], w_p5[2],
    w_p6[0], w_p6[1], w_p6[2],w_p7[0], w_p7[1], w_p7[2]], dtype=np.float32)


def decode_bbox(pred_hms, pred_center, pred_vertex, pred_size, threshold, cuda, topk=100):
    """
    decode
    """
    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    for batch in range(b):
        #   pred_hms        128*128, num_classes    heatmap
        #   pred_center     128*128, 2              centroid xy offset
        #   pred_vertex     128*128, 16             vertex of 3d box in img
        #   pred_size       128*128, 3              vehicle size
        heat_map = pred_hms[batch].permute(1,2,0).view([-1,c])
        pred_center = pred_center[batch].permute(1,2,0).view([-1,2])
        pred_vertex = pred_vertex[batch].permute(1,2,0).view([-1,16])
        pred_size = pred_size[batch].permute(1,2,0).view([-1,3])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))

        #   xv              128*128,    center x
        #   yv              128*128,    center y
        xv, yv = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        #   class_conf      128*128,    cls conf
        #   class_pred      128*128,    cls
        #   mask   index
        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > threshold

        # filtered result
        pred_center_mask = pred_center[mask]
        pred_vertex_mask = pred_vertex[mask]
        pred_size_mask = pred_size[mask]
        if len(pred_center_mask) == 0:
            detects.append([])
            continue     

        xv_mask = torch.unsqueeze(xv[mask] + pred_center_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_center_mask[..., 1], -1)

        # normalization --- [0, 1]
        norm_center = torch.cat([xv_mask, yv_mask], dim=1)
        norm_center[:, 0] /= output_w
        norm_center[:, 1] /= output_h

        # normalization --- [0, 1]
        pred_vertex_mask[:, 0:16:2] = pred_vertex_mask[:, 0:16:2] / output_w
        pred_vertex_mask[:, 1:16:2] = pred_vertex_mask[:, 1:16:2] / output_h

        detect = torch.cat([norm_center, pred_vertex_mask, pred_size_mask, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)

        arg_sort = torch.argsort(detect[:, -2], descending=True)  # cls_conf
        detect = detect[arg_sort]

        detects.append(detect.cpu().numpy()[:topk])
    return detects


def correct_vertex_norm2raw(norm_vertex, raw_image_shape):
    raw_img_h, raw_img_w = raw_image_shape
    if raw_img_h < raw_img_w:  # w > h
        norm_vertex[:, 0:norm_vertex.shape[1]:2] = norm_vertex[:, 0:norm_vertex.shape[1]:2] * max(raw_img_h, raw_img_w)
        norm_vertex[:, 1:norm_vertex.shape[1]:2] = norm_vertex[:, 1:norm_vertex.shape[1]:2] * max(raw_img_h, raw_img_w) - abs(raw_img_h-raw_img_w)//2.
    else:  # w < h
        norm_vertex[:, 0:norm_vertex.shape[1]:2] = norm_vertex[:, 0:norm_vertex.shape[1]:2] * max(raw_img_h, raw_img_w) - abs(raw_img_h-raw_img_w)//2.
        norm_vertex[:, 1:norm_vertex.shape[1]:2] = norm_vertex[:, 1:norm_vertex.shape[1]:2] * max(raw_img_h, raw_img_w)
    return norm_vertex


def coord_to_homog(points):
    # [x,y] -> [x, y, 1]
    # points: [num_point, [pt_x, pt_y]]
    return np.hstack((points, np.ones((points.shape[0], 1))))


def coord_to_normal(points):
    # [x, y, z] -> [x/z, y/z]
    # points: [num_point, [pt_x, pt_y]]
    points[:, 0] = points[:, 0] / points[:, 2]
    points[:, 1] = points[:, 1] / points[:, 2]
    return np.delete(points, 2, axis=1)


def calc_cross_point(line_one, line_two):
    """
    求解两条直线斜率，及交点
    line_one: [x1,y1,x2,y2]
    line_two: [x1,y1,x2,y2]
    """
    k1 = (line_one[3] - line_one[1]) / (line_one[2] - line_one[0] + 1e-6)
    b1 = line_one[1] - k1*line_one[0]

    k2 = (line_two[3] - line_two[1]) / (line_two[2] - line_two[0] + 1e-6)
    b2 = line_two[1] - k2*line_two[0]

    cross_x = (b1-b2)/(k2-k1)
    cross_y = k1*cross_x + b1
    return np.array([cross_x, cross_y]).astype(np.float32)


def calc_vps_on_hl(horizon_line, lines):
    """
    求解多条直线与地平线的交点，直线采用点斜式表示
    horizon_line: [pt1, pt2]
    lines: [[pt1, pt2], [pt1,pt2], ..., [pt1, pt2]]
    """
    all_cross = []
    for line in lines:
        cross = calc_cross_point(horizon_line, line)
        all_cross.append(cross)
    all_cross = np.array(all_cross).astype(np.float32)
    final_cross = all_cross.mean(axis=0)
    return final_cross


# -----------------------------gaussian-----------------------------------------------------#
def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)
# -----------------------------gaussian-----------------------------------------------------#
