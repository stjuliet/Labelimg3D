# include tools

import cv2 as cv
import numpy as np
import math
from xml.etree import ElementTree as ET
from xml.dom.minidom import Document


def CalVPLine(rd_vpx, rd_vpy, prd_vpx, prd_vpy):
    """
    func: calculate horizon line
    """
    k = (prd_vpy - rd_vpy) / (prd_vpx - rd_vpx)
    return (k, rd_vpx, rd_vpy)


def ReadCalibParam(calib_xml_path):
    """
    func: read calib parameters
    """
    xml_dir = ET.parse(calib_xml_path)
    if xml_dir.find('f'):
        focal = float(xml_dir.find('f').text)
        fi = float(xml_dir.find('fi').text)
        theta = float(xml_dir.find('theta').text)
        cam_height = float(xml_dir.find('h').text)
        list_vps = xml_dir.find('vanishPoints').text.split()
        np_list_vps = np.array(list_vps).astype(np.float32)
        rd_vpx = int(np_list_vps[0])
        rd_vpy = int(np_list_vps[1])
        prd_vpx = int(np_list_vps[2])
        prd_vpy = int(np_list_vps[3])
        vpline = CalVPLine(rd_vpx, rd_vpy, prd_vpx, prd_vpy)
        return focal, fi, theta, cam_height, (rd_vpx, rd_vpy), vpline
    else:
        xml_f = cv.FileStorage(calib_xml_path, cv.FileStorage_READ)
        calib_matrix = np.array(xml_f.getNode("calib_matrix").mat())
        xml_f.release()
        return calib_matrix


def ParamToMatrix(focal, fi, theta, h, pcx, pcy):
    """
    func: tansfer calib parameters to matrix
    :param focal:
    :param fi:
    :param theta:
    :param h:
    :param pcx:
    :param pcy:
    :return:
    """
    K = np.array([focal, 0, pcx, 0, focal, pcy, 0, 0, 1]).reshape(3, 3).astype(np.float)
    Rx = np.array([1, 0, 0, 0, -math.sin(fi), -math.cos(fi), 0, math.cos(fi), -math.sin(fi)]).reshape(3, 3).astype(np.float)
    Rz = np.array([math.cos(theta), -math.sin(theta), 0, math.sin(theta), math.cos(theta), 0, 0, 0, 1]).reshape(3,3).astype(np.float)
    R = np.dot(Rx, Rz)
    T = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -h]).reshape(3, 4).astype(np.float)
    trans = np.dot(R, T)
    H = np.dot(K, trans)
    return H


def RDUVtoXYZ(CalibTMatrix, u, v, z):
    """
    img -> world, y axis, mm
    :param CalibTMatrix:
    :param u:
    :param v:
    :param z:
    :return:
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
    b1 = u * (h33 * z + h34) - (h13 * z + h14)  # fix
    b2 = v * (h33 * z + h34) - (h23 * z + h24)
    x = (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21)
    y = (a11 * b2 - b1 * a21) / (a11 * a22 - a12 * a21)
    return (x, y, z)


def RDXYZToUV(CalibTMatrix, x, y, z):
    """
    world -> img, y axis, mm
    :param CalibTMatrix:
    :param x:
    :param y:
    :param z:
    :return:
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

    u = (h11 * x + h12 * y + h13 * z + h14) / (h31 * x + h32 * y + h33 * z + h34)  # fix
    v = (h21 * x + h22 * y + h23 * z + h24) / (h31 * x + h32 * y + h33 * z + h34)
    return (int(u), int(v))


def dashLine(img, p1, p2, color, thickness, interval):
    """
    draw dashline
    :param img:
    :param p1:
    :param p2:
    :param color:
    :param thickness:
    :param interval:
    :return:
    """
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    if p1[0] == p2[0]:
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
    len = math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
    k = (float)(p2[1] - p1[1]) / (float)(p2[0] - p1[0] + 1e-8)
    seg = (int)(len / (float)(2 * interval))
    dev_x = 2 * interval / math.sqrt(1 + k * k)
    dev_y = k * dev_x
    pend1 = (p1[0] + dev_x / 2, p1[1] + dev_y / 2)
    for i in range(seg):
        pbeg = (round(p1[0] + dev_x * i), round(p1[1] + dev_y * i))
        pend = (round(pend1[0] + dev_x * i), round(pend1[1] + dev_y * i))
        cv.line(img, pbeg, pend, color, thickness)
    plastBeg = (round(p1[0] + dev_x * seg), round(p1[1] + dev_y * seg))
    if plastBeg[0] < p2[0]:
        cv.line(img, plastBeg, p2, color, thickness)


def cal_3dbbox(perspective, m_trans, veh_base_point, veh_turple_vp, l, w, h, rot):
    """
    draw 3d box
    :param perspective:
    :param m_trans:
    :param veh_base_point:
    :param veh_turple_vp:
    :param l:
    :param w:
    :param h:
    :param rot:
    :return:
    """
    veh_world_base_point = RDUVtoXYZ(m_trans, veh_base_point[0], veh_base_point[1], 0)
    veh_world_vp = RDUVtoXYZ(m_trans, veh_turple_vp[0], veh_turple_vp[1], 0)
    k0 = (veh_world_vp[1] - veh_world_base_point[1]) / (veh_world_vp[0] - veh_world_base_point[0] + 1e-8)
    k1 = -1.0 / k0
    dev_x0 = l / math.sqrt(1 + k0 * k0)  # length vector (+)
    dev_y0 = k0 * dev_x0  # the same sign to k0
    dev_x1 = w / math.sqrt(1 + k1 * k1)  # width vector (+)
    dev_y1 = k1 * dev_x1  # opposite sign to K0

    if k0 > 0:  # tan < 90°
        dev_x0 = - dev_x0
        dev_y0 = - dev_y0
    else:
        pass

    p1_3d = veh_world_base_point
    p5_3d = (p1_3d[0], p1_3d[1], h)
    if perspective == 'left':
        p0_3d = (p1_3d[0] + dev_x1, p1_3d[1] + dev_y1, 0)  # width
        centroid_3d = (p1_3d[0] + dev_x1 / 2 - dev_x0 / 2, p1_3d[1] + dev_y1 / 2 - dev_y0 / 2, h / 2)  # centroid
    else:  # right
        p0_3d = (p1_3d[0] - dev_x1, p1_3d[1] - dev_y1, 0)  # width
        centroid_3d = (p1_3d[0] - dev_x1 / 2 - dev_x0 / 2, p1_3d[1] - dev_y1 / 2 - dev_y0 / 2, h / 2)  # centroid

    p2_3d = (p1_3d[0] - dev_x0, p1_3d[1] - dev_y0, 0)  # length
    p3_3d = (p0_3d[0] - dev_x0, p0_3d[1] - dev_y0, 0)
    p4_3d = (p0_3d[0], p0_3d[1], h)
    p6_3d = (p2_3d[0], p2_3d[1], h)
    p7_3d = (p3_3d[0], p3_3d[1], h)

    centroid_2d = RDXYZToUV(m_trans, centroid_3d[0], centroid_3d[1], centroid_3d[2])
    # list_3dbbox_3dvertex = [p0_3d, p1_3d, p2_3d, p3_3d, p4_3d, p5_3d, p6_3d, p7_3d]
    np_3dbbox_3dveretx = np.array([p0_3d, p1_3d, p2_3d, p3_3d, p4_3d, p5_3d, p6_3d, p7_3d]).reshape(-1, 3)
    np_3dbbox_centroid = np.array(centroid_3d).reshape(3,)
    np_3dbbox_3dveretx_rot = rotate_point(np_3dbbox_3dveretx, np_3dbbox_centroid, rot)
    list_3dbbox_3dvertex = np_3dbbox_3dveretx_rot.tolist()

    list_3dbbox_2dvertex = []
    for i in range(len(list_3dbbox_3dvertex)):
        # if i == 1:
        #     list_3dbbox_2dvertex.append((veh_base_point[0], veh_base_point[1]))
        # else:
        list_3dbbox_2dvertex.append(RDXYZToUV(m_trans, list_3dbbox_3dvertex[i][0], list_3dbbox_3dvertex[i][1], list_3dbbox_3dvertex[i][2]))
    return list_3dbbox_2dvertex, list_3dbbox_3dvertex, centroid_2d


def rotate_point(point_3d, centroid_3d, degree):
    """
    :param: point_3d, [N, 3]
    :param: centroid_3d, [1, 3]
    :param: degree, need to be transferred to rad
    """
    rot_rad = np.deg2rad(degree - 180)
    rot_matrix = np.array([np.cos(rot_rad), np.sin(rot_rad), -np.sin(rot_rad), np.cos(rot_rad)]).reshape(2, 2)
    rot_center = np.expand_dims(centroid_3d, 0).T
    xy_result = np.matmul(rot_matrix, point_3d[:, :2].T - rot_center[:2]) + rot_center[:2]
    xy_result = xy_result.T
    z_result = np.expand_dims(point_3d[:, 2], 1)
    return np.hstack((xy_result, z_result))


def save3dbbox_result(xml_path, filepath, calib_file_path, frame, bbox_2d, bbox_type, bbox_2dvertex, veh_size, veh_rot, perspective, veh_base_point, bbox_3dvertex, vehicle_location, key_points, keypoint_flag):
    # create dom
    doc = Document()

    # root
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(filepath)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    calibfile = doc.createElement('calibfile')
    calibfile_text = doc.createTextNode(calib_file_path)
    calibfile.appendChild(calibfile_text)
    annotation.appendChild(calibfile)

    size = doc.createElement('size')

    width = doc.createElement('width')
    width_text = doc.createTextNode(str(frame.shape[1]))
    width.appendChild(width_text)
    size.appendChild(width)

    height = doc.createElement('height')
    height_text = doc.createTextNode(str(frame.shape[0]))
    height.appendChild(height_text)
    size.appendChild(height)

    depth = doc.createElement('depth')
    depth_text = doc.createTextNode(str(frame.shape[2]))
    depth.appendChild(depth_text)
    size.appendChild(depth)

    annotation.appendChild(size)

    # create <node>, then insert to parent node <object>:
    for i in range(len(bbox_2dvertex)):
        object = doc.createElement('object')

        type = doc.createElement('type')
        type_text = doc.createTextNode(str(bbox_type[i]))
        type.appendChild(type_text)
        object.appendChild(type)

        # <2dbbox>
        bbox2d = doc.createElement('bbox2d')
        temp_bbox2d_str = " ".join(str(i) for i in bbox_2d[i])  # list -> str
        bbox2d_text = doc.createTextNode(temp_bbox2d_str)
        bbox2d.appendChild(bbox2d_text)
        # <2dbbox> -> <object>
        object.appendChild(bbox2d)

        # <3dbbox>
        bbox = doc.createElement('vertex2d')
        temp_bbox_str = " ".join(str(i) for i in bbox_2dvertex[i])
        bbox_text = doc.createTextNode(temp_bbox_str)
        bbox.appendChild(bbox_text)
        object.appendChild(bbox)

        vehiclesize = doc.createElement('veh_size')
        temp_veh_size_str = " ".join(str(i) for i in veh_size[i])
        vehiclesize_text = doc.createTextNode(temp_veh_size_str)
        vehiclesize.appendChild(vehiclesize_text)
        object.appendChild(vehiclesize)

        vehiclerot = doc.createElement('veh_rot')
        temp_veh_rot_str = str(veh_rot[i])
        vehiclerot_text = doc.createTextNode(temp_veh_rot_str)
        vehiclerot.appendChild(vehiclerot_text)
        object.appendChild(vehiclerot)

        perspect = doc.createElement('perspective')
        temp_perspect_str = perspective[i]
        perspect_text = doc.createTextNode(temp_perspect_str)
        perspect.appendChild(perspect_text)
        object.appendChild(perspect)

        base_point = doc.createElement('base_point')
        base_point_str = " ".join(str(i) for i in veh_base_point[i])
        base_point_text = doc.createTextNode(base_point_str)
        base_point.appendChild(base_point_text)
        object.appendChild(base_point)

        bbox3d = doc.createElement('vertex3d')
        temp_bbox3d_str = " ".join(str(i) for i in bbox_3dvertex[i])
        bbox3d_text = doc.createTextNode(temp_bbox3d_str)
        bbox3d.appendChild(bbox3d_text)
        object.appendChild(bbox3d)

        loc = doc.createElement('veh_loc_2d')
        temp_loc_str = " ".join(str(i) for i in vehicle_location[i])
        loc_text = doc.createTextNode(temp_loc_str)
        loc.appendChild(loc_text)
        object.appendChild(loc)

        if i <= len(key_points) - 1 and keypoint_flag:
            key_point = doc.createElement('key_points')
            temp_key_point_str = " ".join(str(i) for i in key_points[i])
            key_point_text = doc.createTextNode(temp_key_point_str)
            key_point.appendChild(key_point_text)
            object.appendChild(key_point)

        annotation.appendChild(object)

    # write dom object
    with open(xml_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
