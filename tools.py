
'''
包含工具函数

'''
import cv2 as cv
import numpy as np
import math
from xml.etree import ElementTree as ET
from xml.dom.minidom import Document

'''
func: 计算地平线
'''


def CalVPLine(rd_vpx, rd_vpy, prd_vpx, prd_vpy):
    k = (prd_vpy - rd_vpy) / (prd_vpx - rd_vpx)
    return (k, rd_vpx, rd_vpy)


'''
func: 读入标定参数, 消失点
'''


def ReadCalibParam(calib_xml_path):
    xml_dir = ET.parse(calib_xml_path)
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


'''
func: 将标定参数转换为变换矩阵(世界坐标y轴沿道路方向)
'''


def ParamToMatrix(focal, fi, theta, h, pcx, pcy):
    K = np.array([focal, 0, pcx, 0, focal, pcy, 0, 0, 1]).reshape(3, 3).astype(np.float)
    Rx = np.array([1, 0, 0, 0, -math.sin(fi), -math.cos(fi), 0, math.cos(fi), -math.sin(fi)]).reshape(3, 3).astype(np.float)
    Rz = np.array([math.cos(theta), -math.sin(theta), 0, math.sin(theta), math.cos(theta), 0, 0, 0, 1]).reshape(3,3).astype(np.float)
    R = np.dot(Rx, Rz)
    T = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -h]).reshape(3, 4).astype(np.float)
    trans = np.dot(R, T)
    H = np.dot(K, trans)
    return H


'''
func: 图像坐标--->世界坐标, 世界坐标z需要指定
'''


def RDUVtoXYZ(CalibTMatrix, u, v, z):
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
    b1 = u * (h33 * z + h34) - (h13 * z + h14)  # 与之前版本有修改
    b2 = v * (h33 * z + h34) - (h23 * z + h24)
    x = (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21)
    y = (a11 * b2 - b1 * a21) / (a11 * a22 - a12 * a21)
    return (x, y, z)


'''
func: 世界坐标--->图像坐标
'''


def RDXYZToUV(CalibTMatrix, x, y, z):
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

    u = (h11 * x + h12 * y + h13 * z + h14) / (h31 * x + h32 * y + h33 * z + h34)  # 与之前版本有修改
    v = (h21 * x + h22 * y + h23 * z + h24) / (h31 * x + h32 * y + h33 * z + h34)
    return (int(u), int(v))

def dashLine(img, p1, p2, color, thickness, interval):
    '''绘制虚线'''
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    if p1[0] == p2[0]:
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
    len = math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
    k = (float)(p2[1] - p1[1]) / (float)(p2[0] - p1[0] + 0.000000000001)
    seg = (int)(len / (float)(2 * interval))
    dev_x = 2 * interval / math.sqrt(1 + k * k)
    dev_y = k * dev_x   # 短直线向量
    pend1 = (p1[0] + dev_x / 2, p1[1] + dev_y / 2)
    # 绘制虚线点
    for i in range(seg):
        pbeg = (round(p1[0] + dev_x * i), round(p1[1] + dev_y * i))
        pend = (round(pend1[0] + dev_x * i), round(pend1[1] + dev_y * i))
        cv.line(img, pbeg, pend, color, thickness)
    # 补齐最后一段
    plastBeg = (round(p1[0] + dev_x * seg), round(p1[1] + dev_y * seg))
    if plastBeg[0] < p2[0]:
        cv.line(img, plastBeg, p2, color, thickness)


def cal_3dbbox(perspective, m_trans, veh_base_point, veh_turple_vp, l, w, h):
    # 重绘3D bbox
    veh_world_base_point = RDUVtoXYZ(m_trans, veh_base_point[0], veh_base_point[1], 0)
    veh_world_vp = RDUVtoXYZ(m_trans, veh_turple_vp[0], veh_turple_vp[1], 0)
    k0 = (veh_world_vp[1] - veh_world_base_point[1]) / (veh_world_vp[0] - veh_world_base_point[0] + 1e-8)
    k1 = -1.0 / k0
    dev_x0 = l / math.sqrt(1 + k0 * k0)  # 车长向量(+)
    dev_y0 = k0 * dev_x0  # 与k0同符号
    dev_x1 = w / math.sqrt(1 + k1 * k1)  # 车宽向量(+)
    dev_y1 = k1 * dev_x1  # 与k0相反符号

    if k0 > 0:  # 如果消失点连线与基准点斜率为正, 说明tan<90°, 长度方向向量需要与原始计算反向
        dev_x0 = - dev_x0
        dev_y0 = - dev_y0
    else:
        pass

    p1_3d = veh_world_base_point
    p5_3d = (p1_3d[0], p1_3d[1], h)
    if perspective == 'left':
        p0_3d = (p1_3d[0] + dev_x1, p1_3d[1] + dev_y1, 0)  # 宽度方向
        centroid_3d = (p1_3d[0] + dev_x1 / 2 - dev_x0 / 2, p1_3d[1] + dev_y1 / 2 - dev_y0 / 2, h / 2)  # 质心
    else:  # right
        p0_3d = (p1_3d[0] - dev_x1, p1_3d[1] - dev_y1, 0)  # 宽度方向
        centroid_3d = (p1_3d[0] - dev_x1 / 2 - dev_x0 / 2, p1_3d[1] - dev_y1 / 2 - dev_y0 / 2, h / 2)  # 质心

    p2_3d = (p1_3d[0] - dev_x0, p1_3d[1] - dev_y0, 0)  # 长度方向
    p3_3d = (p0_3d[0] - dev_x0, p0_3d[1] - dev_y0, 0)
    p4_3d = (p0_3d[0], p0_3d[1], h)
    p6_3d = (p2_3d[0], p2_3d[1], h)
    p7_3d = (p3_3d[0], p3_3d[1], h)

    centroid_2d = RDXYZToUV(m_trans, centroid_3d[0], centroid_3d[1], centroid_3d[2])
    list_3dbbox_3dvertex = [p0_3d, p1_3d, p2_3d, p3_3d, p4_3d, p5_3d, p6_3d, p7_3d]
    list_3dbbox_2dvertex = []
    for i in range(len(list_3dbbox_3dvertex)):
        if i == 1:
            list_3dbbox_2dvertex.append((veh_base_point[0], veh_base_point[1]))
        else:
            list_3dbbox_2dvertex.append(RDXYZToUV(m_trans, list_3dbbox_3dvertex[i][0], list_3dbbox_3dvertex[i][1], list_3dbbox_3dvertex[i][2]))
    return list_3dbbox_2dvertex, list_3dbbox_3dvertex, centroid_2d


def save3dbbox_result(xml_path, filepath, calib_file_path, frame, bbox_2d, bbox_type, bbox_2dvertex, veh_size, perspective, veh_base_point, bbox_3dvertex, vehicle_location):
    # 创建dom文档
    doc = Document()

    # 创建根节点
    annotation = doc.createElement('annotation')
    # 根节点插入dom树
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

    # 每一组信息先创建节点<...>，然后插入到父节点<object>下
    for i in range(len(bbox_2dvertex)):
        object = doc.createElement('object')

        type = doc.createElement('type')
        type_text = doc.createTextNode(str(bbox_type[i]))
        type.appendChild(type_text)
        object.appendChild(type)

        # 创建<2dbbox>标签元素
        bbox2d = doc.createElement('bbox2d')
        temp_bbox2d_str = " ".join(str(i) for i in bbox_2d[i])  # 将list拆分为str
        # 创建<3dbbox>下的文本节点
        bbox2d_text = doc.createTextNode(temp_bbox2d_str)
        # 将文本节点插入到<3dbbox>下
        bbox2d.appendChild(bbox2d_text)
        # 将<bbox>插入到父节点<object>下
        object.appendChild(bbox2d)

        # 创建<3dbbox>标签元素
        bbox = doc.createElement('vertex2d')
        temp_bbox_str = " ".join(str(i) for i in bbox_2dvertex[i])  # 将list拆分为str
        # 创建<3dbbox>下的文本节点
        bbox_text = doc.createTextNode(temp_bbox_str)
        # 将文本节点插入到<3dbbox>下
        bbox.appendChild(bbox_text)
        # 将<bbox>插入到父节点<object>下
        object.appendChild(bbox)

        vehiclesize = doc.createElement('veh_size')
        temp_veh_size_str = " ".join(str(i) for i in veh_size[i])  # 将list拆分为str
        vehiclesize_text = doc.createTextNode(temp_veh_size_str)
        vehiclesize.appendChild(vehiclesize_text)
        object.appendChild(vehiclesize)

        perspect = doc.createElement('perspective')
        temp_perspect_str = perspective[i]
        perspect_text = doc.createTextNode(temp_perspect_str)
        perspect.appendChild(perspect_text)
        object.appendChild(perspect)

        base_point = doc.createElement('base_point')
        base_point_str = " ".join(str(i) for i in veh_base_point[i])  # 将list拆分为str
        base_point_text = doc.createTextNode(base_point_str)
        base_point.appendChild(base_point_text)
        object.appendChild(base_point)

        bbox3d = doc.createElement('vertex3d')
        temp_bbox3d_str = " ".join(str(i) for i in bbox_3dvertex[i])  # 将list拆分为str
        bbox3d_text = doc.createTextNode(temp_bbox3d_str)
        bbox3d.appendChild(bbox3d_text)
        object.appendChild(bbox3d)

        loc = doc.createElement('veh_loc_2d')
        temp_loc_str = " ".join(str(i) for i in vehicle_location[i])  # 将list拆分为str
        loc_text = doc.createTextNode(temp_loc_str)
        loc.appendChild(loc_text)
        object.appendChild(loc)

        annotation.appendChild(object)


    # 将dom对象写入本地xml文件
    with open(xml_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))



