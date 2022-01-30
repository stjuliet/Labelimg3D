import os
import xml.etree.ElementTree as ET

# 901--903
# add one vehicle annotation

save_dir = "real_scene_cam0/"
start_index = 901
end_index = 904
base_xml_file_name = "bg_%06d" % (start_index - 1)  # base

base_xml_file_path = os.path.join(save_dir, base_xml_file_name + ".xml")
base_xml_tree = ET.parse(base_xml_file_path)
base_root = base_xml_tree.getroot()


def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def SubElementWithText(parent, tag, text):
    element = parent.makeelement(tag, {})
    parent.append(element)
    element.text = text
    return element


for i in range(start_index, end_index):
    xml_file_name = "bg_%06d" % i
    xml_file_path = os.path.join(save_dir, xml_file_name + ".xml")

    if os.path.exists(xml_file_path):
        revise_base_xml_tree = ET.parse(xml_file_path)
        revise_base_root = revise_base_xml_tree.getroot()

        for idx, obj in enumerate(base_root.iter('object')):
            type = obj.find("type").text
            bbox2d = obj.find("bbox2d").text
            vertex2d = obj.find("vertex2d").text
            veh_size = obj.find("veh_size").text
            perspective = obj.find("perspective").text
            base_point = obj.find("base_point").text
            vertex3d = obj.find("vertex3d").text
            veh_loc_2d = obj.find("veh_loc_2d").text

            object_ele = ET.SubElement(revise_base_root, "object")

            sub_object = SubElementWithText(object_ele, "type", type)
            sub_object = SubElementWithText(object_ele, "bbox2d", bbox2d)
            sub_object = SubElementWithText(object_ele, "vertex2d", vertex2d)
            sub_object = SubElementWithText(object_ele, "veh_size", veh_size)
            sub_object = SubElementWithText(object_ele, "perspective", perspective)
            sub_object = SubElementWithText(object_ele, "base_point", base_point)
            sub_object = SubElementWithText(object_ele, "vertex3d", vertex3d)
            sub_object = SubElementWithText(object_ele, "veh_loc_2d", veh_loc_2d)

        pretty_xml(revise_base_root, "\t", "\n")
        with open(xml_file_path, "w") as xml:
            revise_base_xml_tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)

print("successfully add!")
