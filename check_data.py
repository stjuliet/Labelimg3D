import os
import xml.etree.ElementTree as ET

# cam0:342 322  cam1:311 105

save_dir = "real_scene_cam1/"
start_index = 0
end_index = 3701

for i in range(start_index, end_index):
    xml_file_name = "bg_%06d" % i
    xml_file_path = os.path.join(save_dir, xml_file_name + ".xml")

    if os.path.exists(xml_file_path):
        xml_tree = ET.parse(xml_file_path)
        xml_root = xml_tree.getroot()

        per_frame_valid_check_count = 0

        for obj in xml_root.findall("object"):
            if str(obj.find("veh_size").text) == "4.669999999999996 1.78 1.6":
                per_frame_valid_check_count += 1
        if per_frame_valid_check_count != 1:
            print("unvalid sample: ", i)
