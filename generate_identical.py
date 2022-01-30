# generate identical object xmls
import os
import xml.etree.ElementTree as ET

save_dir = "real_scene_cam0/"
start_index = 134
end_index = 400

for i in range(start_index, end_index):
    base_xml_file_name = "bg_%06d" % (i-1)
    xml_file_name = "bg_%06d" % i
    base_xml_file_path = os.path.join(save_dir, base_xml_file_name + ".xml")
    xml_file_path = os.path.join(save_dir, xml_file_name + ".xml")

    base_xml_tree = ET.parse(base_xml_file_path)
    base_root = base_xml_tree.getroot()

    # revise
    base_filename = base_xml_tree.find("filename").text
    new_base_filename_dir = base_filename.replace(base_xml_file_name, xml_file_name)
    base_xml_tree.find("filename").text = new_base_filename_dir

    with open(xml_file_path, "w") as xml:
        base_xml_tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)

print("successfully generate!")
