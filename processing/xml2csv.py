'''
需要修改两个位置
fileDir: 为xml文件所在的目录
outputName：为csv文件的输出名称
'''

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

fileDir = 'images-train'  # xml文件地址
outputName = 'train.csv' # 生成csv文件地址
os.chdir(fileDir)
path = fileDir

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            print('value: ', value)
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = path
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(outputName, index=None)
    print('Successfully converted xml to csv.')


main()