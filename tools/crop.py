import cv2
import os
import shutil
import numpy as np

from xml.dom.minidom import parse
import xml.dom.minidom

def parse_xml(xml_name):
    res = []
    DOMTree = xml.dom.minidom.parse(xml_name)
    anno = DOMTree.documentElement
    boxes = anno.getElementsByTagName("bndbox")
    for box in boxes:
        xmin = box.getElementsByTagName('xmin')[0]
        xmin = int(xmin.childNodes[0].data)
        ymin = box.getElementsByTagName('ymin')[0]
        ymin = int(ymin.childNodes[0].data)
        xmax = box.getElementsByTagName('xmax')[0]
        xmax = int(xmax.childNodes[0].data)
        ymax = box.getElementsByTagName('ymax')[0]
        ymax = int(ymax.childNodes[0].data)
        res.append([xmin, ymin, xmax, ymax])
    return res
 
base = '/home/liuliang/Desktop/alpha_pictures/kisspng'

imgDir = '{}/ok_pictures'.format(base)
alaDir = '{}/ok_alpha'.format(base)
rgbDir = '{}/ok_rgb'.format(base)
xmlDir = '{}/crop_xml'.format(base)

imgSavDir = '{}/crop_pictures'.format(base)
alaSavDir = '{}/crop_alpha'.format(base)
rgbSavDir = '{}/crop_rgb'.format(base)

lists = '{}/ok.txt'.format(base)
f = open(lists)
img_ids = f.readlines()

append_img_id = 1505

for img_id in img_ids:
    img_id = img_id[:5]

    img_name = '{}/{}.png'.format(imgDir, img_id) 
    ala_name = '{}/{}.png'.format(alaDir, img_id) 
    rgb_name = '{}/{}.png'.format(rgbDir, img_id) 
    xml_name = '{}/{}.xml'.format(xmlDir, img_id) 

    img_sav_name = '{}/{}.png'.format(imgSavDir, img_id)
    ala_sav_name = '{}/{}.png'.format(alaSavDir, img_id)
    rgb_sav_name = '{}/{}.png'.format(rgbSavDir, img_id)

    if os.path.exists(xml_name):
        crops = parse_xml(xml_name)
        print(crops)  
        cnt = len(crops)      
        if cnt > 0:    
            img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            ala = cv2.imread(ala_name, cv2.IMREAD_UNCHANGED)
            rgb = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
            for i in range(cnt):
                if i > 0:
                    img_sav_name = '{}/{:0>5}.png'.format(imgSavDir, append_img_id)
                    ala_sav_name = '{}/{:0>5}.png'.format(alaSavDir, append_img_id)
                    rgb_sav_name = '{}/{:0>5}.png'.format(rgbSavDir, append_img_id)
                    append_img_id += 1
                x1, y1, x2, y2 = crops[i]
                crop_img = img[y1 : y2 + 1, x1 : x2 + 1]
                crop_ala = ala[y1 : y2 + 1, x1 : x2 + 1]
                crop_rgb = rgb[y1 : y2 + 1, x1 : x2 + 1]
                cv2.imwrite(img_sav_name, crop_img)
                cv2.imwrite(ala_sav_name, crop_ala)
                cv2.imwrite(rgb_sav_name, crop_rgb)
        else:
            shutil.copyfile(img_name, img_sav_name)
            shutil.copyfile(ala_name, ala_sav_name)
            shutil.copyfile(rgb_name, rgb_sav_name)
    else:
        shutil.copyfile(img_name, img_sav_name)
        shutil.copyfile(ala_name, ala_sav_name)
        shutil.copyfile(rgb_name, rgb_sav_name)
