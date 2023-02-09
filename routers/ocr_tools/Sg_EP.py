#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os,shutil,math,warnings,calendar,datetime,re,random
import cv2
from collections import defaultdict
from dateutil.parser import parse
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import pandas as pd
from datetime import datetime
from routers.ocr_tools.Barcode_detect import crop_barcode
from routers.ocr_tools.ep_tool.Crop_image import crop_image
import zxingcpp

#['EMPLOYMENT PASS', 'Employer', 'Employment of Foreign Manpower Act (Chapter 91A)', 'FIN', 'Name', 'Republic of Singapore']

def label_detect(datafolder):
    ocr_model = PaddleOCR(lang="en")
    #crop image using houghline method and concatenate extracted text
    os.chdir(datafolder)
    
    current_directory = os.getcwd()
    crop_directory = current_directory+'/Cropped_image/'
    if not os.path.exists(crop_directory):
        os.makedirs(crop_directory)
    
    text_all = []
    sample_number = 0
    for file in os.listdir(datafolder):
        if os.path.splitext(file)[1] in ['.png', '.jpg', '.jpeg']:
            cropped_file = crop_image(datafolder+'/'+file,crop_directory)
            text_all=text_all+ocr_model.ocr(cropped_file)[0]
            sample_number += 1
            
    #identify labels in the document
    text_list = [i[1][0] for i in text_all]
    text_strip = [i.replace(' ','') for i in text_list]
    label_list = [i for i in text_list if text_strip.count(i.replace(' ','')) == sample_number]
    label_list.sort()
    
    if sample_number>0 and len(label_list)%sample_number==0:
        label_len = int(len(label_list) / sample_number)
    
    label_list2=[''] * label_len
    for i in range(label_len):
        for j in range(sample_number):
            if len(label_list[i*sample_number+j])>len(label_list2[i]):
                label_list2[i] = label_list[i*sample_number+j]
    
    value_list = [i for i in text_list if text_strip.count(i.replace(' ',''))< sample_number]
    
    value_left = [i[:i.rfind(' ')] for i in value_list if ' ' in i]
    value_right = [i[i.rfind(' '):] for i in value_list if ' ' in i]
    left_strip = [i.replace(' ','') for i in value_left]
    right_strip = [i.replace(' ','') for i in value_right]
    
    
    add_label = []
    for i in range(len(value_left)):
        if left_strip.count(left_strip[i]) > sample_number* 0.6 \
            and right_strip.count(right_strip[i]) < sample_number* 0.8 \
            and add_label.count(value_left[i])==0:
                add_label.append(value_left[i])
    
    label_list2 = label_list2 + add_label
    print(label_list2)
    return label_list2

def sg_EP(IC_file,label_list):
    ocr_model = PaddleOCR(lang="en")
    #match label with value in a document
    IC_doc = ocr_model.ocr(IC_file)[0]
    output = defaultdict(list)
    label_segm = []
    value_segm = []
    
    for i in IC_doc:
        for n,j in enumerate(label_list):
            if j.replace(' ','') == i[1][0].replace(' ','')[:len(j.replace(' ',''))] \
                and ' ' in i[1][0] and len(j)<len(i[1][0]):
                output[j] = i[1][0][len(j):].strip()
            elif j.replace(' ','') == i[1][0].replace(' ',''):
                label_segm.append(i)
                break
            elif n== len(label_list)-1:
                value_segm.append(i)
                
    label_value = defaultdict(list)
    value_label = defaultdict(list)
    for i in label_segm:
        i_ctr = [i[0][0][0],(i[0][0][1]+i[0][2][1])/2]
        sml_dist = 0
        for j in value_segm:
            j_ctr = [0.8*j[0][0][0]+0.2*j[0][2][0],(j[0][0][1]+j[0][2][1])/2]
            try:
                dist = (math.log(j_ctr[0]-i_ctr[0]+10))**2 \
                    + (math.log(j_ctr[1]-i_ctr[1]+10))**2 \
                    + math.log(abs(j[0][0][0]-i_ctr[0])+5)*math.log(abs(j_ctr[1]-i_ctr[1])+5)
                #print(i[1][0],j[1][0],math.log(j_ctr[0]-i_ctr[0]+10),\
                #math.log(j_ctr[1]-i_ctr[1]+10),math.log(abs(j[0][0][0]-i_ctr[0])+5),dist)
                if sml_dist==0 or dist<sml_dist:
                    sml_dist=dist
                    label_value[i[1][0]]=j[1][0]
            except:
                continue
                
    for i in value_segm:
        i_ctr = [0.8*i[0][0][0]+0.2*i[0][2][0],(i[0][0][1]+i[0][2][1])/2]
        sml_dist = 0
        for j in label_segm:
            j_ctr = [j[0][0][0],(j[0][0][1]+j[0][2][1])/2]
            try:
                dist = (math.log(i_ctr[0]-j_ctr[0]+10))**2 \
                    + (math.log(i_ctr[1]-j_ctr[1]+10))**2 \
                    + math.log(abs(i[0][0][0]-j_ctr[0])+5)*math.log(abs(i_ctr[1]-j_ctr[1])+5)
                #print(i[1][0],j[1][0],(math.log(i_ctr[0]-j_ctr[0]+10)),\
                #(math.log(i_ctr[1]-j_ctr[1]+10)),dist)
                if sml_dist==0 or dist<sml_dist:
                    sml_dist=dist
                    value_label[i[1][0]]=j[1][0]
            except:
                continue
    
    for k1,v1 in label_value.items():
        for k2, v2 in value_label.items():
            if k1==v2 and v1==k2:
                output[k1] = v1
    
    #scan barcode
    #image = cv2.imread(IC_file)
    #image = cv2.imdecode(np.fromstring(IC_file, np.uint8), cv2.IMREAD_UNCHANGED)
    image = IC_file
    string = zxingcpp.read_barcodes(image)[0].text
    if len(string) == 0:
        cropped_image = crop_barcode(image)
        #cv2.imwrite('barcode.png', cropped_image)
        string = zxingcpp.read_barcodes(cropped_image)[0].text
    
    output['Issue date'] = datetime.strptime(string[-8:],'%d%m%Y').strftime("%d/%m/%Y")
    output['FIN'] = string[:-8]
    
    print(output)
    return output

if __name__ == "__main__":
    datafolder='/Users/feiranyang/Documents/Python_projects/Doc_OCR/Singapore_IC/EP'
    label_list = label_detect(datafolder)
    for file in os.listdir(datafolder+'/Cropped_image/'):
        if os.path.splitext(file)[1] in ['.png', '.jpg', '.jpeg']:
            sg_EP(datafolder+'/Cropped_image/'+file,label_list)


# =============================================================================
# value_test = ['s9385672d',
#  'dai yufeng',
#  'm',
#  '27-08-1993',
#  'china',
#  'republic of singapore',
#  'chinese',
#  '3',
#  '#']
# import spacy
# nlp = spacy.load('en_core_web_lg')
# word1 = 'Date of birth' 
# word2 = 'Singapore'
# token1 = nlp(word1)
# token2 = nlp(word2)
# 
# token1.similarity(token2)
# 
# for i in label_list:
#     for j in value_test:
#         token1 = nlp(i)
#         token2 = nlp(j)
#         print(i, j, " Similarity:", token1.similarity(token2))
# =============================================================================

