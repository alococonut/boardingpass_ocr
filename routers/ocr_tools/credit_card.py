#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os,shutil,math,warnings,calendar,datetime,re,random
import cv2
from collections import defaultdict
from dateutil.parser import parse
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import pandas as pd
from pytesseract import Output
import pytesseract as pt
import argparse
import imutils
#from Crop_image import crop_image
pt.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.3.0/bin/tesseract'

#file='/Users/feiranyang/Documents/Python_projects/Doc_OCR/Credit_card/CITI.jpeg'
def credit_kard(card_file):
#rotate image
    image = card_file
    rotation_angle = pt.image_to_osd(card_file,output_type = Output.DICT)['rotate']
    if rotation_angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #cv2.imwrite(file, image)
    
    
    # =============================================================================
    # #crop image
    # current_directory = os.getcwd()
    # crop_directory = current_directory+'/Cropped_image/'
    # cropped_file = crop_image(file,crop_directory)
    # =============================================================================
    
    ocr_model = PaddleOCR(lang="en",det_db_score_mode='slow')
    credit_card = ocr_model.ocr(image)[0]
    
    output = defaultdict(list)
    text_list = [i[1][0] for i in credit_card]
    
    for i in text_list:
        if len(i)>=10 and re.sub(r'[^\w]', ' ', i).replace(' ','').isdigit():
            output['card number'] = re.sub(r'[^\w]', ' ', i).replace(' ','')
        elif re.search(r'\d\d\/\d\d', i):
            output['valid thru'] = re.search(r'\d\d\/\d\d', i).group()
        elif len(i) > 5 and not bool(re.search(r'\d', i)) and i.isupper():
            output['name'] =i
    
    print(output)
    return output

if __name__ == "__main__":
    file='/Users/feiranyang/Documents/Python_projects/Doc_OCR/Credit_card/CITI.jpeg'
    credit_card(file)