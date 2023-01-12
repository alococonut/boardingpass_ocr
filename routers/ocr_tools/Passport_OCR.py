#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from paddleocr import PaddleOCR
from collections import defaultdict
from dateutil.parser import parse
from paddleocr import PaddleOCR, draw_ocr
import re
import numpy as np
import datetime
import pandas as pd


def sg_passport(image,ocr_model):

    result = ocr_model.ocr(image)[0]
    output = defaultdict(list)
    # Get keyword location
# =============================================================================
#     def is_date(string, fuzzy=True):
#         try: 
#             parse(string, fuzzy=fuzzy)
#             return True
#     
#         except:
#             return False
# =============================================================================
    def date_trans(date_str):
        split = [date_str[0:2],date_str[2:len(date_str)-4].strip(),date_str[-4:]]
        date_comb = split[0].replace('O','0')+' '+split[1].replace('0','O')+' '+split[2].replace('O','0')
        date_time = parse(date_comb).strftime("%d/%m/%Y")
        return date_time
    
    try:
        # Get info from MRZ
        mrz_1 = result[-2][1][0]
        mrz_2 = result[-1][1][0]
        output['name'] = mrz_1[5:].replace("<"," ").rstrip().replace("  "," ")
        output['passport No'] = mrz_2[:9]
        output['dob'] = parse(mrz_2[15:19]+mrz_2[13:15], fuzzy=False).strftime("%d/%m/%Y")
        output['expiry'] = parse(mrz_2[23:27]+mrz_2[21:23], fuzzy=False).strftime("%d/%m/%Y")
        output['sex'] = mrz_2[20]
        output['ID'] = mrz_2[28:37]
        
        dob_loc = [ i[0] for i in result if 'birth' in i[1][0] ][0]
        modi_loc = [ i[0] for i in result if 'Modi' in i[1][0] ][0]
        #date_lis = [ date_trans(i[1][0]) for i in result if re.search(r'\d\d\ (.*?)\d\d\d\d', i[1][0])]
        #date_lis = [ parse(i[1][0], fuzzy=True).strftime("%d/%m/%Y") for i in result if is_date(i[1][0]) ]
        date_lis = [date_trans(i[1][0]) for i in result if dob_loc[0][1]<i[0][0][1]<modi_loc[0][1] and sum(k.isdigit() for k in i[1][0])>=4]
        
        output['issue'] = date_lis[1]
        
    except:
        name_loc = [ i[0] for i in result if 'Name' in i[1][0] ][0]
        name_ind = [ i for i,j in enumerate(result) if 'Name' in j[1][0] ][0]
        pass_loc = [ i[0] for i in result if ('Pass' in i[1][0]) and ('No' in i[1][0]) ][0]
        

        # Get info from OCR
        if 'name' not in output.keys():
            output['name'] = result[name_ind+1][1][0]
        if 'dob' not in output.keys():
            output['dob'] = date_lis[0]
        if 'expiry' not in output.keys():
            output['expiry'] = date_lis[2]
        if 'ID' not in output.keys():    
            output['ID'] = result[-3][1][0]
        if 'passport No' not in output.keys():
            output['passport No']=[i for i in result if pass_loc[2][1]<(i[0][0][1]+i[0][2][1])/2 < name_loc[0][1]][-1][1][0]
        
        if 'sex' not in output.keys():
            for i in result:
                if dob_loc[0][1]-15 < i[0][0][1] < dob_loc[0][1] and i[0][0][0] < dob_loc[1][0]:
                    output['sex'] = i[1][0]
                    break
    
    return output


if __name__ == "__main__":
    image = cv2.imread('sg_sample2.png')
    ocr_model = PaddleOCR(lang="en")
    ocr_output = sg_passport(image,ocr_model)
    print(ocr_output)


#for i,j in enumerate (result): print (i,j)
    
    
    
    
    
    