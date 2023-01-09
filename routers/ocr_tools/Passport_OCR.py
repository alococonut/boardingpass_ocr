#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from paddleocr import PaddleOCR
from collections import defaultdict
from dateutil.parser import parse
from paddleocr import PaddleOCR, draw_ocr
import re
import numpy as np


def date_trans(date_str):
    split = date_str.split()
    date_comb = split[0]+' '+split[1].replace('0','O')+' '+split[2]
    date_time = parse(date_comb).strftime("%d/%m/%Y")
    return date_time

def sg_passport(image,ocr_model):

    result = ocr_model.ocr(image)[0]
    output = defaultdict(list)
    # Get keyword location
    try:
        name_loc = [ i[0] for i in result if 'Name' in i[1][0] ][0]
        name_ind = [ i for i,j in enumerate(result) if 'Name' in j[1][0] ][0]
        pass_loc = [ i[0] for i in result if ('Pass' in i[1][0]) and ('No' in i[1][0]) ][0]
        date_lis = [ date_trans(i[1][0]) for i in result if re.search(r'\d\d\ (.*?)\d\d\d\d', i[1][0])]
        dob_loc = [ i[0] for i in result if 'birth' in i[1][0] ][0]
        
        # Get data info
        output['name'] = result[name_ind+1][1][0]
        output['dob'] = date_lis[0]
        output['issue'] = date_lis[1]
        output['expiry'] = date_lis[2]
        output['ID'] = result[-3][1][0]
        
        for i in result:
            if pass_loc[2][1]<(i[0][0][1]+i[0][2][1])/2 < name_loc[0][1] and i[0][0][0] > pass_loc[0][0]:
                output['passport No'] = i[1][0]
            elif dob_loc[0][1]-30 < i[0][0][1] < dob_loc[0][1] and i[0][0][0] < dob_loc[1][0]:
                output['sex'] = i[1][0]
                break

    except:
        # Get info from MRZ
        mrz_1 = result[-2][1][0]
        mrz_2 = result[-1][1][0]
        if 'name' not in output.keys():
            output['name'] = mrz_1[5:].replace("<"," ").rstrip().replace("  "," ")
        if 'passport No' not in output.keys():
            output['passport No'] = mrz_2[:9]
        if 'dob' not in output.keys():
            output['dob'] = parse(mrz_2[13:19], fuzzy=False).strftime("%d/%m/%Y")
        if 'expiry' not in output.keys():
            output['expiry'] = parse(mrz_2[21:27], fuzzy=False).strftime("%d/%m/%Y")
        if 'sex' not in output.keys():
            output['sex'] = mrz_2[20]
        if 'ID' not in output.keys():
            output['ID'] = mrz_2[28:37]
    
    return output


if __name__ == "__main__":
    image = cv2.imread('/Users/feiranyang/Documents/Python_projects/Passport/SG_FR2.jpeg')
    ocr_model = PaddleOCR(lang="en")
    result = sg_passport(image,ocr_model)
    print(result)
    