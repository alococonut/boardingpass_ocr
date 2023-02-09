#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import imutils
import cv2
import heapq
import sys
import math
import os
from routers.ocr_tools.ep_tool.color import hsv


def crop_image(image_url,crop_directory):

    # Load image
    #image_url = '/Users/feiranyang/Documents/Python_projects/Doc_OCR/Singapore_IC/EP/EP_Anil.jpeg'
    #image_url = '/Users/feiranyang/Documents/Python_projects/Doc_OCR/Singapore_IC/EP/EP_Ron.jpeg'
    image = cv2.imread(image_url)
    src = cv2.imread(image_url, cv2.IMREAD_GRAYSCALE)
    dst = cv2.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 30, None, 100, 80)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    #cv2.imwrite('image_Houghline_PH1.png', cdstP) 
      
    
    blurred = cv2.blur(cdstP, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    #cv2.imwrite('blurred.png', blurred)
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite('kernel.png', kernel)
    #cv2.imwrite('closed.png', closed)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    #cv2.imwrite('erode.png', closed)
    closed = cv2.dilate(closed, None, iterations = 4)
    #cv2.imwrite('Houghline_closed1.png', closed)
    
    #change red color to white
    
    closed[np.where((closed==[0,0,255]).all(axis=2))] = [255,255,255]
    closed = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY) 
    #contours, hierarchy = cv2.findContours(image=closed, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(image=closed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    #src_copy = src.copy()
    #cv2.drawContours(image=src_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #cv2.imwrite('imgGry_houline_contour2.png', src_copy) 
    
    cnt_all = np.concatenate(contours)
    x,y,w,h = bbox = cv2.boundingRect(cnt_all)
    
    #compare Houghline with hsv
    cnt_hsv = hsv(image_url)
    x2,y2,w2,h2 = bbox = cv2.boundingRect(cnt_hsv)
    
    if w*h<w2*h2:
        cropped_image = image[max(int(y-0.05*h),0):int(y+1.05*h), max(int(x-0.05*w),0):int(x+1.05*w)]
    else:
        cropped_image = image[y2:y2+h2, x2:x2+w2]
    #cropped_image = image[int(y):int(y+1*h), int(x):int(x+1*w)]
    cropped_file_name = 'crop_'+os.path.basename(image_url)
    save_file = crop_directory+cropped_file_name
    cv2.imwrite(save_file, cropped_image) 
    
    return save_file
    
# =============================================================================
#     src_copy2 = image.copy()
#     nh, nw = src_copy.shape[:2]
#     if h >= 0.3 * nh:
#         cv2.rectangle(src_copy2, (x,y), (x+w, y+h), (0, 0, 255), 5, cv2.LINE_AA)
#     cv2.imwrite('rect_contour_SG5.png', src_copy2) 
# =============================================================================
    
# =============================================================================
# image_copy_appr = src.copy()
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
#     #cv2.drawContours(image_copy_appr, [approx], 0, (0, 0, 0), 5)
#     cv2.drawContours(image=image_copy_appr, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# 
# cv2.imwrite('Houghline_closed1_Appr.png', image_copy_appr)   
# =============================================================================


