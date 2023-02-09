#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cv2


##(1) read into  bgr-space
#img = cv2.imread("/Users/feiranyang/Documents/Python_projects/Doc_OCR/Singapore_IC/EP/EP_Ron.jpeg")

def hsv(img_file):
    ##(2) convert to hsv-space, then split the channels
    img = cv2.imread(img_file)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    
    ##(3) threshold the S channel using adaptive method(`THRESH_OTSU`) or fixed thresh
    th, threshed_s = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY_INV)
    th, threshed_v = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY_INV)
    
    def morph(img):
        blurred = cv2.blur(img, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations = 4)
        closed = cv2.dilate(closed, None, iterations = 4)
    
        return closed
    
    closed_s = morph(threshed_s)
    closed_v = morph(threshed_v)
    closed_v_int = cv2.bitwise_not(closed_v)
    
    ##(4) find all the external contours on the threshed S
    cnts_s = cv2.findContours(closed_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts_v = cv2.findContours(closed_v_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    #cv2.drawContours(canvas, cnts, -1, (0,255,0), 1)
    
    ## sort and choose the largest contour
    cnt_v = sorted(cnts_v, key = cv2.contourArea)[-1]
    cnt_s = sorted(cnts_s, key = cv2.contourArea)[-1]
    
    if cv2.contourArea(cnt_v)/(cv2.arcLength(cnt_v,True)+1) > \
        cv2.contourArea(cnt_s)/(cv2.arcLength(cnt_s,True)+1):
            cnt = cnt_v
    else: cnt = cnt_s
    
    ## approx the contour, so the get the corner points
    canvas  = img.copy()
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02* arclen, True)
    
    # =============================================================================
    # cv2.drawContours(canvas, [cnt], -1, (255,0,0), 1, cv2.LINE_AA)
    # cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 1, cv2.LINE_AA)
    # 
    # ## Ok, you can see the result as tag(6)
    # cv2.imwrite("canvas.png", canvas)
    # =============================================================================
    
    # =============================================================================
    # cv2.imwrite("closed_s.png", closed_s)
    # cv2.imwrite("closed_v.png", closed_v_int)
    # =============================================================================
    return approx



