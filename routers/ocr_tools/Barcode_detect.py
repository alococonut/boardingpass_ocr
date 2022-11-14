#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import imutils
import cv2
import zxingcpp
import datetime

def crop_barcode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    # =============================================================================
    # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    # cv2.imwrite('box.png', image)
    # =============================================================================
    
    
    # define cropping area
    x1 = min(box[0][0],box[1][0],box[2][0],box[3][0])
    x2 = max(box[0][0],box[1][0],box[2][0],box[3][0])
    y1 = min(box[0][1],box[1][1],box[2][1],box[3][1])
    y2 = max(box[0][1],box[1][1],box[2][1],box[3][1])
    
    
    if (x2-x1)*1.5 < (y2-y1):
        width = x2 - x1
        length = y2 - y1
        x1_1 = int(max(x1 - width*0.2,0))
        x2_1 = int(x2 + width*0.2)
        y1_1 = int(max(y1 - length*0.8,0))
        y2_1 = int(y2 + length*0.8)
    
    else:
        length = x2 - x1
        width = y2 - y1
        x1_1 = int(max(x1 - length*0.8,0))
        x2_1 = int(x2 + length*0.8)
        y1_1 = int(max(y1 - width*0.2,0))
        y2_1 = int(y2 + width*0.2)
      
    cropped_image = image[y1_1:y2_1,x1_1:x2_1]

    return cropped_image

def extract_info(results):

    for result in results:
        content_dic = {}
        text_lis = result.text.split(' ')
        text_lis = [i for i in text_lis if i ]
        index = text_lis.index([i for i in text_lis if i.isnumeric()][0])

        content_dic['Name'] = ' '.join(text_lis[:index-2]).replace('/',' ')[2:]
        content_dic['Flight_number'] = text_lis[index-1][-2:] + text_lis[index]
        content_dic['Julian_date'] = text_lis[index+1][:3]
        now_time = datetime.datetime.now().strftime('%y%j')
        print(now_time)
        
        if int(now_time[-3:])>=int(content_dic['Julian_date']):
            content_dic['Julian_date'] = str(int(now_time[:-3]))+content_dic['Julian_date']
        else:
            content_dic['Julian_date'] = str(int(now_time[:-3])-1)+content_dic['Julian_date']    

        content_dic['Date'] = datetime.datetime.strptime(content_dic['Julian_date'],'%y%j').date().strftime('%d-%b-%Y')

    return content_dic

# decode using zxing
if __name__ == "__main__":
    image = cv2.imread('./image/sin.jpg')
    results = zxingcpp.read_barcodes(image)
    if len(results) == 0:
        cropped_image = crop_barcode(image)
        results = zxingcpp.read_barcodes(cropped_image)
    
    output = extract_info(results)
    print(output)

    
    
    