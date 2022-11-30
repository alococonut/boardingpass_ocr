import fitz
import cv2
import numpy as np
from commons.logger import logger as logging
from paddleocr import PaddleOCR
# from pytesseract import Output
# import pytesseract
from collections import defaultdict



class receipt():

    def __init__(self,input,model,name):
        self.content = input
        self.model = model
        self.filename = name
    
    def chose_file(self):
        if self.filename.split('.')[-1] == 'pdf':
            doc = fitz.open(stream=self.content,filetype='pdf')
            page = doc.load_page(0)
            text = page.get_text('text')
            if len(text)>20:
                output = self.process_rawpdf(doc)
                return output
            else:
                output = self.process_image(doc)
                return output
        else:
            image = cv2.imdecode(np.fromstring(self.content, np.uint8), cv2.IMREAD_UNCHANGED)
            return self.process_image(image)

    def process_rawpdf(self,doc):
        page_0 = doc.load_page(0)
        if 'Grab' in page_0.get_text('text'):
            output = self.extract_grab(doc)
            return output
        elif 'Gojek Receipts' in page_0.get_text('text'):
            output = self.extract_gojek(doc)
            return output


    def process_image(self,image):
        result = self.model.ocr(image)
        for i in result[0]:
            print(i)
        return result[0]


    def extract_grab(self,doc):
        page_0 = doc.load_page(0)
        page_1 = doc.load_page(1)
        text_0 = page_0.get_text('text')
        text_1 = page_1.get_text('text')
        text_lis1 = text_0.split('\n')
        text_lis2 = text_1.split('\n')
        info_dict = {}
        for i,j in enumerate(text_lis1):
            if 'Picked up' in j:
                info_dict['Date'] = j.split()[-3:]
            elif 'Passenger' in j:
                info_dict['Name'] = text_lis1[i+1]
        
        for i in range(len(text_lis2)-1,-1,-1):
            if text_lis2[i] == '⋮':
                text_lis2.remove('⋮')
        print(text_lis2)

        index_trip = text_lis2.index('Your Trip')
        info_dict['distance and durance'] = text_lis2[index_trip+1]
        info_dict['start_place'] = text_lis2[index_trip+2]
        info_dict['start_time'] = text_lis2[index_trip+3]
        info_dict['end_place'] = text_lis2[index_trip+4]
        info_dict['end_time'] = text_lis2[index_trip+5]


        return info_dict
    
    def extract_gojek(self,doc):
        page_0 = doc.load_page(0)
        text_0 = page_0.get_text('text')
        text_lis0 = text_0.split('\n')
 

        return text_lis0





        