import fitz
import cv2
import numpy as np
import re
from commons.logger import logger as logging
from paddleocr import PaddleOCR
# from pytesseract import Output
# import pytesseract
from collections import defaultdict
from PIL import Image



class receipt():

    def __init__(self,input,model,name):
        self.content = input
        self.model = model
        self.filename = name
    
    def chose_file(self):
        if self.filename.split('.')[-1] == 'pdf':
            doc = fitz.open(stream=self.content,filetype='pdf')
            page = doc[0]
            text = page.get_text('text')
            if len(text)>20:
                output = self.process_rawpdf(doc)
                return output
            else:
                mat = fitz.Matrix(2,2)
                pm = page.get_pixmap(matrix=mat,alpha=False)
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
                img = Image.frombytes("RGB",[pm.width,pm.height],pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                output = self.process_image(img)
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
        text_content = ' '.join([i[1][0] for i in result[0]])
        output = self.extract_bluebird(result)
        return output


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
        info_dict = {}
        for i,j in enumerate(text_lis0):
            if 'Total paid' in j:
                info_dict['fare'] = text_lis0[i+1]
            elif 'Trip details' in j:
                info_dict['name'] = text_lis0[i+1]
            elif 'Distance' in j:
                info_dict['distance'] = ''.join(text_lis0[i].split(' ')[-2:])
            elif 'Picked up' in j:
                info_dict['start time'] = text_lis0[i].split(' ')[-2]
                info_dict['start place'] = text_lis0[i+1]
            elif 'Arrived at' in j:
                info_dict['end time'] = text_lis0[i].split(' ')[-2]
                info_dict['end place'] = text_lis0[i+1]
 

        return info_dict


    def extract_bluebird(self,result):
        recep_info = {}
        time_list = []
        date_list = []
      
        text = ' '.join([i[1][0] for i in result[0]])
        time_list = re.findall(r'\d\d:\d\d',text)
        date_list = re.findall(r'\d\d/\d\d/\d\d\d\d',text)
        recep_info['start_time'],recep_info['end_time'] = time_list[0],time_list[1]
        recep_info['date']=date_list[0]

        for i in result:
            if 'Km' in i[1][0]:
                recep_info['Distance'] = i[1][0]
        
        return recep_info
            
                
        




        