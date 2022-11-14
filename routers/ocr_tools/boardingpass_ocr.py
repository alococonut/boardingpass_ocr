from commons.logger import logger as logging
import re
from paddleocr import draw_ocr
# from pytesseract import Output
# import pytesseract
from collections import defaultdict

class ocr():
    
    def __init__(self,input,model):
        self.image = input
        self.model = model
    def processs(self):
        # h,w = self.image.shape
        # if h>w:
        #     rotation_angle = pytesseract.image_to_osd(self.image,output_type = Output.DICT)['rotate']
        #     print(rotation_angle)
        #     if rotation_angle == 90:
        #             self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        #     elif rotation_angle == 270:
        #             self.image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        result = self.model.ocr(self.image)
        content_list = [i[1][0] for i in result]
        content = ' '.join(content_list)


        if re.search(r'Garuda.*Indonesia',content):
            output = self.Garuda_Indonesia(result)
        elif re.search(r'SQ',content) and re.search(r'S.*PORE ECONOMY CLASS',content):
            output = self.Singpore_Airline(result)
        elif re.search(r'Batik.*air',content) or re.search(r'Lion (.*)air',content):
            output = self.batik_ticket(result)
        elif re.search(r'malaysia.* ECONOMY CLASS',content):
            output = self.malaysia_ticket(result)
        elif re.search(r'SUPER AIR JET',content):
            output =self.superairjet(result)
        elif re.search(r'Air.*Asia',content):
            output = self.airasia(result)
        elif re.search(r'QATAR',content):
            output = self.qatar_ticket(result)

        if len(output['Flight_number']) == 5:
            output['Flight_number'] = output['Flight_number'][:2]+'0'+output['Flight_number'][2:]
        
        return output

        
    def Singpore_Airline(self,result):
        output = defaultdict(list)
        #get the location of keywords.
        name_loc = [i[0] for i in result if 'AIRLINE' in i[1][0]][0]
        time_loc = [i[0] for i in result if 'Boarding ti' in i[1][0]][0]

        for i in result:
            if name_loc[0][0]>i[0][0][0] and i[0][0][1]>name_loc[2][1]:
                output['Name'].append(i[1][0])
            if i[0][0][0]<time_loc[1][0] and i[0][1][0]>time_loc[0][0] and time_loc[2][1]<i[0][0][1]:
                output['time'].append(i[1][0])
            
        output['Name'] = output['Name'][0]
        output['Flight_number'] = [i[1][0] for i in result if 'Flight' in i[1][0]][0]    
        output['Date'] = [i[1][0] for i in result if 'ate' in i[1][0]][0] 
        output['time'] = output['time'][0]

        #postprocessing
        output['Flight_number'] = ''.join(output['Flight_number'].split(' ')[1:])
        output['Date'] = output['Date'][-7:]
        output['Day'] = output['Date'][:2]
        output['Month'] = output['Date'][2:5].replace('0','O')
        output['Year'] = '20'+output['Date'][-2:]
        output.pop('Date')
        
        return output

    

    def Garuda_Indonesia(self,result):

        output = defaultdict(list)
        #get the location of keywords.If the boarding pass have two keywords,choose the left one   
        name_loc = [ i[0] for i in result if 'Name' in i[1][0] ][0]
        flight_loc = [ i[0] for i in result if 'Flight' in i[1][0] ][0]
        time_loc = [ i[0] for i in result if 'Boarding' in i[1][0] ][0]
        from_loc = [ i[0] for i in result if 'From' in i[1][0] ][0]
        to_loc = [ i[0] for i in result if 'To' in i[1][0] ][0]



        #accroding the location fof keywords,extract the needed information
        for i in result:
            if name_loc[2][1]<(i[0][0][1]+i[0][2][1])/2 < flight_loc[0][1] and i[0][0][0] < name_loc[1][0]:
                output['Name'].append(i[1][0])
            
            elif flight_loc[2][1]<(i[0][0][1]+i[0][2][1])/2<from_loc[0][1] and i[0][0][0] < flight_loc[1][0]:
                output['Flight_number'] = i[1][0]
            
            elif time_loc[2][1]<(i[0][0][1]+i[0][2][1])/2<to_loc[0][1] and i[0][0][0]<time_loc[1][0] and \
                i[0][1][0]>time_loc[0][0]:
                output['time'].append(i[1][0])
        
        #postprocesing
        if len(output['Name']) == 1:
            output['Name']=output['Name'][0].split(sep=' ')[0].replace('/',' ')


        if len(output['time']) == 1:
            output['Date'] = output['time'][0][5:]
            output['time'] = output['time'][0][:5]
            output['Month'] = output['Date'][2:].replace('0','O')
            output['Day'] = output['Date'][:2]
            output.pop('Date',None)     

        return output


    
    def batik_ticket(self,result):
        output = defaultdict(list)

        loc = [i[0] for i in result if 'BOARDING PASS' in i[1][0]][0]

        for i in result:
            if i[0][0][0] < loc[1][0] and i[0][0][1]>loc[2][1]: 
                output['content'].append(i[1][0])
        
        output['Name'] = output['content'][0]
        output['Flight_number'] = output['content'][2]
        time_lis = output['content'][3].split(' ')
        output['Day'] = time_lis[1]
        output['Month'] = time_lis[0]
        output['Year'] = time_lis[2]
        output['time'] = [i[1][0] for i in result if 'Departure Time' in i[1][0]][0][-5:]
        output.pop('content')
        
        return output


    
    def malaysia_ticket(self,result):
        output = defaultdict(list)

        time_loc = [i[0] for i in result if 'DEPARTURE' in i[1][0]][0]
        flight_loc = [i[0] for i in result if 'Flight' in i[1][0]][0]
        for i in result:
            if i[0][0][0]<flight_loc[1][0] and i[0][0][1]>flight_loc[2][1]:
                output['Flight_number'].append(i[1][0])
            elif i[0][0][0]<time_loc[1][0] and i[0][0][1]>time_loc[2][1] and i[0][2][1]<flight_loc[0][1]:
                output['time'].append(i[1][0])

        output['Name'] = [i[1][0] for i in result if 'Name' in i[1][0]][0][6:-3]
        output['Flight_number'] = output['Flight_number'][0]
        time_list = output['time'][1].split(' ')
        output['Day'] = time_list[0]
        output['Month'] = time_list[1]
        output['Year'] = time_list[2]
        output['time'] = output['time'][0]
        return output
    
    def superairjet(self,result):
        output = defaultdict(list)

        name_loc = [i[0] for i in result if 'BOARDING PASS' in i[1][0]][0]
        flight_loc = [i[0] for i in result if 'Penerbangan' in i[1][0]][0]

        for i in result:
            if i[0][0][0]<name_loc[1][0] and i[0][0][1]>name_loc[2][1]:
                output['content'].append(i[1][0])
        
        output['Name'] = output['content'][0]
        output['Flight_number'] = output['content'][2]
        time_lis= output['content'][3].split()
        output['Day'],output['Month'],output['Year'] = time_lis[1],time_lis[0],time_lis[2]
        output['time'] = [i[1][0] for i in result if 'Waktu Keberangkatan' in i[1][0]][0][-5:]
        output.pop('content')
        return output
    
    def airasia(self,result):
        output = defaultdict(list)

        name_loc = [ i[0] for i in result if 'AirAsia' in i[1][0] ][0]
        depart_loc = [ i[0] for i in result if 'Depart' in i[1][0] ][0]
        booking_loc = [ i[0] for i in result if 'Booking no' in i[1][0] ][0]
        arrive_loc = [ i[0] for i in result if 'Arrive' in i[1][0] ][0]
        flight_loc =  [ i[0] for i in result if 'Jakarta(CGK)' in i[1][0] ][0]
    
        
        for i in result:
            print(i)    
            if name_loc[0][1]<(i[0][0][1]+i[0][2][1])/2< depart_loc[0][1] and i[0][0][0] < name_loc[1][0]:
                output['Name'] = i[1][0]
            if booking_loc[0][1]>(i[0][0][1]+i[0][2][1])/2:
                output['Time'] = i[1][0]
            if arrive_loc[0][1]>(i[0][0][1]+i[0][2][1])/2:
                output['Date'] = i[1][0]
            if flight_loc[0][1]>(i[0][0][1]+i[0][2][1])/2: 
                output['Flight_number'] = i[1][0]
        
        output['Day'] = output['Date'].split(' ')[0]
        output['Month'] = output['Date'].split(' ')[1]
        output['Year'] = '20'+output['Date'].split(' ')[2]
    
    
        return output
    
    def qatar_ticket(self,result):

        output = {}
        #get the location of keywords.If the boarding pass have two keywords,choose the left one   
        name_loc = [ i[0] for i in result if 'Name' in i[1][0] ][0]
        boarding_loc = [ i[0] for i in result if 'Boarding' in i[1][0] ][0]
        flight_loc = [ i[0] for i in result if 'Flight' in i[1][0] ][0]

            
        #accroding the location fof keywords,extract the needed information
        output['Name'] = ''
        for i in result:
            if name_loc[0][1]<(i[0][0][1]+i[0][2][1])/2<boarding_loc[0][1] and i[0][0][0] < name_loc[1][0] and i[0][0][1] > name_loc[0][1]:
                output['Name'] = output['Name']+' '+i[1][0]
                #name_loc2 = [ i[0] for i in result if output['name'][0] in i[1][0] ][0]
                    
                
            elif i[0][0][1] > boarding_loc[1][0] and boarding_loc[3][0]< i[0][0][0] < boarding_loc[1][0]:
                output['time'] = i[1][0][:2]+':'+i[1][0][2:4]
            elif i[0][0][1] < flight_loc[1][0] and flight_loc[3][0]< i[0][0][0] < flight_loc[1][0]:
                output['Flight_number'] = i[1][0]
                flight_loc2 = [ i[0] for i in result if output['Flight_number'] in i[1][0] ][0]            
                
        #output['name'] = [j for j in result if name_loc[0][1]<(j[0][0][1]+j[0][2][1])/2<name_loc2[0][1] and j[0][0][0] < name_loc[1][0]][1][1][0]+' '+output['name'] 
        output['Flight_number'] = output['Flight_number']+[k for k in result if flight_loc2[0][0]<(k[0][0][0]+k[0][1][0])/2<flight_loc2[1][0] and k[0][2][1]>flight_loc2[2][1] ][0][1][0]
        
        return output
    


if __name__ == "__main__":
    pass
    