import random
from ultralytics import YOLO
import cv2
import requests
from url import *
from PIL import Image
import numpy as np
import time
import math


import threading

#from tele import send_to_telegram


def next_multiple_of_32(n: int) -> int:
        return (n + 31) // 32 * 32
def myFunc(e):
        return e['x']
def yFunc(e):
        return e['y']
    


def ANPR_with_motion(stream_url,cam_mac):

    rtsp_linke=stream_url
   
        
    video = cv2.VideoCapture(rtsp_linke)


    previous_frame=None
    maxx=1000000000
    maxy=1000000000
    smallh=0
    smallw=0
    all_time=0
    all_detected_number=[]
    all_detected_codes=[]
    all_detected_cities=[]
    detectoin_data=[]
    plate_detectoin=[]
    plate_data = []
    plate_conf=[]
    per_plate_num=[]
    per_plate_num_full_info=[]
    per_plate_code=[]
    per_plate_city=[]
    per_plate_city_full_info=[]
    frame_freez_counter=0
    x=0
    
    model = YOLO("only_plate.pt")
    model1 = YOLO("full_plate_only_last1300.pt")
    if not video.isOpened():
        print("error")
        time.sleep(2) 


        
    while True:
        try:
    
            status, frame = video.read()
            if not status:
                print("Connection to RTSP stream lost. Trying to reconnect... ",rtsp_linke)
                video.release()
                time.sleep(2)
                video = cv2.VideoCapture(rtsp_linke)

                continue
            
            if status:
                
            
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray,(5,5), 0)
                if (previous_frame is None):
                    # First frame; there is no previous one yet
                    previous_frame = gray
                    continue

                diff = cv2.absdiff(previous_frame,gray)
                previous_frame=gray


                thresh = cv2.threshold(diff,10,255,cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations = 1)

                cnts,res = cv2.findContours(thresh.copy(),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in cnts:
                
                    
                    if cv2.contourArea(contour) < 20000 :

                        continue
                    
                    print ("______________________________________________")
                    

                    (x,y,w,h) = cv2.boundingRect(contour)
                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 3)
                    if x<maxx:
                        maxx=x
                    if y<maxy:
                        maxy=y
                    if x+w>smallw:
                        smallw=x+w
                    if y+h>smallh:
                        smallh=y+h
                
                
                if maxx!=1000000000  :
                    #cv2.rectangle(frame,(maxx,maxy),(smallw,smallh),(0,255,0), 3)

                    cropimg=frame[maxy:smallh,maxx:smallw]
                    # h,w,c=plate_image.shape
                    
                    
                    # if w >600:
                    # 	w=600
                    
                    # h=next_multiple_of_32(h)
                    # w=next_multiple_of_32(w)
                    
                    
                    results = model.predict(np.ascontiguousarray(cropimg),imgsz=640,conf=0.2,device=0)

                    
                    for det in results:  # per image

                        if len(det):
                    
                            for pred in det:
                            
                                # circle=cv2.circle(cropimg,center_point,5,(255,255,0),2)
                                
                                #(x1,y1) is the top left point of plate 
                                #(x2,y2) is the bottom riight point of plate  
                                x1=int(pred.boxes.xyxy[0][0].item())
                                y1=int(pred.boxes.xyxy[0][1].item())
                                x2=int(pred.boxes.xyxy[0][2].item())
                                y2=int(pred.boxes.xyxy[0][3].item())

                                conf=pred.boxes.conf
                                # print((conf))
        
                                # crop=cropimg[y1:y2,x1:x2]
                                plate_detectoin.append({'detected_data':model.names[int(pred.boxes.cls.item())], 'x':x1,'width':x2,'y':y1,'height':y2,'conf':conf})
                        
                        plate_detectoin.sort(key=myFunc)
                        plate_counter=0
                        for detected_item in plate_detectoin:
                            # if detected_item["detected_data"] == "back_car":
                            #         plate_x1=(detected_item['x'])
                            #         plate_y1=(detected_item['y'])
                            #         plate_x2=(detected_item['width'])
                            #         plate_y2=(detected_item['height'])
                            #         cv2.rectangle(frame,(int(detected_item['x'])+maxx,int(detected_item['y'])+maxy),(int(detected_item['width'])+maxx,int(detected_item['height'])+maxy),(255,0,0), 3)
                            #         cv2.putText(frame, 'Back Car', (int(detected_item['x'])+maxx, int(detected_item['y'])+maxy-10), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2, cv2.LINE_AA)
                
                            # if detected_item["detected_data"] == "front_car":
                            #         plate_x1=(detected_item['x'])
                            #         plate_y1=(detected_item['y'])
                            #         plate_x2=(detected_item['width'])
                            #         plate_y2=(detected_item['height'])
                            #         cv2.rectangle(frame,(int(detected_item['x'])+maxx,int(detected_item['y'])+maxy),(int(detected_item['width'])+maxx,int(detected_item['height'])+maxy),(255,0,0), 3)
                            #         cv2.putText(frame, 'front car', (int(detected_item['x'])+maxx, int(detected_item['y'])+maxy-10), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2, cv2.LINE_AA)
                
                            if detected_item["detected_data"] == "plate":
                                # print(detected_item["detected_data"]+':'+str(detected_item['conf']))
                                all_detected_number.clear()
                                all_detected_codes.clear()
                                all_detected_cities.clear()
                                plate_data.clear()
                                #plate_detectoin.clear()
                                per_plate_num.clear()
                                per_plate_num_full_info.clear()
                                per_plate_code.clear()
                                per_plate_city_full_info.clear()
                                per_plate_city.clear()
                                plate_conf.append({'plate':{'plate_x':detected_item['x'],'plate_y':detected_item['y'],'plate_width':detected_item['width'],'plate_height':detected_item['height']}})                
                                
                                

                                plate_counter=plate_counter+1
                                
                                if maxx!=1000000000:
                                # draw rectangle on frame , get croping points and add them to scailing ppoint 
                                    
                                    plate_x1=(detected_item['x'])
                                    plate_y1=(detected_item['y'])
                                    plate_x2=(detected_item['width'])
                                    plate_y2=(detected_item['height'])
                                    cv2.rectangle(frame,(int(detected_item['x'])+maxx,int(detected_item['y'])+maxy),(int(detected_item['width'])+maxx,int(detected_item['height'])+maxy),(255,255,0), 2)
                                    cv2.putText(frame, 'Plate', (int(detected_item['x'])+maxx, int(detected_item['y'])+maxy-10), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,255,0), 1, cv2.LINE_AA)

                                    
                                
                                plate_image=frame[int(plate_y1)+maxy:int(plate_y2)+maxy,int(plate_x1)+maxx:int(plate_x2)+maxx]
                                cv2.imwrite("ff.jpg",plate_image)
                                h,w,c=plate_image.shape
                                h=next_multiple_of_32(h)
                                




                                if h<90:
                                    w=128
                                else :
                                    w=192
                                w=next_multiple_of_32(w)
                                results_plate = model1.predict(np.ascontiguousarray(plate_image),imgsz=[192],conf=0.4,agnostic_nms=True,device=0)
                                
                                0
                                for plate_det in results_plate:
                                    for pred in range(len(plate_det)):
                                        # for plate_pred in plate_det:
                                        
                                            data_x1=plate_det.boxes[pred].xyxyn[0][0].item()
                                            data_y1=plate_det.boxes[pred].xyxyn[0][1].item()
                                            data_x2=plate_det.boxes[pred].xyxyn[0][2].item()
                                            data_y2=plate_det.boxes[pred].xyxyn[0][3].item()
                                            confd=plate_det.boxes[pred].conf
                                            
                                            plate_data.append({'detected_data':plate_det.names[int(plate_det.boxes.cls[pred].item()) ] , 'x':data_x1,'width':data_x2,'y':data_y1,'height':data_y2,'conf':confd})
                            
                                    
                                    plate_data.sort(key=myFunc)
                                    # Define the sets of valid data
                                    valid_numbers = set(str(i) for i in range(10))
                                    valid_codes = set(chr(i) for i in range(ord('A'), ord('Z')+1))
                                    valid_cities = set(["new_DUBAI", "new_am", "new_RAK", "old_DUBAI", "old_RAK", "old_am", "old_ajman", "new_ajman", "old_fujira", "new_fujairah", "old_abudabi", "new_abudabi", "old_sharka"])

                                    # Iterate over the detected items
                                    for detected_item in plate_data:
                                        detected_data = detected_item["detected_data"]
                                        item_info = {"x": detected_item["x"], "width": detected_item["width"], "y": detected_item["y"]}

                                        # Check the type of the detected data and append it to the corresponding list
                                        if detected_data in valid_numbers:
                                            all_detected_number.append({"num": detected_data, **item_info})
                                        elif detected_data in valid_codes:
                                            all_detected_codes.append({"char": detected_data, **item_info})
                                        elif detected_data in valid_cities:
                                            all_detected_cities.append({"city": detected_data, **item_info})

                                    # Extract the data from the detected items
                                    per_plate_num = [item["num"] for item in all_detected_number]
                                    per_plate_num_full_info = all_detected_number.copy()
                                    per_plate_code = [item["char"] for item in all_detected_codes]
                                    per_plate_city_full_info = all_detected_cities.copy()
                                    per_plate_city = [item["city"] for item in all_detected_cities]

                                    # Process the data based on the city
                                    if per_plate_city and per_plate_city[0] in ["old_abudabi", "new_abudabi", "old_sharka"]:
                                        if len(per_plate_num_full_info) > 2:
                                            if per_plate_num_full_info[1]['x'] - per_plate_num_full_info[0]['x'] > per_plate_city_full_info[0]['width'] - per_plate_city_full_info[0]['x']:
                                                per_plate_code.append(per_plate_num[0])
                                                del per_plate_num[0]
                                            elif per_plate_num_full_info[2]['x'] - per_plate_num_full_info[1]['x'] > per_plate_city_full_info[0]['width'] - per_plate_city_full_info[0]['x']:
                                                per_plate_code.append(per_plate_num[0] + per_plate_num[1])
                                                del per_plate_num[1]
                                                del per_plate_num[0]
                                            else:
                                                per_plate_num_full_info.sort(key=lambda item: item['y'])
                                                if len(per_plate_num_full_info) > 2:
                                                    if per_plate_num_full_info[1]['y'] - per_plate_num_full_info[0]['y'] > 0.01:
                                                        per_plate_code.append(per_plate_num_full_info[0]['num'])
                                                        del per_plate_num_full_info[0]
                                                        per_plate_num = [item['num'] for item in per_plate_num_full_info]
                                                    elif per_plate_num_full_info[2]['y'] - per_plate_num_full_info[1]['y'] > 0.01:
                                                        if per_plate_num_full_info[0]['x'] > per_plate_num_full_info[1]['x']:
                                                            per_plate_code.append(per_plate_num_full_info[1]['num'] + per_plate_num_full_info[0]['num'])
                                                        else:
                                                            per_plate_code.append(per_plate_num_full_info[0]['num'] + per_plate_num_full_info[1]['num'])
                                                        del per_plate_num_full_info[1]
                                                        del per_plate_num_full_info[0]
                                                        per_plate_num = [item['num'] for item in per_plate_num_full_info]
                                    if per_plate_num!=[] and per_plate_city!=[]:
                                        fileName=cam_mac.replace(*":","_")
                                        save_pathim = str(fileName)+'image.jpeg'
                                        cv2.imwrite(save_pathim, cropimg)
                                        
                                        if per_plate_city[0] == "new_abudabi":
                                            per_plate_city[0]="new_abu-dhabi"
                                        elif per_plate_city[0] == "old_abudabi":
                                            per_plate_city[0]="old_abu-dhabi"
                                        elif per_plate_city[0] == "new_DUBAI":
                                            per_plate_city[0]="new_dubai"
                                        elif per_plate_city[0] == "old_DUBAI":
                                            per_plate_city[0]="old_dubai"
                                        elif per_plate_city[0] == "old_fujira":
                                            per_plate_city[0]="old_fujairah"
                                        elif per_plate_city[0] == "new_fujairah":
                                            per_plate_city[0]="new_fujairah"
                                        elif per_plate_city[0] == "new_RAK":
                                            per_plate_city[0]="new_ras-al-khaima"
                                        elif per_plate_city[0] == "old_RAK":
                                            per_plate_city[0]="old_ras-al-khaima"
                                        elif per_plate_city[0] == "old_sharka":
                                            per_plate_city[0]="old_sharjah"
                                        elif per_plate_city[0] == "new_am":
                                            per_plate_city[0]="new_umm-al-quwain"
                                        elif per_plate_city[0] == "old_am":
                                            per_plate_city[0]="old_umm-al-quwain"
                                        #if(len(str(''.join(per_plate_num)))>3) and (len(str(''.join(per_plate_num)))<6):
                                        if True:
                                            respon={
                                            "MACAddress":cam_mac,
                                            "plateNumber": ''.join(per_plate_num),
                                            "plateCharacter": ''.join(per_plate_code),
                                            "plateCity": ''.join(per_plate_city),
                                            }
                                            #cv2.imwrite(str(''.join(per_plate_num))+"_____"+str(random.randint(0, 2000))+'.jpg',plate_image)
                                            files = {'image':(str(fileName)+'image.jpeg', open(str(fileName)+'image.jpeg', 'rb'),'image/jpeg')}
                                            url = postCarDetile
                                            print (respon)
                                            print("==============================")
                                            #fi={'photo':open(str(fileName)+'image.jpeg','rb')}
                                            #xtte = threading.Thread(target=send_to_telegram, args=(respon,fi))
#                                            xtte.start()
                                        
                                            try:
                                                x = requests.post(url, data = respon,files=files) 
                                            except requests.exceptions.RequestException as e:
                                                print(e)  # This is the correct syntax
                                                raise SystemExit(e)

                                                
                    
                                        
                                            
                #frame= cv2.resize(frame,(500,300))
                # frame=cv2.Canny(frame, 100, 200)                
                #cv2.imshow(str(cam_mac),frame)
                #cv2.waitKey(1)
                all_detected_number.clear()
                all_detected_codes.clear()
                all_detected_cities.clear()
                plate_data.clear()
                plate_detectoin.clear()
                per_plate_num.clear()
                per_plate_num_full_info.clear()
                per_plate_code.clear()
                per_plate_city_full_info.clear()
                per_plate_city.clear()
                maxx=1000000000
                maxy=1000000000
                smallh=0
                smallw=0
                


                
            else:
                print("Unable to connect to RTSP stream. Trying to reconnect in 30 seconds... ",rtsp_linke)
                video.release()
                time.sleep(3)
        except Exception as e:
            print(f"Error: {str(e)}. Reconnecting...")
            video.release()
           
            time.sleep(3)

            continue
   



