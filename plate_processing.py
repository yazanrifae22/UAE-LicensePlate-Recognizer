import cv2
import numpy as np
import uuid
from ultralytics import YOLO
from utils import next_multiple_of_32, sort_by_x, city_name_replacement
from city_replacements import city_replacements

def process_stream(stream_url, camera_mac_address):
    start_time = time.time()
    rtsp_link = stream_url

    # Initialize video capture
    video = cv2.VideoCapture(rtsp_link)
    
    while not video.isOpened():
        print("Error opening the stream. Retrying...")
        time.sleep(5)
        video = cv2.VideoCapture(rtsp_link)

    previous_frame = None

    # Initialize lists for detection data
    all_detected_numbers, all_detected_codes, all_detected_cities = [], [], []
    plate_detection, per_plate_num = [], []
    
    # Load YOLO model
    model = YOLO(weights='your yolo v8 model.pt')

    while True:
        try:
            status, frame = video.read()
            
            if not status:
                break

            height, width, _ = frame.shape
            min_x, min_y = width, height
            max_width, max_height = 0, 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if previous_frame is None:
                previous_frame = gray
                continue

            diff = cv2.absdiff(previous_frame, gray)
            previous_frame = gray

            thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=10)

            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in cnts:
                if cv2.contourArea(contour) < 20000:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_width = max(max_width, x + w)
                max_height = max(max_height, y + h)

            if min_x != width:
                crop_img = frame[min_y:max_height, min_x:max_width]
                h, w, c = crop_img.shape
                w = 128 if h < 79 else 256

                height = next_multiple_of_32(h)
                width = next_multiple_of_32(w)

                results = model.predict(np.ascontiguousarray(frame), imgsz=1024, conf=0.2)

                for det in results:
                    for pred in range(len(det)):
                        x1 = int(det.boxes[pred].xyxy[0][0].item())
                        y1 = int(det.boxes[pred].xyxy[0][1].item())
                        x2 = int(det.boxes[pred].xyxy[0][2].item())
                        y2 = int(det.boxes[pred].xyxy[0][3].item())
                        conf = det.boxes[pred].conf
                        
                        plate_detection.append({'detected_data': det.names[int(det.boxes.cls[pred].item())], 'x': x1, 'width': x2, 'y': y1, 'height': y2, 'conf': conf})

                    plate_detection.sort(key=sort_by_x)

                    for detected_item in plate_detection:
                        if detected_item["detected_data"] in ["back_car", "front_car"]:
                            plate_x1, plate_y1, plate_x2, plate_y2 = detected_item['x'], detected_item['y'], detected_item['width'], detected_item['height']

                        if detected_item["detected_data"] == "plate":
                            # Clear previous data
                            all_detected_numbers.clear()
                            all_detected_codes.clear()
                            all_detected_cities.clear()
                            per_plate_num.clear()

                            cv2.putText(frame, 'Plate', (int(detected_item['x']), int(detected_item['y']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

                            if min_x != width:
                                plate_x1, plate_y1, plate_x2, plate_y2 = detected_item['x'], detected_item['y'], detected_item['width'], detected_item['height']
                                cv2.rectangle(frame, (int(detected_item['x']), int(detected_item['y'])), (int(detected_item['width']), int(detected_item['height'])), (255, 255, 0), 3)

                                plate_image = frame[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                                unique_filename = str(uuid.uuid4())

                                try:
                                    cv2.imwrite("uae_only_plate/" + unique_filename + '.jpg', plate_image)
                                except:
                                    continue

                                results_plate = model1.predict(np.ascontiguousarray(plate_image), imgsz=[192], conf=0.2)

                                for plate_det in results_plate:
                                    for pred in range(len(plate_det)):
                                        data_x1 = plate_det.boxes[pred].xyxyn[0][0].item()
                                        data_y1 = plate_det.boxes[pred].xyxyn[0][1].item()
                                        data_x2 = plate_det.boxes[pred].xyxyn[0][2].item()
                                        data_y2 = plate_det.boxes[pred].xyxyn[0][3].item()
                                        confd = plate_det.boxes[pred].conf

                                        plate_data.append({'detected_data': plate_det.names[int(plate_det.boxes.cls[pred].item())], 'x': data_x1, 'width': data_x2, 'y': data_y1, 'height': data_y2, 'conf': confd})

                                plate_data.sort(key=sort_by_x)

                                for detected_items in plate_data:
                                    if detected_items["detected_data"].isdigit():
                                        all_detected_numbers.append({"num": detected_items["detected_data"], "x": detected_items["x"], "width": detected_items["width"], "y": detected_items["y"]})

                                    if detected_items["detected_data"].isalpha():
                                        all_detected_codes.append({"char": detected_items["detected_data"], "x": detected_items["x"], "width": detected_items["width"], "y": detected_items["y"]})

                                    if detected_items["detected_data"] in city_replacements:
                                        all_detected_cities.append({"city": city_name_replacement(detected_items["detected_data"]), "x": detected_items["x"], "width": detected_items["width"], "y": detected_items["y"]})

                                for y in range(len(all_detected_numbers)):
                                    per_plate_num.append(all_detected_numbers[y]["num"])

                                if per_plate_num and all_detected_codes:
                                    file_name = camera_mac_address.replace(":", "_")
                                    save_path_image = f"{file_name}image.jpeg"
                                    cv2.imwrite(save_path_image, crop_img)

                                    response = {
                                        "MACAddress": camera_mac_address,
                                        "plateNumber": ''.join(per_plate_num),
                                        "plateCharacter": ''.join(all_detected_codes),
                                        "plateCity": ''.join([city["city"] for city in all_detected_cities]),
                                    }

                                    print(response)
                                    print("==============================")

                    cv2.imshow(str(camera_mac_address), frame)
                    all_detected_numbers.clear()
                    all_detected_codes.clear()
                    all_detected_cities.clear()
                    plate_data.clear()
                    plate_detection.clear()
                    per_plate_num.clear()
                    min_x, min_y = 1000000000, 1000000000
                    max_width, max_height = 0, 0

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                else:
                    video.release()
                    print("--- %s seconds ---" % (time.time() - start_time))
                    cv2.destroyAllWindows()
                    break

        except Exception as e:
            print("Error extracting frames:", e)
            time.sleep(5)

    video.release()
    cv2.destroyAllWindows()
