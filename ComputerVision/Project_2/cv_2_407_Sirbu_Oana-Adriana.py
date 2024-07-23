import os # import the os module for interacting with the file system
import cv2  # import the OpenCV library for image processing
import glob # the glob module will be used for finding files matching a specified pattern
from ultralytics import YOLO # import the YOLO class from the ultralytics library for object detection 
import xml.etree.ElementTree as ET # import the ElementTree module for parsing XML data
import numpy as np # import the numpy library for numerical operations
from collections import defaultdict # import the defaultdict class from the collections module


# define the MAIN_DIR path, in which the folder with the entire project is located
MAIN_DIR = '/home/oana/Documents/master/1st_year/CV/project_2/CV-2024-Project2'

# the work directory stores the current code and the solutions generated by the code
WORK_DIR = os.path.join(MAIN_DIR, 'work')
print(WORK_DIR)

# for task 1, indicate the paths where the images are located and where the solutions will be saved
task1_images_dir = os.path.join(MAIN_DIR, 'test', 'Task1')
task1_sol_dir = os.path.join(WORK_DIR, 'exam_solutions', 'Task1')
os.makedirs(task1_sol_dir, exist_ok=True)

# same thing for Task 2, except we also need to store the final frames of the videos as images
video_dir = os.path.join(MAIN_DIR, 'test', 'Task2')
task2_data_dir = os.path.join(WORK_DIR, 'data', 'task2_images')
os.makedirs(task2_data_dir, exist_ok=True)
task2_sol_dir = os.path.join(WORK_DIR, 'exam_solutions', 'Task2')
os.makedirs(task2_sol_dir, exist_ok=True)

# also adjust the paths for Task 3 data and solutions
task3_videos_dir = os.path.join(MAIN_DIR, 'test', 'Task3')
task3_sol_dir = os.path.join(WORK_DIR, 'exam_solutions', 'Task3')
os.makedirs(task3_sol_dir, exist_ok=True)

# same changes for Task 4, if applicable
task4_videos_dir = os.path.join(MAIN_DIR, 'test', 'Task4')
task4_sol_dir = os.path.join(WORK_DIR, 'exam_solutions', 'Task4')
os.makedirs(task4_sol_dir, exist_ok=True)


# load the YOLO model from the ultralytics library
model_8 = YOLO("yolov8m.pt")


# the indexes for cars/trucks (object we want to detect) are extracted from yolo classes dictionary
car_class_id = 2
truck_class_id = 7


# parse the XML file, which contains the bounding boxes for the parking spots
tree = ET.parse(os.path.join(WORK_DIR, 'parkings-3.xml'))  
root = tree.getroot()


# piece of code to extract the parking spots from the previously parsed XML file
parking_spots = []
for obj in root.iter('object'):
    name = obj.find('name').text
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    parking_spots.append((name, [xmin, ymin, xmax, ymax]))


# custom function to calculate the Intersection over Union between two bounding boxes
def get_iou(bbox1, bbox2):
    inter_left = max(bbox1[0], bbox2[0])
    inter_top = max(bbox1[1], bbox2[1])
    inter_right = min(bbox1[2], bbox2[2])
    inter_bottom = min(bbox1[3], bbox2[3])

    inter_width = max(0, inter_right - inter_left + 1)
    inter_height = max(0, inter_bottom - inter_top + 1)
    intersection_area = inter_width * inter_height

    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / float(union_area)

    return iou


# TASK 1. 

print("Beginning of Task 1" + "\n")

image_paths = glob.glob(os.path.join(task1_images_dir, "*.jpg"))

# analyze each image in the Task 1 directory
for image_path in image_paths:
    query_path = image_path.replace(".jpg", "_query.txt")
    with open(query_path, 'r') as file:
        query_data = file.readlines()

    img = cv2.imread(image_path)

    # yolov8m model is used to detect objects in an image
    detection_results = model_8.predict(img, show=False)

    # in Task 1, the number of parking spots for which the occupancy needs to be predicted varies,
    # therefore we extract how many we need to predict and which ones of them
    num_spots = int(query_data[0].strip())
    spots_to_check = ['p' + line.strip() for line in query_data[1:num_spots + 1]]

    # initialize all of them as 'free'
    spot_status = {spot: '0' for spot in spots_to_check}

    # check if the bounding box computed by Yolo intersects the parking spot more than 35%
    for result in detection_results:
        for bbox, cls_id in zip(result.boxes.xyxy, result.boxes.cls):  
            if cls_id in [car_class_id, truck_class_id]:
                for spot_name, spot_coords in parking_spots:
                    intersection_over_union = get_iou(bbox, spot_coords)
                    if intersection_over_union > 0.35:  
                        spot_number = int(spot_name[1:])  
                        if f'p{spot_number}' in spot_status:
                            spot_status[f'p{spot_number}'] = '1'

    # write the results in the output file
    output_path = os.path.join(task1_sol_dir, os.path.basename(image_path).replace(".jpg", "_predicted.txt"))
    with open(output_path, 'w') as output_file:
        output_file.write(query_data[0].strip() + "\n")  
        for idx, spot in enumerate(spots_to_check):
            output_line = f"{int(spot[1:])} {spot_status[spot]}"
            if idx < len(spots_to_check) - 1:  
                output_line += "\n"
            output_file.write(output_line)

print("All images for Task 1 were analysed successfully!" + "\n")


# TASK 2.

print("Beginning of Task 2" + "\n")

# we iterate thourgh each video from those 15 available for Task 2
for vid_index in range(1, 16):
    video_path = os.path.join(video_dir, f"{vid_index:02}.mp4")
    video_capture = cv2.VideoCapture(video_path)

    # the total number of frames in the video is extracted 
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # the last frame of the video is captured 
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    success, last_frame = video_capture.read()

    # save the last frame as an image if the video capture is a success
    if success:
        frame_output_path = os.path.join(task2_data_dir, f"{vid_index:02}.jpg")
        cv2.imwrite(frame_output_path, last_frame)
        print(f"Saved last frame of video {vid_index:02}")

    video_capture.release()

# print a message for an easier debugging
print("All frames extracted successfully" + "\n")


def analyze_image(img_file):
    # as in Task 1, the Yolo model is used for object detection
    detection_results = model_8.predict(img_file, show=False)

    # this time we want to predict the occupancy status for all 10 parking spots
    spot_status = {f'p{idx+1}': '0' for idx in range(10)}  

    # check if the bounding box of each detected car or truck has an iou > 35% with the parking spots
    for detection in detection_results:
        for bounding_box, class_id in zip(detection.boxes.xyxy, detection.boxes.cls):
            if class_id in [car_class_id, truck_class_id]:
                for spot_name, coordinates in sorted(parking_spots):
                    intersection_over_union = get_iou(bounding_box, coordinates)
                    if intersection_over_union > 0.35:  
                        spot_number = int(spot_name[1:])  
                        if f'p{spot_number}' in spot_status:
                            spot_status[f'p{spot_number}'] = '1'

    # write the results in the output file
    output_filename = os.path.join(task2_sol_dir, os.path.basename(img_file).replace(".jpg", "_predicted.txt"))
    with open(output_filename, 'w') as file:
        for idx, (spot, status) in enumerate(spot_status.items()):
            if idx == len(spot_status) - 1:
                file.write(f"{status}")  
            else:
                file.write(f"{status}\n")

# analyse each image from the Task 2 final frames captures
image_files_list = sorted(glob.glob(os.path.join(task2_data_dir, "*.jpg")))
for img_file in image_files_list:
    analyze_image(img_file)

print("The end of Task 2, all predictions successfully generated" + "\n")


# TASK 3.

print("Beginning of Task 3" + "\n")

# initialize the CSRT tracker from OpenCV
object_tracker = cv2.TrackerCSRT_create()

video_list = sorted(glob.glob(os.path.join(task3_videos_dir, '*.mp4')))

# iterate through each video file
for vid_file in video_list:
    annotation_file = os.path.splitext(vid_file)[0] + '.txt'
    with open(annotation_file, 'r') as file:
        initial_line = file.readline().strip()
        total_frames = int(initial_line.split()[0])
        _, x1, y1, x2, y2 = map(int, file.readline().split())

    video_capture = cv2.VideoCapture(vid_file)

    # initialize the tracker with the first frame and the initial bounding box
    success, first_frame = video_capture.read()
    if not success:
        print(f"Error reading the first frame from {vid_file}")
        continue

    # convert bounding box format (x1, y1, x2, y2) to (x, y, width, height)
    initial_bbox = (x1, y1, x2 - x1, y2 - y1)
    object_tracker.init(first_frame, initial_bbox)

    # define the output file path in the solution directory
    base_name = os.path.splitext(os.path.basename(annotation_file))[0]
    solution_file = os.path.join(task3_sol_dir, base_name + "_predicted.txt")

    with open(solution_file, 'w') as file:
        # copy the first row from the annotation file (we always need to have this)
        file.write(initial_line + '\n')
        # also write the initial frame number and bounding box
        file.write('{} {} {} {} {}\n'.format(0, x1, y1, x2, y2))

        # track the object in the remaining frames, but adjust the bounding box using Yolo each 10 frames
        for frame_index in range(1, total_frames):
            success, current_frame = video_capture.read()
            if not success:
                print(f"Error reading frame {frame_index} from {vid_file}")
                break

            if frame_index % 10 == 0:
                detection_results = model_8.predict(current_frame, show=False)
                closest_bbox = None
                minimum_distance = float('inf')

                for detection in detection_results:
                    for bbox, class_id in zip(detection.boxes.xyxy, detection.boxes.cls):
                        if class_id in [car_class_id, truck_class_id]:
                            (x, y, x2, y2) = map(int, bbox)
                            center_x = (x + x2) // 2
                            center_y = (y + y2) // 2

                            # compute the distance from the center of the last known bounding box
                            prev_center_x = (initial_bbox[0] + initial_bbox[0] + initial_bbox[2]) // 2
                            prev_center_y = (initial_bbox[1] + initial_bbox[1] + initial_bbox[3]) // 2
                            distance = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5

                            # find the closest YOLO detection to the tracker frame
                            if distance < minimum_distance:
                                minimum_distance = distance
                                closest_bbox = (x, y, x2 - x, y2 - y)

                if closest_bbox:
                    initial_bbox = closest_bbox
                    object_tracker.init(current_frame, initial_bbox)

            # update the tracker with the bounding box from YOLO detection
            success, bbox = object_tracker.update(current_frame)

            if success:
                # if the tracker has found the object, write the bounding box to the file
                (x, y, w, h) = bbox
                x2, y2 = x + w, y + h
                print(f"Frame {frame_index}: Tracked bbox = ({x}, {y}, {x2}, {y2})")
                file.write('{} {} {} {} {}\n'.format(frame_index, int(x), int(y), int(x2), int(y2)))
            else:
                # if the tracker has lost the object, do not write anything in the output file, as specified
                print(f"Frame {frame_index}: Tracking lost")

    video_capture.release()

cv2.destroyAllWindows()

print("End of Task 3!")


# TASK 4.
print("Beginning of Task 4" + "\n")

# define the coordinates of the polygon for the region of interest (ROI)
# the coordinates were computed using a basic Image Viewer tool from Ubuntu
roi_corners = np.array([(389, 208), (497, 213), (950, 597), (798, 609)], dtype=np.int32)

# define a threshold for the number of frames a car should be stationary to be considered stopped
# and the minimum number of frames a car should be stationary to be considered stopped
# (in case the car is not detected by Yolo in all frames)
stationary_threshold = 30
min_stationary_frames = int(stationary_threshold * 0.7)

def detect_vehicles(frame):
    detection_results = model_8.predict(frame, show=False)
    vehicles = []
    for result in detection_results:
        for bbox, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
            if cls_id in [car_class_id, truck_class_id]:
                vehicles.append(tuple(map(int, bbox)))
    return vehicles

# check if a bounding box is within the polygon defined by the ROI corners
def is_within_roi(bbox, roi_corners):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return cv2.pointPolygonTest(roi_corners, (center_x, center_y), False) >= 0

# check if two bounding boxes are approximately the same (they may slightly differ due to Yolo detection)
def is_same_bbox(bbox1, bbox2, tolerance=10):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    return (abs(x1_1 - x1_2) <= tolerance and abs(y1_1 - y1_2) <= tolerance and
            abs(x2_1 - x2_2) <= tolerance and abs(y2_1 - y2_2) <= tolerance)

video_list = sorted(glob.glob(os.path.join(task4_videos_dir, '*.mp4')))

for vid_file in video_list:
    video_capture = cv2.VideoCapture(vid_file)
    base_name = os.path.splitext(os.path.basename(vid_file))[0]
    
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # start processing from the frame index: total_frames - 30
    start_frame = max(0, total_frames - stationary_threshold)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # dictionary to keep track of bounding box occurrences
    bbox_occurrences = defaultdict(int)
    
    frame_index = start_frame
    while video_capture.isOpened() and frame_index < total_frames:
        success, frame = video_capture.read()
        if not success:
            break
        
        vehicles = detect_vehicles(frame)
        
        for vehicle in vehicles:
            if is_within_roi(vehicle, roi_corners):
                found = False
                for known_vehicle in bbox_occurrences.keys():
                    if is_same_bbox(vehicle, known_vehicle):
                        bbox_occurrences[known_vehicle] += 1
                        found = True
                        break
                if not found:
                    bbox_occurrences[vehicle] += 1
        
        frame_index += 1
    
    video_capture.release()
    
    # filter out vehicles that were stationary for less than the minimum required frames
    stationary_vehicles = {vehicle: count for vehicle, count in bbox_occurrences.items() if count >= min_stationary_frames}
    
    output_file_path = os.path.join(task4_sol_dir, f"{base_name}_predicted.txt")
    with open(output_file_path, 'w') as file:
        file.write(f"{len(stationary_vehicles)}\n")
    
cv2.destroyAllWindows()

print("End of Task 4!" + "\n")
print("End of the second project!")