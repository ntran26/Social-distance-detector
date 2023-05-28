# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from functions import social_distancing_config as config
from functions.detection import *
from functions.bird_view_functions import *
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import time
import yaml
import itertools

video = 'videos/people1.mp4'
# video = 1

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

def get_points_from_box(box, centroid):
	"""
	Get the center of the bounding and the point "on the ground"
	@ param = box : 2 points representing the bounding box
	@ return = centroid (x1,y1) and ground point (x2,y2)
	"""
	# Center of the box x = (x1+x2)/2 and y = (y1+y2)/2
	(startX, startY, endX, endY) = box
	(center_x, center_y) = centroid
	# Coordiniate on the point at the bottom center of the box
	center_y_ground = center_y + ((endY - startY)/2)
	return (center_x,int(center_y_ground))

def get_centroids_and_groundpoints(array_boxes_detected, centroids):
	"""
	For every bounding box, compute the centroid and the point located on the bottom center of the box
	@ array_boxes_detected : list containing all our bounding boxes 
	"""
	array_groundpoints = [] # Initialize empty centroid and ground point lists 
	for index,box in enumerate(array_boxes_detected):
		# Draw the bounding box 
		# Get the both important points
		ground_point = get_points_from_box(box, centroids[index])
		array_groundpoints.append(ground_point)
	return array_groundpoints

def draw_rectangle(corner_points):
	# Draw rectangle box over the delimitation area
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)

def bfs(graph, node): 		# function for BFS
	visited = [] 			# List for visited nodes.
	queue = [] 				# init a queue
	visited.append(node)
	queue.append(node)

	while queue:          # Creating loop to visit each node
		m = queue.pop(0) 

		for neighbour in graph[m]:
			if neighbour not in visited:
				visited.append(neighbour)
				queue.append(neighbour)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
ap.add_argument("-v", "--variation", type=str,default="v4",
	help="which kind of YoloV3 variation to be used")
args = vars(ap.parse_args())

##################### Load the config for the top-down view ############################
print("[ Loading config file for the bird view transformation ] ")

with open("config_birdview.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0,0
corner_points = []
for section in cfg:
	corner_points.append(cfg["image_parameters"]["p1"])
	corner_points.append(cfg["image_parameters"]["p2"])
	corner_points.append(cfg["image_parameters"]["p4"])
	corner_points.append(cfg["image_parameters"]["p3"])
	width_og = int(cfg["image_parameters"]["width_og"])
	height_og = int(cfg["image_parameters"]["height_og"])
	img_path = cfg["image_parameters"]["img_path"]
	size_frame = cfg["image_parameters"]["size_frame"]
	# 2 points to define 1.5 meters on the frame (camera view)
	point_1 = cfg["image_parameters"]["p5"]
	point_2 = cfg["image_parameters"]["p6"]
print(" Done : [ Config file loaded ] ...")

##################### load the COCO class labels our YOLO model was trained on ############################
LABELS = open("yolo-coco/coco.names").read().strip().split("\n")
filename = ""
fileWritable = False

print("[INFO] loading YOLO from disk...")

weights = 'yolo-coco/yolov4.weights'
cfg = 'yolo-coco/yolov4.cfg'

net = cv2.dnn.readNetFromDarknet(cfg, weights)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

ln = net.getUnconnectedOutLayersNames()  

###################### Compute transformation matrix ######################

# Compute  transformation matrix from the original frame
matrix,imgOutput = compute_perspective_transform(corner_points,width_og,height_og,cv2.imread(img_path))
height,width,_ = imgOutput.shape
blank_image = np.zeros((height,width,3), np.uint8)
dim = (width, height)

# Apply the transformation matrix to the selected points to acquire the transformed coordinates
d_point_1 = compute_point_perspective_transformation(matrix,point_1)
d_point_2 = compute_point_perspective_transformation(matrix,point_2)

# Calculate the pixel distance between the transformed points
min_dis = int(dist.euclidean(d_point_1, d_point_2))
min_crowd_dis = int((1.8*(dist.euclidean(d_point_1, d_point_2)))/1.5)
print('1.5 meters in pixels: '+ str(min_dis))
print('2 meters in pixels: '+ str(min_crowd_dis))

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(video)

# Configure video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output.mp4', fourcc, 15, (width*2, height+150))

# camera_view_writer = cv2.VideoWriter('camera view.mp4', fourcc, 10, (600, 337))
# bird_view_writer = cv2.VideoWriter('bird view.mp4', fourcc, 10, (600, 337))
# noti_panel_writer = cv2.VideoWriter('notification panel.mp4', fourcc, 10, (1200, 150))

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

#vs = cv2.VideoCapture(args["input"] if args["input"] else 1)
writer = None

crowd_frame_counter = 0

# loop over the frames from the video stream
while True:
	bird_view_img = cv2.resize(blank_image, dim, interpolation = cv2.INTER_AREA)

	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	else:
		# resize the frame and then detect people (and only people) in it
		frame = imutils.resize(frame, width=int(size_frame))
		# print(width, height)

		results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

		d_list = []
		if len(results) > 0:
			centroids = np.array([r[2] for r in results])
			array_boxes_detected = np.array([r[1] for r in results])
			
			# Both of our lists that will contain the centroïds coordonates and the ground points
			array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected, centroids)
			# print(array_groundpoints)
			# Use the transform matrix to get the transformed coordinates
			d_list = compute_point_perspective_transformation(matrix,array_groundpoints)
			# d_list.append(transformed_downoids)

		# initialize the set of indexes that violate the minimum social
		# distance
		distance_violate = set()
		possible_crowd = set()
		crowd = dict()
		#Indicating whether there is a crowd gathered within the frame
		crowd_counter_flag = False
		crowd_bool = False

		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		if len(d_list) >= 2:
			D = dist.cdist(d_list, d_list, metric="euclidean")
			# print(D)
			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number
					# of pixels
					if D[i, j] < min_crowd_dis:
						# update our violation set with the indexes of
						# the centroid pairs
						l = [i,j]
						possible_crowd.add(tuple(l))
						if D[i, j] < min_dis:
							distance_violate.add(i)
							distance_violate.add(j)

		if len(possible_crowd) > 0:
			for i,pair in enumerate(itertools.combinations(possible_crowd, r=2)):
				set_0 = set(pair[0])
				set_1 = set(pair[1])
				inter = set_0.intersection(set_1)
				if (len(inter) > 0):
					difference0_1 = (set_0 - set_1).pop()
					difference1_0 = (set_1 - set_0).pop()
					crowd[inter.pop()] = [difference0_1,difference1_0]
					crowd_frame_counter += 1
		
		# # loop over the results
		for (i, (prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			
			# # if the index pair exists within the violation set, then
			# # update the color
			if i in distance_violate:
				color = (0, 0, 255)
			else:
				color = (0, 255, 0)

			# draw (1) a bounding box around the person and (2) the
			# centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

			point = d_list[i]
			x,y = point
			cv2.circle(bird_view_img, (int(x),int(y)), 5, color, -1)

		visited = [] # List for visited nodes.
		queue = [] # init a queue
		if len(possible_crowd) > 1:
			for i,pair in enumerate(itertools.combinations(possible_crowd, r=2)):
				set_0 = set(pair[0])
				set_1 = set(pair[1])
				inter = set_0.intersection(set_1)
				if (len(inter) > 0):
					difference0_1 = (set_0 - set_1).pop()
					difference1_0 = (set_1 - set_0).pop()
					crowd[inter.pop()] = [difference0_1,difference1_0]
					crowd_frame_counter += 1
			# Traversing the crowd graph to draw the conenction between them 
			if len(crowd) > 0:
				centroids = np.array([r[2] for r in results]) 
				first_node = next(iter(crowd))
				
				visited.append(first_node)
				queue.append(first_node)
				while queue:          # Creating loop to visit each node
					m = queue.pop(0) 
					startX, startY = d_list[m]
					startX_frame, startY_frame = centroids[m]
					if m in crowd:
						for neighbour in crowd[m]:
							endX, endY = d_list[neighbour]
							endX_frame, endY_frame = centroids[neighbour]
							line_color = (255, 128, 0)
							cv2.line(bird_view_img,(int(startX),int(startY)), (int(endX),int(endY)), line_color, 2 )
							cv2.line(frame,(int(startX_frame),int(startY_frame)), (int(endX_frame),int(endY_frame)), line_color, 2 )

							if neighbour not in visited:
								visited.append(neighbour)
								queue.append(neighbour)
			if len(visited) > 0:
				crowd_frame_counter += 1
			else:
				crowd_frame_counter = 0
			# When the crowd still persist for 12 frames (2 seconds)
			if crowd_frame_counter > 12:
				crowd_bool = True	
			else:
				crowd_bool = False		

		# Calculating the fps

		# time when we finish processing for this frame
		new_frame_time = time.time()

		# fps will be number of frame processed in given time frame
		# since their will be most of time error of 0.001 second
		# we will be subtracting it to get more accurate result
		fps = int(1/(new_frame_time-prev_frame_time))
		prev_frame_time = new_frame_time

		# text setup
		font = cv2.FONT_HERSHEY_TRIPLEX
		font_size = 0.7
		font_thickness = 1
		line_type = cv2.LINE_AA

		# create a notification panel at the bottom
		noti_height = 150
		noti_panel = np.zeros((noti_height,width*2,3), np.uint8)
		
		if weights == 'yolo-coco/yolov4.weights':
			algorithm = 'YOLOv4'
		elif weights == 'yolo-coco/yolov4-tiny.weights':
			algorithm = 'YOLOv4-tiny'
		# put title and name of algorithm
		cv2.putText(noti_panel, 'ANALYSIS', (10, 30), font, font_size, COLOR_GREEN, font_thickness, line_type)
		cv2.putText(noti_panel, 'Algorithm: '+ algorithm, (width, noti_panel.shape[0] - 50), font, font_size, COLOR_GREEN, font_thickness, line_type)

		# put FPS count
		cv2.putText(noti_panel, 'FPS: '+ str(fps), (width, noti_panel.shape[0] - 20), font, font_size, COLOR_GREEN, font_thickness, line_type)

		# draw the total number of social distancing violations
		text = "Social Distancing Violations: {}".format(len(distance_violate))
		cv2.putText(noti_panel, text, (10, noti_panel.shape[0] - 50), font, font_size, COLOR_GREEN, font_thickness, line_type)

		# Show if there's crowd gather 
		text = "Crowd People Possibility: {}".format(len(visited))
		cv2.putText(noti_panel, text, (10, noti_panel.shape[0] - 20), font, font_size, COLOR_GREEN, font_thickness, line_type)

		# Display warning for violation
		if len(distance_violate) > 0:
			cv2.putText(noti_panel, '!!! DISTANCE VIOLATION !!!', (10, int(noti_panel.shape[0]/2) - 10), font, font_size, COLOR_RED, font_thickness, line_type)
		if crowd_bool is True:
			cv2.putText(noti_panel, '!!! CROWD VIOLATION !!!', (width, int(noti_panel.shape[0]/2) - 10), font, font_size, COLOR_RED, font_thickness, line_type)
		
		
	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# Draw the green rectangle to delimitate the detection zone
		draw_rectangle(corner_points)

		# Display all frames
		# cv2.imshow("Frame", frame)
		# cv2.moveWindow("Frame", 40,60)
		# cv2.imshow("Bird View", bird_view_img)
		# cv2.moveWindow("Bird View", width+40,60)
		# cv2.imshow("Notification Panel", noti_panel)
		# cv2.moveWindow("Notification Panel",40,height+90)
		combined_view = np.concatenate((frame, bird_view_img), axis=1)
		output_video = np.concatenate((combined_view, noti_panel), axis=0)
		cv2.imshow('Combined', output_video)

		# Write to .mp4 video files 
		video_writer.write(output_video)
		# camera_view_writer.write(frame)
		# bird_view_writer.write(bird_view_img)
		# noti_panel_writer.write(noti_panel)

		# if the 'q' or 'ESC' key was pressed, break from the loop
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q") or key == 27:
			break

cv2.destroyAllWindows()
vs.release()
video_writer.release()
# camera_view_writer.release()
# bird_view_writer.release()
# noti_panel_writer.release()

