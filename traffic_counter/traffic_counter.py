# To read from webcam and write back out to disk:
# python traffic_counter.py --conf config/config.json \
# 	--mode vertical --output output/webcam_output.avi

# import the necessary packages
from complement.directioncounter import DirectionCounter
from complement.centroidtracker import CentroidTracker
from complement.trackableobject import TrackableObject
from complement.utils import Conf
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Value
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

def set_points(event, x, y, flags, param):
	# declare a global variable to store difference point
	global diffPt

	# check if a left button down event has occurred
	if event == cv2.EVENT_LBUTTONDOWN:
		# if the direction is set as vertical, set the difference
		# point as the x-coordinates, otherwise set it as the
		# y-coordinate
		diffPt = x if param[0] == "vertical" else y

def write_video(output, writeVideo, frameQueue, W, H):
	# initialize the FourCC and video writer object
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(output, fourcc, 30,
		(W, H), True)

	# loop while the write flag is set or the output frame queue is
	# not empty
	while writeVideo.value or not frameQueue.empty():
		# check if the output frame queue is not empty
		if not frameQueue.empty():
			# get the frame from the queue and write the frame
			frame = frameQueue.get()
			writer.write(frame)

	# release the video writer object
	writer.release()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="Path to the input configuration file")
ap.add_argument("-m", "--mode", type=str, required=True,
	choices=["horizontal", "vertical"],
	help="direction in which vehicles will be moving")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# initialize the MOG foreground background subtractor object
mog = cv2.bgsegm.createBackgroundSubtractorMOG()

# initialize and define the dilation kernel
dKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# initialize the video writer process
writerProcess = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker and initialize a dictionary to
# map each unique object ID to a trackable object
ct = CentroidTracker(conf["max_disappeared"], conf["max_distance"])
trackableObjects = {}

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# check if the user wants to use the difference flag feature
if conf["diff_flag"]:
	# initialize the start counting flag and mouse click callback
	start = False
	cv2.namedWindow("set_points")
	cv2.setMouseCallback("set_points", set_points,
		[args["mode"]])

# otherwise, the user does not want to use it
else:
	# set the start flag as true indicating to start traffic counting
	start = True

# initialize the direction info variable (used to store information
# such as up/down or left/right vehicle count) and the difference
# point (used to differentiate between left and right lanes)
directionInfo = None
diffPt = None

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()

	#inputがない場合はframeの横幅の値のみ適合、inoutがある場合は高さも横幅も適合
	frame = frame[1] if args.get("input", False) else frame

	# check if the start flag is set, if so, we will start traffic
	# counting
	if start:
		# if the frame dimensions are empty, grab the frame
		# dimensions, instantiate the direction counter, and set the
		# centroid tracker direction
		if W is None or H is None:
			# start the frames per second throughput estimator
			fps = FPS().start()

			(H, W) = frame.shape[:2]
			dc = DirectionCounter(args["mode"],
				W - conf["x_offset"], H - conf["y_offset"])
			ct.direction = args["mode"]

			# check if the difference point is set, if it is, then
			# set it in the centroid tracker object
			if diffPt is not None:
				ct.diffPt = diffPt

		# begin writing the video to disk if required
		if args["output"] is not None and writerProcess is None:
			# set the value of the write flag (used to communicate when
			# to stop the process)
			writeVideo = Value('i', 1)

			# initialize a shared queue for the exhcange frames,
			# initialize a process, and start the process
			frameQueue = Queue()
			writerProcess = Process(target=write_video, args=(
				args["output"], writeVideo, frameQueue, W, H))
			writerProcess.start()

		# initialize a list to store the bounding box rectangles
		# returned by background subtraction model
		rects = []

		# convert the frame to grayscale image and then blur it
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)

		# apply the MOG background subtraction model which returns
		# a mask
		mask = mog.apply(gray)

		# apply dilation
		dilation = cv2.dilate(mask, dKernel, iterations=2)

		# find contours in the mask
		cnts = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# loop over each contour
		for c in cnts:
			# if the contour area is less than the minimum area
			# required then ignore the object
			if cv2.contourArea(c) < conf["min_area"]:
				continue

			# get the (x, y)-coordinates of the contour, along with
			# height and width
			(x, y, w, h) = cv2.boundingRect(c)

			# check if direction is *vertical and the vehicle is
			# further away from the line, if so then, no need to
			# detect it
			if args["mode"] == "vertical" and y < conf["limit"]:
				continue

			# otherwise, check if direction is horizontal and the
			# vehicle is further away from the line, if so then,
			# no need to detect it
			elif args["mode"] == "horizontal" and x > conf["limit"]:
				continue

			# add the bounding box coordinates to the rectangles list
			rects.append((x, y, x + w, y + h))

			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

		# check if the direction is vertical
		if args["mode"] == "vertical":
			# draw a horizontal line in the frame -- once an object
			# crosses this line we will determine whether they were
			# moving 'up' or 'down'


			cv2.line(frame, (0, H - conf["y_offset"]),
				(W, H - conf["y_offset"]), (0, 255, 0), 3)


		# otherwise, the direction is horizontal
		else:
			# draw a vertical line in the frame -- once an object
			# crosses this line we will determine whether they were
			# moving 'left' or 'right'
			cv2.line(frame, (W - conf["x_offset"], 0),
				(W - conf["x_offset"], H), (0, 255, 255), 2)

			# check if a difference point has been set, if so, draw a
			# line dividing the two lanes
			if diffPt is not None:
				cv2.line(frame, (0, diffPt), (W, diffPt),
					(255, 0, 0), 2)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids

		#ct == CentroidTracker
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the
			# current object ID and initialize the color
			to = trackableObjects.get(objectID, None)
			color = (255, 255, 255)

			# create a new trackable object if needed
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can
			# utilize it to determine direction
			else:
				# find the direction and update the list of centroids
				dc.find_direction(to, centroid)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:
					# find the direction of motion of the vehicles
					directionInfo = dc.count_object(to, centroid)

				# otherwise, the object has been counted and set the
				# color to green indicate it has been counted
				else:
					color = (0, 255, 0)

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "vehicle-{}".format(objectID)
			cv2.putText(frame, text, (centroid[0],
				centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				color, 1)


		# extract the traffic counts and write/draw them
		if directionInfo is not None:
			# text = "{}: {} : {}".format(directionInfo[0])
			# cv2.putText(frame, directionInfo[0], (diffPt + 50,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
			for (i, (k, v)) in enumerate(directionInfo):
				# Up : Right
				if i == 0:
					text = "{}: {}".format(k, v)
					cv2.putText(frame, text, (diffPt + 200,  100),
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

				# Down : Left
				elif i == 1:
					text = "{}: {}".format(k, v)
					cv2.putText(frame, text, (diffPt - 500,  100),
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)			

		# put frame into the shared queue for video writing
		if writerProcess is not None:
			frameQueue.put(frame)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	# otherwise, the user has to select a difference point
	else:
		# show the output frame
		cv2.imshow("set_points", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `s` key was pressed, start traffic counting
		if key == ord("s"):
			# begin counting and eliminate the informational window
			start = True
			cv2.destroyWindow("set_points")

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# terminate the video writer process
if writerProcess is not None:
	writeVideo.value = 0
	writerProcess.join()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()