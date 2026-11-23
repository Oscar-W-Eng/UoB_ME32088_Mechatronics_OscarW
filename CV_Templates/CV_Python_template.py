"""
This script captures camera frames, displays color and grayscale images, and reports the frame rate.

@author: ioannisgeorgilas
"""

# This is the vision library OpenCV
import cv2
import logging
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities
import time

import cv2.aruco as aruco
# sth sth sth

logging.basicConfig(level=logging.INFO)

# Select the first camera (0) that is connected to the machine
# in Laptops should be the built-in camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    logging.error("Could not open camera 0")
    raise SystemExit(1)
logging.info("Camera 0 opened")

# Set the width and height of the camera to 1920x1080
cap.set(3,1920)
cap.set(4,1080)
logging.info(
    "Camera resolution: %dx%d",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

# Load camera calibration data
camera_calibration=np.load('Sample_Calibration.npz') 
CM=camera_calibration['CM'] #camera matrix 
dist_coef=camera_calibration['dist_coef']

marker_size = 40 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) 
parameters = aruco.DetectorParameters() 
detector = aruco.ArucoDetector(aruco_dict, parameters)

half = marker_size / 2 
marker_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32) 


#Create two opencv named windows
cv2.namedWindow("frame-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("gray-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("canny-image", cv2.WINDOW_AUTOSIZE)

#Position the windows next to each other
cv2.moveWindow("frame-image",0,100)
cv2.moveWindow("gray-image",640,100)
cv2.moveWindow("canny-image",1280,100)

# Execute this continuously
last_fps_log = time.perf_counter()
while(True):
    
    # Start the performance clock
    start = time.perf_counter()
    
    # Capture current frame from the camera
    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read a frame from camera 0")
        break
    
    # Convert the image from the camera to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    out = aruco.drawDetectedMarkers(frame, corners, ids)

    if len(corners) > 0:
        ok, rvec, tvec = cv2.solvePnP(
            marker_points,
            corners[0],
            CM,
            dist_coef,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if ok:
            print(f"Marker {ids.flatten()[0]} translation (mm): {tvec.flatten()}: {rvec.flatten()}")

            out = cv2.drawFrameAxes(
                out, CM, dist_coef, rvec, tvec, 10
            )

   

    # Apply Canny edge detection
    canny = cv2.Canny(gray,50,100)
    
    # Display the original frame in a window
    cv2.imshow('frame-image',frame)

    # Display the Canny edge-detected image in another window
    cv2.imshow('canny-image',canny)

    # Display the grey image in another window
    cv2.imshow('gray-image',gray)

    cv2.imshow('out', out)

   
    
    
    # Stop the performance counter
    end = time.perf_counter()
    
    # Report the frame rate once per second rather than on every frame
    if end - last_fps_log >= 1.0:
        logging.info("Camera FPS: %.2f", 1 / (end - start))
        last_fps_log = end

    # If the button q is pressed in one of the windows 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        # Exit the While loop
        break
    

# When everything done, release the capture
cap.release()
# close all windows
cv2.destroyAllWindows()
logging.info("Camera capture stopped")
# exit the kernel
exit(0)