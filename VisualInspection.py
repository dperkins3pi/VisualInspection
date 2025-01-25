"""
ECEn-631 Visual Inspection Project created by Harrison Garrett in 2020

"""

import cv2 as cv
import numpy as np
'''
Set WEBCAM to 1 to use your webcam or 0 to use the Flea2 cameras on the lab machine
Set CATCHER to 1 to use the catcher connected to the lab machine or 0 to use your own computer
'''
WEBCAM = 1
CATCHER = 0
if WEBCAM:
    camera = cv.VideoCapture(0)
else:
    from src.Flea2Camera2 import FleaCam
    camera = FleaCam()

while True:
    # Get Opencv Frame
    if WEBCAM:
        ret0, frame = camera.read()
    else:
        frame = camera.getFrame()

    # print(cv_image.shape)
    cv.imshow('frame',frame)


    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
# ssh dbp52@ssh.et.byu.edu