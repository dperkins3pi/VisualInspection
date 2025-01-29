import cv2 as cv
import numpy as np
import os
from sklearn.svm import LinearSVC
from Training import get_features

# Configuration variables
WEBCAM = 1
CATCHER = 0
PRODUCT = 'Ugly'
VIDEO_PATH = 'videos/' + PRODUCT + '_Cleaned.mp4'
# VIDEO_PATH = "SmallVideo.mp4"
VIDEO_PATH = None
OUTPUT_DIR = 'contours/' + PRODUCT
WEIGHTS_PATH = 'svc_model.npz'
SAVE_IMAGES = False

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

# Load model weights
print("Load in weights")
loaded_data = np.load(WEIGHTS_PATH)
clf = LinearSVC(random_state=0, tol=1e-5)
clf.coef_ = loaded_data['coef']
clf.intercept_ = loaded_data['intercept']
clf.classes_ = loaded_data['classes'] 
print("Weights loaded in")

# Initialize video capture
if VIDEO_PATH is not None:
    camera = cv.VideoCapture(VIDEO_PATH)
else:
    if WEBCAM:
        camera = cv.VideoCapture(0)
    else:
        from src.Flea2Camera2 import FleaCam
        camera = FleaCam()

if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get frame dimensions
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
videout = cv.VideoWriter('./Video1.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))  # Video format
frame_count = 1

# Create a window and a trackbar for the threshold value
cv.namedWindow('Threshold Adjustments')
cv.createTrackbar("Lower Threshold", "Threshold Adjustments", 100, 255, lambda x: None)
cv.createTrackbar("Upper Threshold", "Threshold Adjustments", 200, 255, lambda x: None)

while True:
    # Get frame from camera
    if not camera.isOpened():
        ret, frame = camera.read()
    else:
        if WEBCAM:
            ret, frame = camera.read()
        else:
            frame = camera.getFrame()
        
    if not ret:  # Break if no frame is available
        print("End of video.")
        break

    ########### Make edits to the frame #####################
    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Get the threshold value from the trackbar
    lower_threshold = cv.getTrackbarPos("Lower Threshold", "Threshold Adjustments")
    upper_threshold = cv.getTrackbarPos("Upper Threshold", "Threshold Adjustments")
    
    # Apply thresholding with dynamic value from the trackbar
    _, thresh = cv.threshold(gray, lower_threshold, upper_threshold, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to determine which ones are valid
    valid_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if 1000 < area < 1000000:  # Example: Filter small and overly large areas
            x, y, w, h = cv.boundingRect(contour)
            if 0 < x < 2500 and 75 < h < 800 and 75 < w < 800 and 15 < y < height-100:
                valid_contours.append((contour, area, x, y, w, h))
                
    # Merge contours that are close to each other
    merged_contours = []
    for i, (contour1, area1, x1, y1, w1, h1) in enumerate(valid_contours):
        keep = True
        for j, (contour2, area2, x2, y2, w2, h2) in enumerate(merged_contours):
            # Check if the bounding boxes are sufficiently close
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx < 500 and dy < 500:  # Define a distance threshold 
                keep = False
                break
        if keep:
            merged_contours.append((contour1, area1, x1, y1, w1, h1))
    
    # Save cropped images if required
    if VIDEO_PATH is not None and SAVE_IMAGES:
        for idx, (contour, area, x, y, w, h) in enumerate(valid_contours):
            crop = frame[y:y + h, x:x + w]
            crop_resized = cv.resize(crop, (128, 128))
            crop_normalized = crop_resized / 255.0
            filename = f"{OUTPUT_DIR}/frame{frame_count}_object{idx}.png"
            cv.imwrite(filename, (crop_normalized * 255).astype(np.uint8))  # Save as an image

    for idx, (contour, area, x, y, w, h) in enumerate(valid_contours):
        crop = frame[y:y + h, x:x + w]
        crop_resized = cv.resize(crop, (128, 128))
        crop_normalized = crop_resized / 255.0
        image = (crop_normalized * 255).astype(np.uint8)
        features = get_features(image).reshape(1, -1)
        prediction = clf.predict(features)[0]

    # Iterate through contours and draw bounding boxes based on prediction
    for contour, area, x, y, w, h in merged_contours:
        if prediction == 0:
            cv.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        elif prediction == 1:
            cv.rectangle(frame, (x, y), (x + w, y + h), YELLOW, 2)
        elif prediction == 2:
            cv.rectangle(frame, (x, y), (x + w, y + h), RED, 2)

    # Show the frame
    cv.imshow('frame', frame)

    # Press Q to exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

    videout.write(frame)
    frame_count += 1

camera.release()
cv.destroyAllWindows()
