import cv2 as cv
import numpy as np
import os

def load_images_from_folder(folder_path):
    images = []  # List to store loaded images
    file_names = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv.imread(file_path)  # Load the image using OpenCV
            if img is not None:  # Ensure the image was loaded successfully
                images.append(img)
                file_names.append(filename)
            else:
                print(f"Warning: Failed to load {file_path}")
    return images, file_names

def calculate_red_proportion(image, lower_red1=(0, 50, 50), lower_red2=(170, 50, 50), 
                            upper_red1=(10, 255, 255), upper_red2=(180, 255, 255)):
    """
    Calculates the proportion of red pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in HSV color space
        lower_red (tuple): Lower bound for red in HSV.
        upper_red (tuple): Upper bound for red in HSV.
            In HSV (Red in HSV typically spans two ranges)
                1. Low range (0-10 H): e.g., (0, 50, 50) to (10, 255, 255)
                2. High range (170-180 H): e.g., (170, 50, 50) to (180, 255, 255)

    Returns:
        float: The proportion of red pixels in the image.
    """
    red_mask1 = cv.inRange(image, lower_red1, upper_red1)   # Create a binary mask for red pixels
    red_mask2 = cv.inRange(image, lower_red2, upper_red2)   # Create a binary mask for red pixels
    red_pixel_count = cv.countNonZero(red_mask1)  # Calculate the number of red pixels
    red_pixel_count += cv.countNonZero(red_mask2)
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    red_proportion = red_pixel_count / total_pixels  # Calculate the proportion of red pixels
    return red_proportion

def calculate_white_proportion(image, lower_white=(0, 0, 200), upper_white=(180, 55, 255)):
    """
    Calculates the proportion of white pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in HSV color space
        lower_white (tuple): Lower bound for white in HSV.
        upper_white (tuple): Upper bound for white in HSV.
            In HSV:
                - Low saturation (S) and high value (V) define white.

    Returns:
        float: The proportion of white pixels in the image.
    """
    white_mask = cv.inRange(image, lower_white, upper_white)  # Create a binary mask for white pixels
    white_pixel_count = cv.countNonZero(white_mask)  # Calculate the number of white pixels
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    white_proportion = white_pixel_count / total_pixels  # Calculate the proportion of white pixels
    return white_proportion

def calculate_green_proportion(image, lower_green=(35, 50, 50), upper_green=(85, 255, 255)):
    """
    Calculates the proportion of green pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in HSV color space
        lower_green (tuple): Lower bound for green in HSV.
        upper_green (tuple): Upper bound for green in HSV.
            In HSV (Green typically spans around 60°-180° H):
                - Green: e.g., (35, 50, 50) to (85, 255, 255)
    
    Returns:
        float: The proportion of green pixels in the image.
    """
    green_mask = cv.inRange(image, lower_green, upper_green)  # Create a binary mask for green pixels
    green_pixel_count = cv.countNonZero(green_mask)  # Calculate the number of green pixels
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    green_proportion = green_pixel_count / total_pixels  # Calculate the proportion of green pixels
    return green_proportion

def calculate_yellow_proportion(image, lower_yellow=(20, 100, 100), upper_yellow=(40, 255, 255)):
    """
    Calculates the proportion of yellow pixels in an image.
    
    Args:
        image (numpy.ndarray): The input image in BGR format.
        lower_yellow (tuple): Lower bound for yellow in HSV.
        upper_yellow (tuple): Upper bound for yellow in HSV.
            In HSV (Yellow typically spans around 20°-40° H):
                - Yellow: e.g., (20, 100, 100) to (40, 255, 255)
    
    Returns:
        float: The proportion of yellow pixels in the image.
    """
    yellow_mask = cv.inRange(image, lower_yellow, upper_yellow)  # Create a binary mask for yellow pixels
    yellow_pixel_count = cv.countNonZero(yellow_mask)  # Calculate the number of yellow pixels
    total_pixels = image.shape[0] * image.shape[1]  # Calculate the total number of pixels in the image
    yellow_proportion = yellow_pixel_count / total_pixels  # Calculate the proportion of yellow pixels
    return yellow_proportion



# Load in the images
good_images, good_files = load_images_from_folder("contours/Good1_Slow")
bad_images, bad_files = load_images_from_folder("contours/Bad_Slow")
# ugly_images, ugly_files = load_images_from_folder("contours/Ugly_Slow")

good_features, bad_features, ugly_features = [], [], []

print(type(good_images[0]))


print(calculate_red_proportion(good_images[0]))

# Process each image
for i, img in enumerate(bad_images):
    # Combine masks for both red ranges
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # Convert BGR image to HSV color space
    red_proportion = calculate_red_proportion(hsv_image)
    white_proportion = calculate_white_proportion(hsv_image)
    green_proportion = calculate_green_proportion(hsv_image)
    print(f"Image {i+1}: Red proportion = {red_proportion:.2%}, White proportion = {white_proportion:.2%}, Green proportion = {green_proportion:.2%}")


cv.imshow("Sample Image", good_images[0])  # Display the first image
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()