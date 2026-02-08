# Building a lane perception thingy that can take dashcam video as input and highlight lanes from it. 
import cv2
import numpy as np  
import matplotlib.pyplot as plt 

# Importing test_lane image and converting to grayscale

img_test_lane = cv2.imread('./Images/test_lane.jpg')

# Check if image is loaded or not

if img_test_lane is None: 
    print("Error - no image found")

else: # Convert to grayscale
    img_test_lane_gray = cv2.cvtColor(img_test_lane, cv2.COLOR_BGR2GRAY)
    
plt.imshow(img_test_lane_gray, cmap='gray')
plt.show()
