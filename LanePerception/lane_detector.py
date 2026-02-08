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
    
# plt.imshow(img_test_lane_gray, cmap='gray')
# plt.show()

#  Next we need to apply a gaussian blur in order to smoothen the image, this is to smoothen the images so that a computer doesn't
#  Mistake gravel or coarse edges for lines. 

img_test_lane_blur = cv2.GaussianBlur(img_test_lane_gray, (5, 5), 0)

# plt.imshow(img_test_lane_blur, cmap='gray')
# plt.show()

# Showing them together to highlight the difference 
fig, axes = plt.subplots(1,2)

# Display the images in the respective subplots
axes[0].imshow(img_test_lane_gray, cmap='gray')
axes[1].imshow(img_test_lane_blur, cmap='gray')

# Optional: Add titles to each subplot
axes[0].set_title('Image gray')
axes[1].set_title('Image blur')

plt.tight_layout()
plt.show()

