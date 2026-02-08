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

#  Next we need to apply a gaussian blur in order to smoothen the image, this is to smoothen the images so that a computer doesn't
#  Mistake gravel or coarse edges for lines. 

img_test_lane_blur = cv2.GaussianBlur(img_test_lane_gray, (5, 5), 0)

# Now to find edge detection - for edge detection, we rely on the rate of change or gradient. For example, 
# if there is a sharp change between gravel and white line, there's a gradient there. We use gradients for 
# Edge detection

# Canny Algorithm/Tool!!! 50, 150 are the thresholds.

img_test_lane_canny1 = cv2.Canny(img_test_lane_blur, 50, 150)
img_test_lane_canny2 = cv2.Canny(img_test_lane_blur, 40, 160)
img_test_lane_canny3 = cv2.Canny(img_test_lane_blur, 60, 140)
img_test_lane_cannyNoBlur = cv2.Canny(img_test_lane_gray, 50, 150)


# plt.imshow(img_test_lane_canny, cmap='gray')
# plt.show()

# Showing them together to highlight the difference (2 rows x 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()  # 1D array so we can index 0..5

# Display the images in the respective subplots
images = [
    (img_test_lane_gray, 'Grayscale'),
    (img_test_lane_blur, 'Gaussian blur'),
    (img_test_lane_canny1, 'Canny (50, 150)'),
    (img_test_lane_canny2, 'Canny (40, 160)'),
    (img_test_lane_canny3, 'Canny (60, 140)'),
    (img_test_lane_cannyNoBlur, 'Canny No Blur (50, 150)')
]
for i, (img, title) in enumerate(images):
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(title, fontsize=11)
    axes[i].axis('off')  # cleaner look: no ticks/axes

# Hide the empty 6th subplot
axes[5].axis('off')

fig.suptitle('Lane perception pipeline', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

