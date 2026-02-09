# Fixes needed before linkedin post
# Show oroginal image in the output and then the rest, reduce the canny edge cases
# Fix the lane detection being wonky and un predictable. 



# Building a lane perception thingy that can take dashcam video as input and highlight lanes from it. 
import cv2
import numpy as np  
import matplotlib.pyplot as plt 

# Function to create a region on interest, which in our case is a triangle. This helps us 
# Mask the other parts of the image like trees, fences, branches, etc. 

def region_of_interest (image): 
    # define height and width of grayscale image
    img_height = image.shape[0]
    img_width = image.shape[1]
    horizon_y = int(img_height * 0.2)

    # Define our triangle polygon (Bottom-Left, Top-Peak, Bottom-Right)
    img_triangle = np.array([[(0, img_height), (img_width // 2, horizon_y), (img_width, img_height)]])

    # Create a black mask of the given width and height
    black_mask = np.zeros_like(image)

    # Fill the triangle with white
    cv2.fillPoly(black_mask, img_triangle, 255)

    masked_image = cv2.bitwise_and(image, black_mask)
    return masked_image

# Helper function to make co-ordinates    
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * (3/5)) # Slightly lower than the middle
    
    # x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

# Helper function to draw lines from the co-ordinates we plant to get from cv2.HoughLinesP
def draw_lines(image, lines): 
    # Create black image of the image with same dimensions
    line_image = np.zeros_like(image)

    left_slopes = []
    right_slopes = []

    if lines is not None: 
        for line in lines:
            # Get co-ordinates
            x1, y1, x2, y2 = line.reshape(4)
            if x2 - x1 == 0: 
                slope_m = 0
            else:
                slope_m = (y2 - y1)/(x2 - x1)
            intercept = y1 - slope_m*x1

            if slope_m <= 0: 
                left_slopes.append([slope_m, intercept])
            else: 
                right_slopes.append([slope_m, intercept])
    
    avg_left_slope = np.average(left_slopes, axis=0)
    avg_right_slope = np.average(right_slopes, axis=0)

    # Calculate coordinates for Left Lane
    if avg_left_slope is not None: # Check if we actually found a line
        left_line = make_coordinates(image, avg_left_slope)
        x1, y1, x2, y2 = left_line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # Calculate coordinates for Right Lane
    if avg_right_slope is not None:
        right_line = make_coordinates(image, avg_right_slope)
        x1, y1, x2, y2 = right_line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image

# Importing test_lane image and converting to grayscale
img_test_lane = cv2.imread('./Images/test_lane.jpg')
img_test_lane_rgb = cv2.cvtColor(img_test_lane, cv2.COLOR_BGR2RGB)

# Check if image is loaded or not
if img_test_lane is None: 
    print("Error - no image found")

else: # Convert to grayscale
    img_test_lane_gray = cv2.cvtColor(img_test_lane, cv2.COLOR_BGR2GRAY)

#  Next we need to apply a gaussian blur in order to smoothen the image, this is to smoothen the images so that a computer doesn't
#  Mistake gravel or coarse edges for lines. 

img_test_lane_blur = cv2.GaussianBlur(img_test_lane_gray, (5, 5), 0)
img_test_lane_median_blur = cv2.medianBlur(img_test_lane_gray, 5, 0)

# Now to find edge detection - for edge detection, we rely on the rate of change or gradient. For example, 
# if there is a sharp change between gravel and white line, there's a gradient there. We use gradients for 
# Edge detection

# Canny Algorithm/Tool!!! 50, 150 are the thresholds.

img_test_lane_canny1 = cv2.Canny(img_test_lane_blur, 50, 150)
img_test_lane_canny2 = cv2.Canny(img_test_lane_blur, 40, 160)
img_test_lane_canny3 = cv2.Canny(img_test_lane_blur, 60, 140)
img_test_lane_cropped = region_of_interest(cv2.Canny(img_test_lane_median_blur, 50, 150))

# Hough Transform to find lines
# rho=2, theta=1 degree, threshold=100 votes
# minLineLength=40 pixels, maxLineGap=5 pixels (connects dashed lines!)
lines = cv2.HoughLinesP(img_test_lane_cropped, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# Draw lines on black image
black_line_image = draw_lines(img_test_lane, lines)

# Showing them together to highlight the difference (2 rows x 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()  # 1D array so we can index 0..5

# Display the images in the respective subplots
images = [
    (img_test_lane_rgb, 'Original'),
    (img_test_lane_gray, 'Grayscale'),
    (img_test_lane_blur, 'Gaussian blur'),
    (img_test_lane_median_blur, 'Median Blur'),
    (img_test_lane_canny1, 'Canny (50, 150)'),
    (img_test_lane_cropped, 'Canny cropped (50, 150)')
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

# Blend the original image with the line image
# 0.8 = keep 80% of the original road brightness
# 1.0 = keep 100% of the blue line brightness
combo_image = cv2.addWeighted(img_test_lane, 0.8, black_line_image, 1, 1)

# Show the final result
# We convert BGR (OpenCV format) to RGB (Matplotlib format) so colors look right
plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
plt.show()

# Video Capture
video_lane = cv2.VideoCapture('./Videos/solidWhiteRight.mp4')

# Set up VideoWriter to save the processed video (same size and FPS as input)
width = int(video_lane.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_lane.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_lane.get(cv2.CAP_PROP_FPS)
output_path = './Videos/output_lanes.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while(video_lane.isOpened()):
   ret, frame = video_lane.read()

   if not ret: 
    break
   else:
    #cv2.imshow("Test frame from video", frame)
    canny_image = cv2.Canny(frame, 50, 150)
    cropped_canny_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    black_line_canny = draw_lines(frame, lines)
    combo_frame = cv2.addWeighted(frame, 0.8, black_line_canny, 1, 1)
    out.write(combo_frame)
    cv2.imshow("Test frame from vid", combo_frame)

   # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

video_lane.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved processed video to {output_path}")




