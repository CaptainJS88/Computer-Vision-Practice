# Lane Detection - A Learning Project 

Hey! This is my attempt at learning the basics of lane detection in computer vision. I'm building a simple program that can detect and highlight lane markings from road images and dashcam videos.

## What This Does

The program takes a road image or video and draws blue lines over the detected lane markings. It's pretty basic right now, but it's helping me understand how computer vision works!

## How It Works (The Simple Version)

1. **Load the image/video** - Pretty straightforward, just reading the file
2. **Convert to grayscale** - Makes it easier to work with
3. **Blur it** - Smooths out the image so we don't mistake gravel or rough surfaces for lane lines
4. **Find edges** - Uses something called the Canny edge detector to find sharp changes (like where white paint meets dark asphalt)
5. **Focus on the road** - Creates a triangular mask to ignore everything above the horizon (trees, sky, etc.)
6. **Find the lines** - Uses Hough Transform to turn those edges into actual line segments
7. **Draw them** - Averages the left and right lane lines and draws them on the original image

## Requirements

You'll need Python and these libraries:
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`

Install them with:
```bash
pip install opencv-python numpy matplotlib
```

## How to Use

1. Put your road images in the `Images/` folder
2. Put your dashcam videos in the `Videos/` folder
3. Run the script:
   ```bash
   python lane_detector.py
   ```

The script will:
- Process the test image and show you a visualization of each step
- Process the video and save the result to `Videos/output_lanes.mp4`
- Press 'q' while the video window is open to quit early

## What I'm Learning

- Edge detection (Canny algorithm)
- Region of interest masking
- Hough Transform for line detection
- Basic image processing pipelines
- Working with videos frame by frame

## Current Limitations

This is a learning project, so it's pretty basic:
- Fixed thresholds that might not work in all lighting conditions
- Simple line averaging (no fancy curve fitting yet)
- Hardcoded region of interest (doesn't adapt to different camera angles)
- Only works well on relatively clear roads with visible lane markings

But hey, it's a start! 
