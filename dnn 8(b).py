import cv2
import numpy as np

# Read the image
image = cv2.imread("C:/Users/Merwin S/OneDrive/Pictures/Screenshots/car.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to get the foreground mask
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Perform morphological operations to remove noise
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown==255] = 0

# Apply watershed algorithm
markers = cv2.watershed(image, markers)
image[markers == -1] = [255,0,0]  # Mark watershed boundaries in red

# Evaluate performance based on the number of segmented regions
num_regions = np.max(markers) - 1  # Exclude background region
threshold = 10  # Adjust threshold as needed
if num_regions < threshold:
    print("Performance is good: Number of segmented regions is acceptable.")
else:
    print("Performance is bad: Number of segmented regions is too high.")

# Display the result
cv2.imshow('Segmented Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
