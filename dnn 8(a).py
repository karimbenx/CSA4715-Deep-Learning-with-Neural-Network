import cv2
import numpy as np

# Read the image
image = cv2.imread("C:/Users/Merwin S/OneDrive/Pictures/Screenshots/car.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Estimate the background using morphological operations
kernel = np.ones((5,5),np.uint8)
sure_bg = cv2.dilate(blurred, kernel, iterations=3)

# Thresholding to obtain the foreground
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Marker labeling
_, markers = cv2.connectedComponents(thresholded)

# Visualize the results
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred)
cv2.imshow("Sure Background Area", sure_bg)
cv2.imshow("Thresholded Image", thresholded)
cv2.imshow("Marker Labels", markers.astype(np.uint8) * 50)  # Scaling for better visualization
cv2.waitKey(0)
cv2.destroyAllWindows()
