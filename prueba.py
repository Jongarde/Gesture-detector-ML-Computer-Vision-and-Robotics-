import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = 'img/mano2.jpg'  # Replace with the path to your image
original_image = cv2.imread(image_path)

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Apply background subtraction to the image
foreground_mask = bg_subtractor.apply(original_image)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Display the result of background subtraction
plt.subplot(1, 2, 2)
plt.imshow(foreground_mask, cmap='gray')
plt.title('Foreground Mask')

# Show the plots
plt.show()
