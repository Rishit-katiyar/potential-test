import cv2
import numpy as np

# Load the Google Maps screenshot image
input_image_path = r"C:\Users\DELL\Documents\OpenCV\Ain Al-Asad Base - Google Maps - Google Chrome 5_17_2024 8_14_58 PM.png"

# Load the image
google_map_image = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if google_map_image is None:
    print("Error: Image not loaded correctly.")
    exit()

# Convert the image to grayscale
gray_image = cv2.cvtColor(google_map_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and improve contour detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Use Canny edge detection to find edges
edges = cv2.Canny(blurred_image, 50, 150)

# Perform morphological operations to close gaps between edges
kernel = np.ones((5, 5), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours from the edges
contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on hierarchy and size to identify likely buildings
min_contour_area = 500  # Minimum area for a contour to be considered a building
building_contours = []
for contour, hier in zip(contours, hierarchy[0]):
    if cv2.contourArea(contour) > min_contour_area and hier[3] == -1:  # hier[3] == -1 means it has no parent, likely a top-level contour
        building_contours.append(contour)

# Draw bounding rectangles around buildings and label them
building_count = 0
for contour in building_contours:
    # Get the bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Draw the bounding rectangle on the original image
    cv2.rectangle(google_map_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Color: Red, Thickness: 2
    
    # Increment building count and label each building
    building_count += 1
    cv2.putText(google_map_image, f'Building {building_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Save the processed image
output_image_path = "highlighted_buildings_labeled.png"  # Output image path
cv2.imwrite(output_image_path, google_map_image)

print("Highlighted buildings with labels saved successfully to:", output_image_path)
