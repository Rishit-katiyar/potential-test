import cv2

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

# Apply adaptive thresholding to binarize the image
binary_image = cv2.adaptiveThreshold(
    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

# Find contours of the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding rectangles around buildings and label them
building_count = 0
for contour in contours:
    # Get the bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Filter out small contours that may not be buildings
    if w > 20 and h > 20:
        # Draw the bounding rectangle on the original image
        cv2.rectangle(google_map_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Color: Red, Thickness: 2
        
        # Increment building count and label each building
        building_count += 1
        cv2.putText(google_map_image, f'Building {building_count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Save the processed image
output_image_path = "highlighted_buildings_labeled.png"  # Output image path
cv2.imwrite(output_image_path, google_map_image)

print("Highlighted buildings with labels saved successfully to:", output_image_path)
