import cv2
import json
import time
import numpy as np

def set_hsv_values():
    # Create a window with sliders for adjusting HSV values
    cv2.namedWindow('Set HSV Values')
    cv2.createTrackbar('Hue Min', 'Set HSV Values', 0, 179, lambda x: None)
    cv2.createTrackbar('Hue Max', 'Set HSV Values', 0, 179, lambda x: None)
    cv2.createTrackbar('Saturation Min', 'Set HSV Values', 0, 255, lambda x: None)
    cv2.createTrackbar('Saturation Max', 'Set HSV Values', 0, 255, lambda x: None)
    cv2.createTrackbar('Value Min', 'Set HSV Values', 0, 255, lambda x: None)
    cv2.createTrackbar('Value Max', 'Set HSV Values', 0, 255, lambda x: None)

    # Open the camera
    cap = cv2.VideoCapture(0)

    changes_saved = False

    while True:

        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        if changes_saved:
            time.sleep(2)
            break

        # Get current HSV values from the sliders
        hue_min = cv2.getTrackbarPos('Hue Min', 'Set HSV Values')
        hue_max = cv2.getTrackbarPos('Hue Max', 'Set HSV Values')
        sat_min = cv2.getTrackbarPos('Saturation Min', 'Set HSV Values')
        sat_max = cv2.getTrackbarPos('Saturation Max', 'Set HSV Values')
        val_min = cv2.getTrackbarPos('Value Min', 'Set HSV Values')
        val_max = cv2.getTrackbarPos('Value Max', 'Set HSV Values')

        # Create an HSV image
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a binary mask using the HSV values
        mask = cv2.inRange(hsv_frame, (hue_min, sat_min, val_min), (hue_max, sat_max, val_max))

        # Apply the mask to the original frame
        segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Display the segmented frame
        cv2.imshow('Set HSV Values', segmented_frame)

        # Save the HSV values to a configuration file if 's' key is pressed
        key = cv2.waitKey(1)
        if key == 27:  # 'ESC' key
            break
        elif key == ord('s'):
            cv2.putText(frame, 'Changes Saved!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            config = {
                'hue_min': hue_min,
                'hue_max': hue_max,
                'sat_min': sat_min,
                'sat_max': sat_max,
                'val_min': val_min,
                'val_max': val_max
            }
            with open('hsv.json', 'w') as config_file:
                json.dump(config, config_file)
            changes_saved = True

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    set_hsv_values()
