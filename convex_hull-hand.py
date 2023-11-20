import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread("img/mano.png", cv2.IMREAD_UNCHANGED))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)

contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

height, width = img.shape[:2]
center_point = (width // 2, height // 2)

min_distance = float('inf')
closest_contour = None

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        distance = np.sqrt((cX - center_point[0])**2 + (cY - center_point[1])**2)

        if distance < min_distance:
            min_distance = distance
            closest_contour = cnt

black = np.zeros_like(img)

if closest_contour is not None:
    epsilon = 0.01 * cv2.arcLength(closest_contour, True)
    approx = cv2.approxPolyDP(closest_contour, epsilon, True)
    cv2.drawContours(black, [approx], 0, (255, 255, 0), 2)

cv2.imshow("hull", black)
cv2.waitKey(0)
cv2.destroyAllWindows()
