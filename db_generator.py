import cv2
import mediapipe as mp
import numpy as np
import json
import math

archivo_json = 'hsv.json'

def get_distance(point1, point2):
	x1, y1 = point1
	x2, y2 = point2
	distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
	return distance

def get_mean_point(point1, point2):
	x1, y1 = point1
	x2, y2 = point2
	xm = (x1+x2)//2
	ym = (y1+y2)//2
	return (xm, ym)

with open(archivo_json, 'r') as archivo:
    hsv_values = json.load(archivo)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

img = cv2.imread('img/foto4.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = hands.process(img_rgb)

p0 = (0,0)

if results.multi_hand_landmarks:
	for landmarks in results.multi_hand_landmarks:
	# Draw landmarks on the frame
		height, width, _ = img.shape
		
		wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
		middle_finger_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

		center_x, center_y = int((wrist.x + middle_finger_tip.x) * width / 2), int((wrist.y + middle_finger_tip.y) * height / 2)
		
		margin = 30  # Tratar de encontrar una manera de calcular el margen de forma mas o menos dinámica
		x_min = int(min([landmark.x * width for landmark in landmarks.landmark]) - margin)
		y_min = int(min([landmark.y * height for landmark in landmarks.landmark]) - margin)
		x_max = int(max([landmark.x * width for landmark in landmarks.landmark]) + margin)
		y_max = int(max([landmark.y * height for landmark in landmarks.landmark]) + margin)

		cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
		
		diag_roi = get_distance((x_min, y_min), (x_max, y_max))
		
		wrist_point = (int(wrist.x * img.shape[1]), int(wrist.y * img.shape[0]))
		
		cv2.circle(img, wrist_point, 5, (0, 255, 0), -1)

		roi = img[y_min:y_max, x_min:x_max]
		if roi.size != 0:
			hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			#ret, thresh = cv2.threshold(hsv_img, 130, 255, cv2.THRESH_BINARY_INV)
			
			limite_bajo = np.array([hsv_values['hue_min'], hsv_values['sat_min'], hsv_values['val_min']], dtype=np.uint8)
			limite_alto = np.array([hsv_values['hue_max'], hsv_values['sat_max'], hsv_values['val_max']], dtype=np.uint8)
			
			mask = cv2.inRange(hsv_img, limite_bajo, limite_alto)
			
			gray_mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			ret, thresh = cv2.threshold(gray_mask, 127, 255, 1)
			
			cv2.imshow("thresh", thresh)
			
			contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			height_roi, width_roi = roi.shape[:2]
			center_point_roi = (width_roi // 2, height_roi // 2)

			min_distance = float('inf')
			closest_contour = None

			for cnt in contours:
				M = cv2.moments(cnt)
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])

					distance = np.sqrt((cX - center_point_roi[0])**2 + (cY - center_point_roi[1])**2)

					if distance < min_distance:
						min_distance = distance
						closest_contour = cnt

			if closest_contour is not None:
				epsilon = 0.01 * cv2.arcLength(closest_contour, True)
				approx = cv2.approxPolyDP(closest_contour, epsilon, True)
				
				for point in approx:
					point[0][0] += x_min
					point[0][1] += y_min
					
				hull = cv2.convexHull(closest_contour, returnPoints = False)
				
				hull[::-1].sort(axis=0)
				defects = cv2.convexityDefects(closest_contour, hull)
				
				cv2.drawContours(img, [approx], -1, (0, 255, 0), 1)
				
				if defects is not None:
					fars = [p0,p0,p0,p0]
					fingers = []
					dist_fars = [0,0,0,0]
					dist_fingers = []
					
					count = 0
					count2 = 0
					
					starts = []
					ends = []
					points = []
					for i in range(defects.shape[0]):
						s,e,f,d = defects[i,0]
						
						far = tuple(closest_contour[f][0])
						start  = tuple(closest_contour[s][0])
						end = tuple(closest_contour[e][0])

						far = (far[0] + x_min, far[1] + y_min)
						start = (start[0] + x_min, start[1] + y_min)
						end = (end[0] + x_min, end[1] + y_min)
						
						a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
						b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
						c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
						angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / np.pi

						if angle <= 90:
							starts.append(start)
							ends.append(end)
							cv2.circle(img,far,5,[0,0,255],-1)
							x_far, y_far = far
							x_norm = (x_far-x_min)/(x_max-x_min)
							y_norm = (y_far-y_min)/(y_max-y_min)
							
							fars[count] = (x_norm, y_norm)
							dist_fars[count] = get_distance(wrist_point, far)/diag_roi
							
							count += 1 
					
					if len(starts) == len(ends) and len(starts)>0 and len(starts) < 6:
						points.append(starts[0])
						for i in range(len(starts)-1):
							p = get_mean_point(starts[i+1], ends[i])
							points.append(p)
						points.append(ends[-1])
						
						for point in points:
							cv2.circle(img,point,5,[255,0,0],-1) 
							x_p, y_p = point
							x_norm = (x_p-x_min)/(x_max-x_min)
							y_norm = (y_p-y_min)/(y_max-y_min)
							fingers.append((x_norm, y_norm))
							dist_fingers.append(get_distance(wrist_point, point)/diag_roi)
							
						for i in range(5-len(fingers)):
							fingers.append(p0)
							dist_fingers.append(0)

print(dist_fingers)
print(dist_fars)
							
print(fingers)
print(fars)
cv2.imshow("Hand Tracking", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
