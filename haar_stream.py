import cv2

# Carga el modelo Haar Cascade para detección de manos
fist_cascade = cv2.CascadeClassifier('model/fist.xml')

rpalm_cascade = cv2.CascadeClassifier('model/rpalm.xml')
lpalm_cascade = cv2.CascadeClassifier('model/lpalm.xml')

right_cascade = cv2.CascadeClassifier('model/right.xml')
left_cascade = cv2.CascadeClassifier('model/left.xml')

gest_cascade = cv2.CascadeClassifier('model/aGest.xml')

# Crea un objeto VideoCapture para capturar el video desde la webcam (por defecto, la cámara 0)
cap = cv2.VideoCapture(0)

while True:
	# Lee un frame de la webcam
	ret, frame = cap.read()

	if not ret:
		break

	# Convierte el frame a escala de grises
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detecta manos en el frame usando el clasificador Haar Cascade

	rpalms = rpalm_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2)
	#lpalms = lpalm_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2)
		
	for (x, y, w, h) in rpalms:
		frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
		


	
	
	# Muestra el frame con la detección de manos en tiempo real
	cv2.imshow('Face Detection', frame)

	# Rompe el bucle cuando se presiona la tecla 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Libera el objeto VideoCapture y cierra todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()

